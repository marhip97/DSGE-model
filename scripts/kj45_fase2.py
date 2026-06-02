"""
[STAT] kj45 — Fase 2: AR(2) Taylor-regel med estimert psi_R2 (Alt. A2)

Motivasjon:
  kj44 beviste at psi_R presser genuint mot 0.99 (logit-reparam). Å presse
  psi_R høyere forverrer I_R.q12-problemet (begrensning 6): I_R reverserer ikke.
  AR(2)-leddet psi_R2 (PE-godkjent 2026-06-02, NZ 49→50) gir mean-reversion:
    i_t = psi_R·i_{t-1} + psi_R2·i_{t-2} + (1-psi_R-psi_R2)·[respons] + ε_i
  psi_R2 < 0 gir humpe-formet IRF som kan reversere fortegn (NB I_R.q12=-0.15).

Endringer fra kj44:
  - psi_R2 estimeres fritt: Normal(-0.10, 0.05, [-0.40, 0.00])
  - Warm start: kj44 posterior (best konvergens) + psi_R2 start=-0.05
  - use_reparam=True for psi_R beholdes (presser fortsatt mot grensen)
  - phi_PQ=150, lambda_pi4=0.0, build_matrices_v3_forward (nå NZ=50)

Mål:
  - Avklare om data foretrekker mean-reversion (psi_R2 << 0) eller AR(1) (psi_R2≈0)
  - Reprodusere negativ I_R.q12 (NB-benchmark)
  - PSRF < 1.10, ESS/n så høy som mulig
"""

from __future__ import annotations
import json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nemo.estimation.mcmc import (
    PARAM_NAMES, build_H, build_Sv,
    adaptive_mcmc_with_monitoring, log_posterior, OBS_NAMES,
)
from nemo.model.equations import build_matrices_v3_forward, Y, PI, I_R, RER, E_i
from nemo.model.parameters import Parameters
from nemo.solver.blanchard_kahn import solve, compute_irf

RESULTS = ROOT / "data" / "results"
LOG     = RESULTS / "kj45_run.log"

Parameters.phi_PQ = 150.0
LAMBDA_PI4 = 0.0

def _build_fn(p, theta_H: float = 0.05):
    p.lambda_pi4 = LAMBDA_PI4
    return build_matrices_v3_forward(p, theta_H=theta_H, lambda_pi4=LAMBDA_PI4)

# ── Data ──────────────────────────────────────────────────────────────────────
datafil  = ROOT / "data/processed/nemo_data_kpi_jae.csv"
obs_df   = pd.read_csv(datafil, index_col=0, parse_dates=True)
obs_kols = ['pi_core_obs' if k == 'pi_obs' else k for k in OBS_NAMES]
pre  = obs_df[obs_df.index <= '2019-12-31'][obs_kols].values
post = obs_df[obs_df.index >= '2022-01-01'][obs_kols].values
H  = build_H()
Sv = build_Sv()

# ── NB benchmark ─────────────────────────────────────────────────────────────
NB_KORR = {
    "Y":   {"q1": -0.12, "q4": -0.47, "q8": -0.40, "q12": -0.25},
    "PI":  {"q1": -0.03, "q4": -0.14, "q8": -0.22, "q12": -0.22},
    "I_R": {"q1": +1.00, "q4": +0.55, "q8": +0.10, "q12": -0.15},
    "RER": {"q1": -1.50, "q4": -1.00, "q8": -0.50, "q12": -0.20},
}
VAR_IDX  = {"Y": Y, "PI": PI, "I_R": I_R, "RER": RER}
HORIZONS = {"q1": 0, "q4": 3, "q8": 7, "q12": 11}
SHOCK    = 0.0025

def lag_irf_normalisert(theta: np.ndarray) -> np.ndarray | None:
    p = Parameters()
    for i, n in enumerate(PARAM_NAMES):
        if hasattr(p, n): setattr(p, n, float(theta[i]))
    try:
        G0, G1, Psi, Pi_ = _build_fn(p)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            T, R, diag = solve(G0, G1, Psi, Pi_, verbose=False)
        if not diag.get("stable", False): return None
        irf_raw = compute_irf(T, R, E_i, SHOCK, T_periods=20)
        peak = float(np.max(irf_raw[:, I_R]))
        return None if peak <= 0 else irf_raw / peak
    except Exception: return None

def nb_score(irf: np.ndarray) -> float:
    errs = [(irf[HORIZONS[q], VAR_IDX[v]] - NB_KORR[v][q])**2
            for v in NB_KORR for q in NB_KORR[v]]
    return float(np.sqrt(np.mean(errs)))

# ── Warm start fra kj44 (best konvergens) + psi_R2 ────────────────────────────
pf = RESULTS / "chain_kj44_prod_posterior.json"
d  = json.load(open(pf))
s  = d["summary"]
# kj44-posterior mangler psi_R2 — start fra mild mean-reversion (-0.05)
PSI_R2_START = -0.05
theta_start = np.array([s.get(n, {}).get("mean", PSI_R2_START if n == "psi_R2" else 0.0)
                        for n in PARAM_NAMES])
post_std    = np.array([max(s.get(n, {}).get("std", 0.05), 0.001) for n in PARAM_NAMES])

idx_psiR  = PARAM_NAMES.index('psi_R')
idx_psiR2 = PARAM_NAMES.index('psi_R2')
theta_start[idx_psiR2] = PSI_R2_START
post_std[idx_psiR2]    = 0.05

# Prior overrides — identisk med kj44 (psi_R fri via reparam, psi_R2 fri via default).
prior_overrides = {
    'psi_P1':  ('normal', 0.60, 0.15,  0.10, 2.00),
    'gamma_p': ('normal', 0.75, 0.05,  0.55, 0.90),
    'rho_s':   ('normal', 0.03, 0.03,  0.00, 0.15),
    'rho_rp':  ('normal', 0.33, 0.10,  0.05, 0.65),
    'phi_I1':  ('normal', 0.50, 0.001, 0.40, 0.60),
    'phi_H1':  ('normal', 60.73, 0.001, 60.70, 60.76),
    'rho_A':   ('beta', 5.0, 3.0, 0.01, 0.99),
    'rho_C':   ('beta', 5.0, 3.0, 0.10, 0.99),
    'rho_O':   ('beta', 5.0, 3.0, 0.10, 0.99),
    'rho_Ys':  ('beta', 5.0, 3.0, 0.10, 0.99),
    'rho_H':   ('beta', 5.0, 2.0, 0.30, 0.99),
}
print(f"Warm start kj45: psi_R={theta_start[idx_psiR]:.4f}  psi_R2={theta_start[idx_psiR2]:.4f}")

lp0 = log_posterior(theta_start, H, Sv, pre, post, build_fn=_build_fn,
                    prior_overrides=prior_overrides)
irf0 = lag_irf_normalisert(theta_start)
print(f"  lp0={lp0:.2f}")
if irf0 is not None:
    r0 = nb_score(irf0)
    print(f"  RMSE={r0:.4f}  I_R=[{irf0[0,I_R]:.3f},{irf0[3,I_R]:.3f},{irf0[7,I_R]:.3f},{irf0[11,I_R]:.3f}]")

import logging
logging.basicConfig(
    level=logging.INFO, format="%(message)s",
    handlers=[logging.FileHandler(LOG, mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger()
logger.info(f"Warm start kj45: psi_R={theta_start[idx_psiR]:.4f}  psi_R2={theta_start[idx_psiR2]:.4f}")
logger.info(f"lp0={lp0:.2f}")
logger.info("")
logger.info("=================================================================")
logger.info("  NEMO v3_forward — ADAPTIV MCMC kj45 (AR(2) Taylor, NZ=50)")
logger.info("  phi_PQ=150  lambda_pi4=0.0")
logger.info("  use_reparam=True  (logit for psi_R)")
logger.info("  psi_R2: Normal(-0.10, 0.05, [-0.40, 0.00]) — AR(2) mean-reversion")
logger.info("  Warm start: kj44  Prod=200,000  Burn-in=30,000  seed=45")
logger.info("  Mål: avklar mean-reversion vs AR(1); reproduser I_R.q12<0")
logger.info("=================================================================")

# ── MCMC ──────────────────────────────────────────────────────────────────────
chain, lp_chain, meta = adaptive_mcmc_with_monitoring(
    Y_pre=pre, Y_post=post, H=H, Sv=Sv,
    theta_init=theta_start, post_std_init=post_std,
    n_production=200_000, burnin=30_000,
    adapt_every=500, check_every=10_000, max_recalib=10,
    psrf_thr=1.10, ess_pct_thr=0.02,
    scale_init=0.70, seed=45, verbose=True,
    save_prefix=str(RESULTS / "chain_kj45_prod"),
    build_fn=_build_fn,
    prior_overrides=prior_overrides,
    use_reparam=True,
)

# ── Lagre ─────────────────────────────────────────────────────────────────────
np.save(RESULTS / "chain_kj45_prod.npy", chain)
np.save(RESULTS / "chain_kj45_prod_lp.npy", lp_chain)

import importlib
mcmc_mod = importlib.import_module("nemo.estimation.mcmc")
summarize_fn = getattr(mcmc_mod, "summarize_chain", None)
if summarize_fn:
    posterior = summarize_fn(chain, lp_chain, PARAM_NAMES)
else:
    posterior = {
        "summary": {
            n: {
                "mean": float(chain[:, i].mean()),
                "std":  float(chain[:, i].std()),
                "q025": float(np.percentile(chain[:, i], 2.5)),
                "q975": float(np.percentile(chain[:, i], 97.5)),
            }
            for i, n in enumerate(PARAM_NAMES)
        }
    }
json.dump(posterior, open(RESULTS / "chain_kj45_prod_posterior.json", "w"), indent=2)
json.dump(meta,      open(RESULTS / "chain_kj45_prod_meta.json",      "w"), indent=2)

# ── Evaluering ────────────────────────────────────────────────────────────────
s2 = posterior["summary"]
theta_post = np.array([s2[n]["mean"] for n in PARAM_NAMES])
irf_post   = lag_irf_normalisert(theta_post)
if irf_post is not None:
    rmse_val = nb_score(irf_post)
    by4  = abs(irf_post[3, Y])  / 0.47
    bpi4 = abs(irf_post[3, PI]) / 0.14
    iR_q12 = float(irf_post[11, I_R])
    logger.info(f"\n=== kj45 ENDELIG EVALUERING (Fase 2, AR(2) Taylor) ===")
    logger.info(f"RMSE(korr NB)={rmse_val:.4f}  B5: by4={by4:.3f} {'✅' if 0.80<=by4<=1.50 else '❌'}  bpi4={bpi4:.3f} {'✅' if bpi4>=0.35 else '❌'}")
    logger.info(f"  psi_R={s2['psi_R']['mean']:.4f} (sd={s2['psi_R']['std']:.4f})")
    logger.info(f"  psi_R2={s2['psi_R2']['mean']:.4f} (sd={s2['psi_R2']['std']:.4f})")
    logger.info(f"  I_R.q12={iR_q12:.3f}  (NB: -0.15)  {'✅ reverserer' if iR_q12 < 0 else '❌ ingen reversering'}")
    for vn in ['Y','PI','I_R','RER']:
        vals = [round(float(irf_post[h, VAR_IDX[vn]]), 3) for h in [0,3,7,11]]
        logger.info(f"  {vn}: {vals}  (NB: {list(NB_KORR[vn].values())})")

# Diagnose: foretrekker data mean-reversion?
psi_R2_mean = s2["psi_R2"]["mean"]
psi_R2_q975 = s2["psi_R2"].get("q975", s2["psi_R2"].get("p95", 0.0))
if psi_R2_mean < -0.05 and psi_R2_q975 < 0.0:
    diagnose = "DATA FORETREKKER MEAN-REVERSION (psi_R2 klart < 0)"
elif psi_R2_mean < -0.02:
    diagnose = "svak mean-reversion (psi_R2 mildt negativ)"
else:
    diagnose = "data foretrekker AR(1) (psi_R2 ≈ 0)"
logger.info(f"\n  psi_R2 posterior mean={psi_R2_mean:.4f}")
logger.info(f"  Diagnose: {diagnose}")
logger.info("\nkj45 fullført.")
