"""
[STAT] kj46 — Fase 2: PLT-kanal (prisnivåmål, Woodford 2003)

Motivasjon:
  kj44/kj45 bekreftet: I_R.q12-problemet (begrensning 6) er strukturelt.
  AR(2) mean-reversion forkastes av data (kj45: psi_R2 → 0).
  PLT-kanal legger til akkumulert prisnivå-gap: i_t reagerer på
  (p_t - p*_t), som gir genuin mean-reversion etter strammende sjokk.

  PE-beslutning 2026-06-02: "Test alt B, men bevar exitmulighet"
  → implementer PLT (NZ_PLT=51), estimer psi_PL fritt.
  Exitstrategi: psi_PL=0 → eksakt v3_forward-atferd.

Endringer fra kj41/kj44:
  - build_matrices_v3_plt: ny builder med P_STAR_GAP (index 50)
  - psi_PL: Normal(0.10, 0.05, [0.00, 0.50]) — ny estimert parameter (21. param)
  - H: build_H_plt() (NZ_PLT=51 kolonner)
  - Warm start: kj41 posterior + psi_PL_start=0.05
  - phi_PQ=150, lambda_pi4=0.0, use_reparam=True (psi_R logit)
  - seed=46, Prod=200 000, Burn-in=30 000

Mål:
  - Avklare om data identifiserer psi_PL > 0
  - Teste om I_R.q12 < 0 er oppnåelig med PLT-kanal
  - PSRF < 1.10, ESS/n > 0.02 (Fase 2-krav)
"""

from __future__ import annotations
import json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Monkey-patching strategi: legg psi_PL til PARAM_PRIORS/PARAM_NAMES i mcmc-modulen
# FØR vi importerer noe annet. Da ser alle mcmc-funksjoner 21 parametere.
import nemo.estimation.mcmc as _mcmc_mod

PSI_PL_PRIOR = ('normal', 0.10, 0.05, 0.00, 0.50)

_mcmc_mod.PARAM_PRIORS['psi_PL'] = PSI_PL_PRIOR
_mcmc_mod.PARAM_NAMES = list(_mcmc_mod.PARAM_PRIORS.keys())
_mcmc_mod.N_PARAMS    = len(_mcmc_mod.PARAM_NAMES)

# Nå importerer vi med 21-element PARAM_NAMES
from nemo.estimation.mcmc import (
    PARAM_NAMES, build_H_plt, build_Sv,
    adaptive_mcmc_with_monitoring, log_posterior, OBS_NAMES,
)
from nemo.model.equations import build_matrices_v3_plt, Y, PI, I_R, RER, E_i, NZ_PLT
from nemo.model.parameters import Parameters
from nemo.solver.blanchard_kahn import solve, compute_irf

RESULTS = ROOT / "data" / "results"
LOG     = RESULTS / "kj46_run.log"

Parameters.phi_PQ = 150.0
LAMBDA_PI4 = 0.0
PSI_PL_START = 0.05

PRIOR_OVERRIDES = {
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
    'psi_PL':  PSI_PL_PRIOR,
}

def _build_fn(p, theta_H: float = 0.05):
    p.lambda_pi4 = LAMBDA_PI4
    return build_matrices_v3_plt(p, theta_H=theta_H, lambda_pi4=LAMBDA_PI4)

# ── Data ──────────────────────────────────────────────────────────────────────
datafil  = ROOT / "data/processed/nemo_data_kpi_jae.csv"
obs_df   = pd.read_csv(datafil, index_col=0, parse_dates=True)
obs_kols = ['pi_core_obs' if k == 'pi_obs' else k for k in OBS_NAMES]
pre  = obs_df[obs_df.index <= '2019-12-31'][obs_kols].values
post = obs_df[obs_df.index >= '2022-01-01'][obs_kols].values
H  = build_H_plt()   # NZ_PLT=51 kolonner
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
    p.phi_PQ = 150.0
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

# ── Warm start fra kj41 ───────────────────────────────────────────────────────
pf = RESULTS / "chain_kj41_prod_posterior.json"
d  = json.load(open(pf))
s  = d["summary"]

theta_start = np.array([
    s.get(n, {}).get("mean", PSI_PL_START if n == "psi_PL" else 0.0)
    for n in PARAM_NAMES
])
post_std = np.array([
    max(s.get(n, {}).get("std", 0.05 if n == "psi_PL" else 0.001), 0.001)
    for n in PARAM_NAMES
])

idx_psiPL = PARAM_NAMES.index('psi_PL')
idx_psiR  = PARAM_NAMES.index('psi_R')
theta_start[idx_psiPL] = PSI_PL_START
post_std[idx_psiPL]    = 0.03

print(f"Warm start kj46: psi_R={theta_start[idx_psiR]:.4f}  psi_PL={theta_start[idx_psiPL]:.4f}")

lp0 = log_posterior(theta_start, H, Sv, pre, post, build_fn=_build_fn,
                    prior_overrides=PRIOR_OVERRIDES)
irf0 = lag_irf_normalisert(theta_start)
print(f"  lp0={lp0:.2f}")
if irf0 is not None:
    r0 = nb_score(irf0)
    print(f"  RMSE={r0:.4f}  I_R=[{irf0[0,I_R]:.3f},{irf0[3,I_R]:.3f},{irf0[7,I_R]:.3f},{irf0[11,I_R]:.3f}]")
    print(f"  psi_PL={theta_start[idx_psiPL]:.3f}  (NB I_R.q12=-0.15)")

import logging
logging.basicConfig(
    level=logging.INFO, format="%(message)s",
    handlers=[logging.FileHandler(LOG, mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger()
logger.info(f"Warm start kj46: psi_R={theta_start[idx_psiR]:.4f}  psi_PL={theta_start[idx_psiPL]:.4f}")
logger.info(f"lp0={lp0:.2f}")
logger.info("")
logger.info("=================================================================")
logger.info("  NEMO v3_forward — ADAPTIV MCMC kj46 (PLT prisnivåmål)")
logger.info("  phi_PQ=150  lambda_pi4=0.0")
logger.info("  use_reparam=True  (logit for psi_R)")
logger.info(f"  psi_PL: Normal(0.10, 0.05, [0.00, 0.50]) — PLT-styrke (21. param)")
logger.info(f"  Warm start: kj41  Prod=200,000  Burn-in=30,000  seed=46")
logger.info(f"  NZ_PLT={NZ_PLT}  (P_STAR_GAP=akkumulert prisnivå-gap)")
logger.info(f"  Mål: avklare om data identifiserer psi_PL > 0 og I_R.q12 < 0")
logger.info("=================================================================")

# ── MCMC ──────────────────────────────────────────────────────────────────────
chain, lp_chain, meta = adaptive_mcmc_with_monitoring(
    Y_pre=pre, Y_post=post, H=H, Sv=Sv,
    theta_init=theta_start, post_std_init=post_std,
    n_production=200_000, burnin=30_000,
    adapt_every=500, check_every=10_000, max_recalib=10,
    psrf_thr=1.10, ess_pct_thr=0.02,
    scale_init=0.70, seed=46, verbose=True,
    save_prefix=str(RESULTS / "chain_kj46_prod"),
    build_fn=_build_fn,
    prior_overrides=PRIOR_OVERRIDES,
    use_reparam=True,
)

# ── Lagre ─────────────────────────────────────────────────────────────────────
np.save(RESULTS / "chain_kj46_prod.npy", chain)
np.save(RESULTS / "chain_kj46_prod_lp.npy", lp_chain)

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
json.dump(posterior, open(RESULTS / "chain_kj46_prod_posterior.json", "w"), indent=2)
json.dump(meta,      open(RESULTS / "chain_kj46_prod_meta.json",      "w"), indent=2)

# ── Evaluering ────────────────────────────────────────────────────────────────
s2 = posterior["summary"]
theta_post = np.array([s2[n]["mean"] for n in PARAM_NAMES])
irf_post   = lag_irf_normalisert(theta_post)
if irf_post is not None:
    rmse_val   = nb_score(irf_post)
    psi_PL_est = s2['psi_PL']['mean']
    psi_R_est  = s2['psi_R']['mean']
    logger.info(f"\n=== kj46 ENDELIG EVALUERING (Fase 2, PLT) ===")
    logger.info(f"RMSE(korr NB)={rmse_val:.4f}")
    logger.info(f"  psi_R ={psi_R_est:.4f}  (sd={s2['psi_R']['std']:.4f})")
    logger.info(f"  psi_PL={psi_PL_est:.4f}  (sd={s2['psi_PL']['std']:.4f})")
    logger.info(f"  I_R.q12(modell)={irf_post[11,I_R]:.3f}  (NB: -0.15)")
    for vn in ['Y','PI','I_R','RER']:
        vals = [round(float(irf_post[h, VAR_IDX[vn]]), 3) for h in [0,3,7,11]]
        logger.info(f"  {vn}: {vals}  (NB: {list(NB_KORR[vn].values())})")

    konklusjon = "IDENTIFISERT" if psi_PL_est > 0.02 else "IKKE identifisert (→ ren inflasjonsmål)"
    logger.info(f"\n  PLT-diagnose: psi_PL={psi_PL_est:.4f} — {konklusjon}")
    if irf_post[11, I_R] < 0:
        logger.info("  I_R.q12 < 0 ✅ — PLT løser begrensning 6!")
    else:
        logger.info(f"  I_R.q12 = {irf_post[11,I_R]:.3f} > 0 — begrensning 6 gjenstår")

logger.info("\nkj46 fullført.")
