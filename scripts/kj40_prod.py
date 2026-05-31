"""
[STAT] kj40 — Første rene kjøring med fikset equations.py (Funn A/B/C)

Funn fra kj39A (beste hittil, RMSE=0.283):
  - psi_R=0.902 (data aksepterer 0.90 med moderat prior)
  - I_R.q4=+0.650 (NB: +0.55) ✅
  - RER reverserer for raskt (q4: −0.641 vs NB −1.00) ❌
  - RMSE=0.283

Strukturelle bugfikser i equations.py (2026-05-31):
  - Funn A: EPS_RP koblet inn i UIP via G0[15,EPS_RP]=-_w (var frakoblet → rho_rp nulleffekt)
  - Funn B: Psi[15,E_prem] fjernet fra UIP (dobbel kontemporær effekt eliminert)
  - Funn C: G1[44,I_STAR] bruker nå rho_iS (var rho_piS — copy-paste-feil)

Strategi kj40:
  - Identisk konfigurasjon med kj39B (phi_PQ=150, gamma_p=0.75, psi_R≈0.90)
  - build_matrices_v3_forward + lambda_pi4=0.0
  - Warm start: kj39A posterior
  - Forventet: rho_rp nå identifisert via RER-dynamikk (FEVD-andel bør øke fra 1.2%)
  - Forventet: RER-persistens kan bedres (risikopremiekanal nå aktiv)
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
LOG     = RESULTS / "kj40_run.log"

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

# ── Warm start fra kj39A ──────────────────────────────────────────────────────
pf = RESULTS / "chain_kj39A_prod_posterior.json"
d  = json.load(open(pf))
s  = d["summary"]
theta_start = np.array([s[n]["mean"] for n in PARAM_NAMES])
post_std    = np.array([max(s[n]["std"], 0.001) for n in PARAM_NAMES])

idx_psiR   = PARAM_NAMES.index('psi_R')
idx_gammap = PARAM_NAMES.index('gamma_p')

theta_start[idx_psiR]   = np.clip(theta_start[idx_psiR], 0.87, 0.93)
theta_start[idx_gammap] = np.clip(theta_start[idx_gammap], 0.65, 0.80)
post_std[idx_psiR]      = 0.010

# ── Prior overrides ────────────────────────────────────────────────────────────
prior_overrides = {
    'psi_R':   ('normal', 0.90, 0.010, 0.85, 0.95),
    'psi_P1':  ('normal', 0.50, 0.20,  0.10, 2.00),
    'gamma_p': ('normal', 0.75, 0.05,  0.55, 0.90),
    'rho_s':   ('normal', 0.03, 0.03,  0.00, 0.15),
    'phi_I1':  ('normal', 0.50, 0.001, 0.40, 0.60),
    'phi_H1':  ('normal', 60.73, 0.001, 60.70, 60.76),
    'rho_A':   ('beta', 5.0, 3.0, 0.01, 0.99),
    'rho_C':   ('beta', 5.0, 3.0, 0.10, 0.99),
    'rho_O':   ('beta', 5.0, 3.0, 0.10, 0.99),
    'rho_Ys':  ('beta', 5.0, 3.0, 0.10, 0.99),
    'rho_rp':  ('beta', 5.0, 3.0, 0.10, 0.99),
    'rho_H':   ('beta', 5.0, 2.0, 0.30, 0.99),
}

irf0 = lag_irf_normalisert(theta_start)
lp0  = log_posterior(theta_start, H, Sv, pre, post,
                     build_fn=_build_fn, prior_overrides=prior_overrides)
print(f"Warm start kj40 (phi_PQ=150, forward λ=0, psi_R={theta_start[idx_psiR]:.3f}, gamma_p={theta_start[idx_gammap]:.3f})")
print(f"  lp0={lp0:.2f}")
if irf0 is not None:
    r0 = nb_score(irf0)
    print(f"  RMSE(korr)={r0:.4f}  PI.q4={irf0[3,PI]:.3f}  I_R.q12={irf0[11,I_R]:.3f}  RER.q1={irf0[0,RER]:.3f}")

import logging
logging.basicConfig(
    level=logging.INFO, format="%(message)s",
    handlers=[logging.FileHandler(LOG, mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger()
logger.info(f"Warm start kj40 (phi_PQ=150, forward λ=0, psi_R={theta_start[idx_psiR]:.3f}, gamma_p={theta_start[idx_gammap]:.3f})")
logger.info(f"lp0={lp0:.2f}")
logger.info("")
logger.info("=================================================================")
logger.info("  NEMO v3_forward — ADAPTIV MCMC kj40")
logger.info("  phi_PQ=150 (kappa_P=0.20)  lambda_pi4=0.0")
logger.info("  BUGFIKSET: EPS_RP koblet inn (Funn A), EPS_PREM enkelt (Funn B)")
logger.info("  psi_R→N(0.90,0.010,[0.85,0.95])")
logger.info("  gamma_p→N(0.75,0.05,[0.55,0.90])")
logger.info("  Warm start: kj39A  Prod=200,000  Burn-in=30,000  seed=40")
logger.info("=================================================================")

# ── MCMC ──────────────────────────────────────────────────────────────────────
chain, lp_chain, meta = adaptive_mcmc_with_monitoring(
    Y_pre=pre, Y_post=post, H=H, Sv=Sv,
    theta_init=theta_start, post_std_init=post_std,
    n_production=200_000, burnin=30_000,
    adapt_every=500, check_every=10_000, max_recalib=10,
    psrf_thr=1.10, ess_pct_thr=0.02,
    scale_init=0.70, seed=40, verbose=True,
    save_prefix=str(RESULTS / "chain_kj40_prod"),
    build_fn=_build_fn, prior_overrides=prior_overrides,
)

# ── Lagre ─────────────────────────────────────────────────────────────────────
np.save(RESULTS / "chain_kj40_prod.npy", chain)
np.save(RESULTS / "chain_kj40_prod_lp.npy", lp_chain)

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
json.dump(posterior, open(RESULTS / "chain_kj40_prod_posterior.json", "w"), indent=2)
json.dump(meta,      open(RESULTS / "chain_kj40_prod_meta.json",      "w"), indent=2)

# ── Evaluering ────────────────────────────────────────────────────────────────
s2 = posterior["summary"]
theta_post = np.array([s2[n]["mean"] for n in PARAM_NAMES])
irf_post   = lag_irf_normalisert(theta_post)
if irf_post is not None:
    rmse_val = nb_score(irf_post)
    by4  = abs(irf_post[3, Y])  / 0.47
    bpi4 = abs(irf_post[3, PI]) / 0.14
    logger.info(f"\n=== kj40 ENDELIG EVALUERING (bugfikset equations.py) ===")
    logger.info(f"RMSE(korr NB)={rmse_val:.4f}  B5: by4={by4:.3f} {'✅' if 0.80<=by4<=1.50 else '❌'}  bpi4={bpi4:.3f} {'✅' if bpi4>=0.35 else '❌'}")
    logger.info(f"  psi_R={s2['psi_R']['mean']:.4f}  gamma_p={s2['gamma_p']['mean']:.4f}  rho_rp={s2['rho_rp']['mean']:.4f}")
    for vn in ['Y','PI','I_R','RER']:
        vals = [round(float(irf_post[h, VAR_IDX[vn]]), 3) for h in [0,3,7,11]]
        logger.info(f"  {vn}: {vals}  (NB: {list(NB_KORR[vn].values())})")
logger.info("\nkj40 fullført.")
