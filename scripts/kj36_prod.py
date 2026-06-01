"""
[STAT] kj36 — Sandkasse: rho_s→0 + psi_R fri (korrigert NB-benchmark)

Funn fra kj35-evaluering mot KORRIGERT NB Figur 1-benchmark:
  - RMSE(kj35, korr) = 0.317  (gammel feil benchmark: 0.154)
  - RER var 3× for liten i gammel benchmark
  - I_R q12 skal være −0.15 (NB-modellens undershoot — umulig med AR(1)-Taylor)

2D-sweep funn (psi_R × rho_s, korrigert benchmark):
  - rho_s=0.00 alltid best: RER q1 fra −0.72 → −1.05
  - psi_R=0.92, rho_s=0.00: RMSE=0.287, B5 ✅ — beste oppnåelige med v3

Strategi kj36:
  - rho_s: N(0.05, 0.05, [0.00, 0.25]) — mot 0 → RER-forbedring
  - psi_R: N(0.90, 0.015, [0.85, 0.97]) — la data finne ~0.92
  - gamma_p: N(0.65, 0.05, [0.40, 0.85]) — behold fra kj35
  - Warm start: kj35 posterior
"""

from __future__ import annotations
import json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nemo.estimation.mcmc import (
    PARAM_NAMES, KM, build_H, build_Sv,
    adaptive_mcmc_with_monitoring, log_posterior, OBS_NAMES,
)
from nemo.model.equations import build_matrices_v3, Y, PI, I_R, RER, E_i
from nemo.model.parameters import Parameters
from nemo.solver.blanchard_kahn import solve, compute_irf

RESULTS = ROOT / "data" / "results"
LOG     = RESULTS / "kj36_run.log"

# ── Data ──────────────────────────────────────────────────────────────────────
datafil  = ROOT / "data/processed/nemo_data_kpi_jae.csv"
obs_df   = pd.read_csv(datafil, index_col=0, parse_dates=True)
obs_kols = ['pi_core_obs' if k == 'pi_obs' else k for k in OBS_NAMES]
pre  = obs_df[obs_df.index <= '2019-12-31'][obs_kols].values
post = obs_df[obs_df.index >= '2022-01-01'][obs_kols].values
H  = build_H()
Sv = build_Sv()

# ── Evaluering (korrigert benchmark) ──────────────────────────────────────────
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
        G0, G1, Psi, Pi_ = build_matrices_v3(p)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            T, R, diag = solve(G0, G1, Psi, Pi_, verbose=False)
        if not diag.get("stable", False): return None
        irf_raw = compute_irf(T, R, E_i, SHOCK, T_periods=20)
        peak = float(np.max(irf_raw[:, I_R]))
        return None if peak <= 0 else irf_raw / peak
    except Exception: return None

def nb_score_korr(irf: np.ndarray) -> float:
    errs = [(irf[HORIZONS[q], VAR_IDX[v]] - NB_KORR[v][q])**2
            for v in NB_KORR for q in NB_KORR[v]]
    return float(np.sqrt(np.mean(errs)))

# ── Warm start fra kj35 ───────────────────────────────────────────────────────
pf = RESULTS / "chain_kj35_prod_posterior.json"
d  = json.load(open(pf))
s  = d["summary"]
theta_start = np.array([s[n]["mean"] for n in PARAM_NAMES])
post_std    = np.array([max(s[n]["std"], 0.001) for n in PARAM_NAMES])

# Juster startverdi for rho_s mot ny prior-senter (0.05 vs kj35's 0.302)
idx_rhos = PARAM_NAMES.index('rho_s')
theta_start[idx_rhos] = 0.05
post_std[idx_rhos]    = 0.05

# Klipp psi_R til [0.85, 0.97]
idx_psiR = PARAM_NAMES.index('psi_R')
theta_start[idx_psiR] = np.clip(theta_start[idx_psiR], 0.86, 0.96)

irf0 = lag_irf_normalisert(theta_start)
lp0  = log_posterior(theta_start, H, Sv, pre, post, build_fn=None, prior_overrides={
    'rho_s':   ('normal', 0.05, 0.05, 0.00, 0.25),
    'psi_R':   ('normal', 0.90, 0.015, 0.85, 0.97),
    'gamma_p': ('normal', 0.65, 0.05,  0.40, 0.85),
})
print(f"Warm start kj36 fra kj35 (rho_s={theta_start[idx_rhos]:.3f}  psi_R={theta_start[idx_psiR]:.3f})")
print(f"  lp0={lp0:.2f}")
if irf0 is not None:
    r0 = nb_score_korr(irf0)
    print(f"  RMSE(korr)={r0:.4f}  RER.q1={irf0[0,RER]:.3f}  I_R.q12={irf0[11,I_R]:.3f}")

# ── Prior overrides ────────────────────────────────────────────────────────────
prior_overrides = {
    'rho_s':   ('normal', 0.05,  0.05,  0.00, 0.25),   # → 0 for RER-forbedring
    'psi_R':   ('normal', 0.90,  0.015, 0.85, 0.97),   # la data finne ~0.92
    'gamma_p': ('normal', 0.65,  0.05,  0.40, 0.85),   # behold fra kj35
    'phi_I1':  ('normal', 0.50,  0.001, 0.40, 0.60),   # fast
    'phi_H1':  ('normal', 60.73, 0.001, 60.70, 60.76), # fast
    'rho_A':   ('beta', 5.0, 3.0, 0.01, 0.99),
    'rho_C':   ('beta', 5.0, 3.0, 0.10, 0.99),
    'rho_O':   ('beta', 5.0, 3.0, 0.10, 0.99),
    'rho_Ys':  ('beta', 5.0, 3.0, 0.10, 0.99),
    'rho_rp':  ('beta', 5.0, 3.0, 0.10, 0.99),
    'rho_H':   ('beta', 5.0, 2.0, 0.30, 0.99),
}

import logging
logging.basicConfig(
    level=logging.INFO, format="%(message)s",
    handlers=[logging.FileHandler(LOG, mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger()
logger.info(f"Warm start kj36 fra kj35 (rho_s→0.05, psi_R={theta_start[idx_psiR]:.3f})")
logger.info(f"lp0={lp0:.2f}")
logger.info("")
logger.info("=================================================================")
logger.info("  NEMO v3 — ADAPTIV MCMC kj36")
logger.info("  rho_s→N(0.05,0.05,[0,0.25])  psi_R→N(0.90,0.015,[0.85,0.97])")
logger.info("  Korrigert NB-benchmark: RER 3× stor, I_R q12=−0.15")
logger.info("  Prod=200,000  Burn-in=30,000  seed=36")
logger.info("=================================================================")

# ── MCMC ──────────────────────────────────────────────────────────────────────
chain, lp_chain, meta = adaptive_mcmc_with_monitoring(
    Y_pre=pre, Y_post=post, H=H, Sv=Sv,
    theta_init=theta_start, post_std_init=post_std,
    n_production=200_000, burnin=30_000,
    adapt_every=500, check_every=10_000, max_recalib=10,
    psrf_thr=1.10, ess_pct_thr=0.02,
    scale_init=0.70, seed=36, verbose=True,
    save_prefix=str(RESULTS / "chain_kj36_prod"),
    build_fn=None, prior_overrides=prior_overrides,
)

# ── Lagre ─────────────────────────────────────────────────────────────────────
np.save(RESULTS / "chain_kj36_prod.npy", chain)
np.save(RESULTS / "chain_kj36_prod_lp.npy", lp_chain)

from nemo.estimation.mcmc import summarize_chain
posterior = summarize_chain(chain, lp_chain, PARAM_NAMES)
json.dump(posterior, open(RESULTS / "chain_kj36_prod_posterior.json", "w"), indent=2)
json.dump(meta,      open(RESULTS / "chain_kj36_prod_meta.json",      "w"), indent=2)

# ── Evaluering ─────────────────────────────────────────────────────────────────
s2 = posterior["summary"]
theta_post = np.array([s2[n]["mean"] for n in PARAM_NAMES])
irf_post   = lag_irf_normalisert(theta_post)
if irf_post is not None:
    rmse = nb_score_korr(irf_post)
    by4  = abs(irf_post[3, Y])  / 0.47
    bpi4 = abs(irf_post[3, PI]) / 0.14
    logger.info(f"\n=== kj36 ENDELIG EVALUERING ===")
    logger.info(f"RMSE(korr NB)={rmse:.4f}  B5: by4={by4:.3f} {'✅' if 0.80<=by4<=1.50 else '❌'}  bpi4={bpi4:.3f}")
    logger.info(f"  psi_R={s2['psi_R']['mean']:.4f}  rho_s={s2['rho_s']['mean']:.4f}  gamma_p={s2['gamma_p']['mean']:.4f}")
    for vn in ['Y','PI','I_R','RER']:
        vals = [round(float(irf_post[h, VAR_IDX[vn]]), 3) for h in [0,3,7,11]]
        logger.info(f"  {vn}: {vals}")
logger.info("\nkj36 fullført.")
