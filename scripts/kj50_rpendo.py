"""
kj50 — Fase 3: endogen risikopremie i UIP (Alt A)

PE-godkjent 2026-06-04 (transmisjonsdiagnose):
  - build_matrices_rpendo: ny tilstand RP_ENDO, UIP rad 15 utvidet
  - kappa_rp_endo fri (Normal(0.20,0.15,[0.0,1.0])) — exit ved 0
  - rho_rp_endo fri  (Beta(2,2,[0.05,0.95]))
  - Øvrig transmisjon = kj41-referanse: phi_I1=0.50 fast, rho_s=0.00,
    phi_O fri, sigma_rp=0.006 fast, phi_PQ=150, lambda_pi4=0.0
  - N_PARAMS=21 (kj41-19 + kappa_rp_endo + rho_rp_endo)

Formål: avgjøre om DATA støtter en endogen risikopremie, og om den lukker
RER-IRF-gapet mot NB Figur 1 (håndkalibrert test ga 16pt-RMSE 0.295→0.263).

Warm start: kj41 posterior. Nye parametere startes på prior-mean.

Bruk:
  python scripts/kj50_rpendo.py [n_production] [n_burnin]
  (default 200000 / 30000; gi små tall for pilot, f.eks. 3000 500)
"""

from __future__ import annotations
import sys
import json
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

from nemo.estimation.mcmc import (
    PARAM_NAMES, N_PARAMS, KM, PARAM_PRIORS,
    build_H_rpendo, build_Sv, OBS_NAMES,
    adaptive_mcmc_with_monitoring,
)
from nemo.model.equations import build_matrices_rpendo, I_R, Y, PI, RER, E_i
from nemo.model.parameters import Parameters
from nemo.solver.blanchard_kahn import solve, compute_irf

assert 'kappa_rp_endo' in PARAM_NAMES, "kappa_rp_endo mangler i PARAM_NAMES"
assert 'rho_rp_endo'  in PARAM_NAMES, "rho_rp_endo mangler i PARAM_NAMES"
assert 'phi_I1' not in PARAM_NAMES,   "phi_I1 skal være fast (0.50)"
assert N_PARAMS == 21, f"N_PARAMS={N_PARAMS}, forventet 21"
print(f"PARAM_NAMES ({N_PARAMS}): {PARAM_NAMES}")

# ── Modellbygg-funksjon (kj41-kalibrering + endogen premie) ───────────────────
LAMBDA_PI4_FIXED = 0.0
PHI_PQ_FIXED     = 150.0

def _build_fn(p, theta_H: float = 0.05):
    p.phi_PQ = PHI_PQ_FIXED
    return build_matrices_rpendo(p, theta_H=theta_H, lambda_pi4=LAMBDA_PI4_FIXED)

# ── Observasjon og data ───────────────────────────────────────────────────────
H  = build_H_rpendo()
Sv = build_Sv()
datafil = ROOT / "data/processed/nemo_data_kpi_jae.csv"
obs_df  = pd.read_csv(datafil, index_col=0, parse_dates=True)
obs_kols = ['pi_core_obs' if k == 'pi_obs' else k for k in OBS_NAMES]
Y_pre  = obs_df[obs_df.index <= '2019-12-31'][obs_kols].values
Y_post = obs_df[obs_df.index >= '2022-01-01'][obs_kols].values
print(f"Data: Y_pre={Y_pre.shape}, Y_post={Y_post.shape}")

# ── Warm start fra kj41 ────────────────────────────────────────────────────────
with open(ROOT / "data/results/chain_kj41_prod_posterior.json") as f:
    kj41 = json.load(f)["summary"]

theta_start   = np.array([KM.get(n, 0.5) for n in PARAM_NAMES])
post_std_init = np.ones(N_PARAMS) * 0.05
for i, name in enumerate(PARAM_NAMES):
    if name in kj41:
        theta_start[i]   = float(kj41[name]["mean"])
        post_std_init[i] = float(kj41[name]["std"])
# Nye parametere: start på prior-mean
theta_start[PARAM_NAMES.index('kappa_rp_endo')] = 0.20
theta_start[PARAM_NAMES.index('rho_rp_endo')]   = 0.50
post_std_init[PARAM_NAMES.index('kappa_rp_endo')] = 0.08
post_std_init[PARAM_NAMES.index('rho_rp_endo')]   = 0.10

# Klipp til prior-grenser
for i, name in enumerate(PARAM_NAMES):
    lb, ub = float(PARAM_PRIORS[name][-2]), float(PARAM_PRIORS[name][-1])
    theta_start[i] = float(np.clip(theta_start[i], lb + 1e-6, ub - 1e-6))

print("\nWarm start:")
for i, n in enumerate(PARAM_NAMES):
    src = "kj41" if n in kj41 else ("prior" if n in ('kappa_rp_endo','rho_rp_endo') else "KM")
    print(f"  {n:14s} = {theta_start[i]:.4f}  [std={post_std_init[i]:.4f}, {src}]")

# ── MCMC ───────────────────────────────────────────────────────────────────────
N_PROD   = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
N_BURNIN = int(sys.argv[2]) if len(sys.argv) > 2 else 30_000
SEED     = 50
SAVE_PREFIX = str(ROOT / "data/results/chain_kj50_prod")

print(f"\n=== kj50 MCMC START (n_prod={N_PROD}, burnin={N_BURNIN}) ===")
t0 = time.time()
chain, lp_chain, meta = adaptive_mcmc_with_monitoring(
    Y_pre, Y_post, H, Sv,
    theta_init=theta_start, post_std_init=post_std_init,
    n_production=N_PROD, burnin=N_BURNIN,
    adapt_every=500, check_every=10_000, max_recalib=5,
    psrf_thr=1.10, ess_pct_thr=0.02, scale_init=0.676,
    seed=SEED, verbose=True, save_prefix=SAVE_PREFIX,
    use_reparam=True, block_indices=None,
    build_fn=_build_fn, prior_overrides=None,
)
print(f"\nMCMC fullført på {(time.time()-t0)/60:.1f} min")

# ── Evaluering ─────────────────────────────────────────────────────────────────
posterior_path = str(ROOT / "data/results/chain_kj50_prod_posterior.json")
try:
    with open(posterior_path) as f:
        summary = json.load(f)["summary"]
except FileNotFoundError:
    print("Posterior JSON ikke funnet — kjøring avbrutt tidlig?")
    raise SystemExit(1)

p_post = Parameters()
p_post.phi_PQ = PHI_PQ_FIXED
for name in PARAM_NAMES:
    if name in summary:
        setattr(p_post, name, float(summary[name]["mean"]))

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    G0, G1, Psi, Pi = _build_fn(p_post)
    T, R, d = solve(G0, G1, Psi, Pi, verbose=False)
irf = compute_irf(T, R, E_i, 0.0025, T_periods=13)
peak = float(np.max(irf[:, I_R]))
irf_n = irf / peak if peak > 0 else irf
def _q(var): return [float(irf_n[q-1, var]) for q in [1,4,8,12]]
nb = {"Y":[-0.12,-0.47,-0.40,-0.25],"PI":[-0.03,-0.14,-0.22,-0.22],
      "I_R":[1.00,0.55,0.10,-0.15],"RER":[-1.50,-1.00,-0.50,-0.20]}
mod = {"Y":_q(Y),"PI":_q(PI),"I_R":_q(I_R),"RER":_q(RER)}
pts = [(m,n) for k in nb for m,n in zip(mod[k],nb[k])]
rmse = float(np.sqrt(np.mean([(m-n)**2 for m,n in pts])))

print(f"\n=== kj50 ENDELIG EVALUERING (endogen risikopremie) ===")
print(f"RMSE(16pt NB) = {rmse:.4f}   (kj41: 0.295 mot samme 16pt; håndkalib. Alt A: 0.263)")
for k in ["I_R","Y","PI","RER"]:
    print(f"  {k:4}: {[f'{v:+.3f}' for v in mod[k]]}  NB {nb[k]}")
print(f"  kappa_rp_endo = {summary.get('kappa_rp_endo',{}).get('mean',float('nan')):.4f}"
      f"  (±{summary.get('kappa_rp_endo',{}).get('std',float('nan')):.4f})")
print(f"  rho_rp_endo   = {summary.get('rho_rp_endo',{}).get('mean',float('nan')):.4f}"
      f"  (±{summary.get('rho_rp_endo',{}).get('std',float('nan')):.4f})")
print(f"  psi_R = {summary.get('psi_R',{}).get('mean',float('nan')):.4f}")
print(f"  BK-stabil: {d.get('stable')}, max|eig(T)|={d.get('max_eig_T',float('nan')):.6f}")
print("kj50 fullført.")
