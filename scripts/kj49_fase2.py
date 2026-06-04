"""
kj49 — Fase 2: phi_I1 fast=0.50 (B5-passing) + phi_O fri

PE-godkjent 2026-06-03:
  - phi_I1 kalibreres fast=0.50 (B5-passing; kj31/kj41 posterior ≈ 0.50)
  - phi_O fri (Normal(0.15,0.10,[0.01,0.80])) — beholdes fra kj47/kj48
  - rho_s fast=0.00 (beholdes fra kj47)
  - N_PARAMS=19 (phi_I1 fjernet fra estimering)

Formål: ren test av phi_O-effekten under realistiske investeringsdynamikker.
kj47/kj48 viste at phi_I1 kollapser til nedre grense (0.10) når den estimeres fritt,
noe som gir urealistisk IRF (BNP-respons ~10× NB). kj41 (beste baseline, RMSE=0.277)
brukte phi_I1≈0.50. kj49 = kj41 + phi_O fri → isolert test av phi_O-bidraget.

Warm start: kj47 posterior (bevarer phi_O≈0.255, rho_O≈0.108).
Modell: build_matrices_v3_forward (NZ=50, lambda_pi4=0.0, phi_PQ=150). N_PARAMS=19.
"""

from __future__ import annotations
import json
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

from nemo.estimation.mcmc import (
    PARAM_NAMES, N_PARAMS, KM, PARAM_PRIORS,
    build_H, build_Sv, OBS_NAMES,
    adaptive_mcmc_with_monitoring,
)
from nemo.model.equations import build_matrices_v3_forward
from nemo.model.parameters import Parameters
from nemo.solver.blanchard_kahn import solve, compute_irf
from nemo.model.equations import I_R, Y, PI, RER, E_i

assert 'phi_O' in PARAM_NAMES, f"phi_O mangler i PARAM_NAMES: {PARAM_NAMES}"
assert 'phi_I1' not in PARAM_NAMES, f"phi_I1 skal være kalibrert (ikke i PARAM_NAMES)"
assert 'rho_s' not in PARAM_NAMES, f"rho_s skal være kalibrert (ikke i PARAM_NAMES)"
assert N_PARAMS == 19, f"N_PARAMS={N_PARAMS}, forventet 19"
print(f"PARAM_NAMES ({N_PARAMS}): {PARAM_NAMES}")

# ── Modellbygg-funksjon ──────────────────────────────────────────────────────
LAMBDA_PI4_FIXED = 0.0
PHI_PQ_FIXED     = 150.0

def _build_fn(p, theta_H: float = 0.05):
    """Bygger matriser med faste kalibreringer for kj49."""
    p.phi_PQ = PHI_PQ_FIXED
    return build_matrices_v3_forward(p, theta_H=theta_H, lambda_pi4=LAMBDA_PI4_FIXED)

# ── Observasjonslikning og støy ─────────────────────────────────────────────
H    = build_H()
Sv   = build_Sv()

# ── Data ─────────────────────────────────────────────────────────────────────
datafil  = ROOT / "data/processed/nemo_data_kpi_jae.csv"
obs_df   = pd.read_csv(datafil, index_col=0, parse_dates=True)
obs_kols = ['pi_core_obs' if k == 'pi_obs' else k for k in OBS_NAMES]
Y_pre  = obs_df[obs_df.index <= '2019-12-31'][obs_kols].values
Y_post = obs_df[obs_df.index >= '2022-01-01'][obs_kols].values
print(f"Data: Y_pre={Y_pre.shape}, Y_post={Y_post.shape}")

# ── Warm start fra kj47 ───────────────────────────────────────────────────────
# kj47 har phi_O≈0.255 identifisert — bevares. phi_I1 er nå fast=0.50 (ikke i theta).
_post_path = "data/results/chain_kj47_prod_posterior.json"
with open(_post_path) as f:
    kj47_summary = json.load(f)["summary"]

theta_start   = np.array([KM.get(n, 0.5) for n in PARAM_NAMES])
post_std_init = np.ones(N_PARAMS) * 0.05

for i, name in enumerate(PARAM_NAMES):
    if name in kj47_summary:
        theta_start[i]   = float(kj47_summary[name]["mean"])
        post_std_init[i] = float(kj47_summary[name]["std"])

# Klipp startverdi til prior-grenser
for i, name in enumerate(PARAM_NAMES):
    spec = PARAM_PRIORS[name]
    lb, ub = float(spec[-2]), float(spec[-1])
    old = theta_start[i]
    theta_start[i] = float(np.clip(theta_start[i], lb + 1e-6, ub - 1e-6))
    if abs(theta_start[i] - old) > 1e-8:
        print(f"  KLIPPING {name}: {old:.4f} → {theta_start[i]:.4f} (grenser [{lb},{ub}])")

print(f"\nWarm start (kj47 posterior, phi_I1 fast=0.50):")
for i, n in enumerate(PARAM_NAMES):
    src = "kj47" if n in kj47_summary else "KM"
    print(f"  {n:12s} = {theta_start[i]:.4f}  [std={post_std_init[i]:.4f}, kilde={src}]")

# ── MCMC-kjøring ──────────────────────────────────────────────────────────────
SEED        = 49
N_PROD      = 200_000
N_BURNIN    = 30_000
SAVE_PREFIX = str(ROOT / "data/results/chain_kj49_prod")

print(f"\n=== kj49 MCMC START ===")
print(f"  seed={SEED}, n_prod={N_PROD}, n_burnin={N_BURNIN}")
print(f"  phi_I1 fast=0.50, phi_O fri, rho_s fast=0.00, N_PARAMS={N_PARAMS}")
print(f"  build_fn=build_matrices_v3_forward, lambda_pi4={LAMBDA_PI4_FIXED}, phi_PQ={PHI_PQ_FIXED}")
print()

t0 = time.time()

chain, lp_chain, meta = adaptive_mcmc_with_monitoring(
    Y_pre, Y_post, H, Sv,
    theta_init    = theta_start,
    post_std_init = post_std_init,
    n_production  = N_PROD,
    burnin        = N_BURNIN,
    adapt_every   = 500,
    check_every   = 10_000,
    max_recalib   = 5,
    psrf_thr      = 1.10,
    ess_pct_thr   = 0.02,
    scale_init    = 0.676,
    seed          = SEED,
    verbose       = True,
    save_prefix   = SAVE_PREFIX,
    use_reparam   = True,
    block_indices = None,
    build_fn      = _build_fn,
    prior_overrides = None,
)

print(f"\nMCMC fullført på {(time.time()-t0)/60:.1f} min")

# ── Endelig evaluering ────────────────────────────────────────────────────────
posterior_path = str(ROOT / "data/results/chain_kj49_prod_posterior.json")
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
peak_IR = float(np.max(irf[:, I_R]))
irf_n = irf / peak_IR if peak_IR > 0 else irf

def _q(var, qs): return [float(irf_n[q-1, var]) for q in qs]

nb_Y   = [-0.12, -0.47, -0.40, -0.25]
nb_PI  = [-0.03, -0.14, -0.22, -0.22]
nb_IR  = [+1.00, +0.55, +0.10, -0.15]
nb_RER = [-1.50, -1.00, -0.50, -0.20]

mod_Y   = _q(Y,   [1,4,8,12])
mod_PI  = _q(PI,  [1,4,8,12])
mod_IR  = _q(I_R, [1,4,8,12])
mod_RER = _q(RER, [1,4,8,12])

pts  = [(m,n) for mv,nv in zip([mod_Y,mod_PI,mod_IR,mod_RER],[nb_Y,nb_PI,nb_IR,nb_RER])
        for m,n in zip(mv,nv)]
rmse = float(np.sqrt(np.mean([(m-n)**2 for m,n in pts])))

psi_R_post  = float(summary.get('psi_R',  {}).get('mean', float('nan')))
phi_O_post  = float(summary.get('phi_O',  {}).get('mean', float('nan')))
rho_O_post  = float(summary.get('rho_O',  {}).get('mean', float('nan')))
phi_I2_post = float(summary.get('phi_I2', {}).get('mean', float('nan')))

print(f"\n=== kj49 ENDELIG EVALUERING (phi_I1=0.50 fast + phi_O fri) ===")
print(f"RMSE(korr NB)={rmse:.4f}  (kj41: 0.277, kj47/kj48: 0.603)")
print(f"  psi_R  = {psi_R_post:.4f}")
print(f"  phi_O  = {phi_O_post:.4f}  (K&M: 0.15, kj47: 0.255)")
print(f"  rho_O  = {rho_O_post:.4f}  (K&M: 0.874, kj47: 0.108)")
print(f"  phi_I2 = {phi_I2_post:.4f}  (K&M: 165.66)")
print(f"  phi_I1 = 0.50 (FAST — ikke estimert)")
print(f"  Y:   {[f'{v:.3f}' for v in mod_Y]}  (NB: {nb_Y})")
print(f"  PI:  {[f'{v:.3f}' for v in mod_PI]}  (NB: {nb_PI})")
print(f"  I_R: {[f'{v:.3f}' for v in mod_IR]}  (NB: {nb_IR})")
print(f"  RER: {[f'{v:.3f}' for v in mod_RER]}  (NB: {nb_RER})")
print(f"  I_R.q12(modell)={mod_IR[3]:.3f}  (NB: -0.15)")
print(f"  BK-stabil: {d.get('stable')}, max|eig(T)|={d.get('max_eig_T',float('nan')):.6f}")
print()
print(f"  phi_O-diagnose: {'HEVET (>K&M 0.15)' if phi_O_post>0.15 else 'LAV (≤K&M)'}")
print(f"  RMSE vs kj41: {'BEDRE' if rmse < 0.277 else 'VERRE' if rmse > 0.290 else 'TILSVARENDE'}")
print()
print("kj49 fullført.")
