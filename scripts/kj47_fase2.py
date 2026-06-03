"""
kj47 — Fase 2: Blokksampling + phi_O fri + rho_O-prior strammet

PE-godkjent 2026-06-03 (PE_eskalering_fase2_sampler.md + PE_eskalering_fase3_oljepriskanal.md):
  - rho_s kalibrert fast = 0.00 (kj46: 0.003 ± 0.003 — degenerert)
  - rho_O-prior Beta(6,1.5,[0.50,0.9995]) — mot Brent-empirisk persistens
  - phi_O estimert fritt (prior Normal(0.15,0.10,[0.01,0.80]))
  - Blokksampling for rho-klusteret {rho_A,rho_C,rho_O,rho_Ys,rho_rp} (MwG)

Warm start: kj41 posterior (RMSE=0.2771, beste baseline).
Modell: build_matrices_v3_forward (NZ=50, lambda_pi4=0.0, phi_PQ=150).
N_PARAMS=20 (rho_s→phi_O, phi_H1 beholder plass 19).
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
    PARAM_NAMES, N_PARAMS, KM,
    build_H, build_Sv, OBS_NAMES,
    adaptive_mcmc_with_monitoring,
)
from nemo.model.equations import build_matrices_v3_forward
from nemo.model.parameters import Parameters
from nemo.solver.blanchard_kahn import solve, compute_irf
from nemo.model.equations import I_R, Y, PI, RER, E_i, NZ

assert 'phi_O' in PARAM_NAMES, f"phi_O mangler i PARAM_NAMES: {PARAM_NAMES}"
assert 'rho_s' not in PARAM_NAMES, f"rho_s skal være kalibrert (ikke i PARAM_NAMES)"
assert N_PARAMS == 20, f"N_PARAMS={N_PARAMS}, forventet 20"
print(f"PARAM_NAMES ({N_PARAMS}): {PARAM_NAMES}")

# ── Modellbygg-funksjon ──────────────────────────────────────────────────────
LAMBDA_PI4_FIXED = 0.0
PHI_PQ_FIXED     = 150.0

def _build_fn(p, theta_H: float = 0.05):
    """Bygger matriser med faste kalibreringer for kj47."""
    p.phi_PQ = PHI_PQ_FIXED
    return build_matrices_v3_forward(p, theta_H=theta_H, lambda_pi4=LAMBDA_PI4_FIXED)

# ── Observasjonslikning og støy ─────────────────────────────────────────────
H    = build_H()    # (14×50)
Sv   = build_Sv()   # diagonalt, fra OBS_NAMES målefeil

# ── Data ─────────────────────────────────────────────────────────────────────
datafil  = ROOT / "data/processed/nemo_data_kpi_jae.csv"
obs_df   = pd.read_csv(datafil, index_col=0, parse_dates=True)
obs_kols = ['pi_core_obs' if k == 'pi_obs' else k for k in OBS_NAMES]
Y_pre  = obs_df[obs_df.index <= '2019-12-31'][obs_kols].values
Y_post = obs_df[obs_df.index >= '2022-01-01'][obs_kols].values
print(f"Data: Y_pre={Y_pre.shape}, Y_post={Y_post.shape}")

# ── Warm start fra kj41 ───────────────────────────────────────────────────────
_post_path = "data/results/chain_kj41_prod_posterior.json"
with open(_post_path) as f:
    kj41_summary = json.load(f)["summary"]

# Startverdier: K&M som fallback
theta_start = np.array([KM.get(n, 0.5) for n in PARAM_NAMES])
post_std_init = np.ones(N_PARAMS) * 0.05

for i, name in enumerate(PARAM_NAMES):
    if name in kj41_summary:
        theta_start[i] = float(kj41_summary[name]["mean"])
        post_std_init[i] = float(kj41_summary[name]["std"])

# phi_O var ikke estimert i kj41 — start på K&M-verdi
phi_O_idx = PARAM_NAMES.index('phi_O')
theta_start[phi_O_idx] = 0.15
post_std_init[phi_O_idx] = 0.05

print(f"\nWarm start (kj41 posterior + phi_O=0.15):")
for i, n in enumerate(PARAM_NAMES):
    src = "kj41" if n in kj41_summary else "KM"
    print(f"  {n:12s} = {theta_start[i]:.4f}  [std={post_std_init[i]:.4f}, kilde={src}]")

# ── Blokkindekser (Metropolis-within-Gibbs) ──────────────────────────────────
# Blokk A: rho-kluster {rho_A,rho_C,rho_O,rho_Ys,rho_rp} — joint proposal
# Blokk B-T: resterende 15 parametre — singleton proposals (tilsvarer komponentvis MH)
RHO_BLOCK = [PARAM_NAMES.index(n) for n in ['rho_A','rho_C','rho_O','rho_Ys','rho_rp']]
SINGLETON_BLOCKS = [[i] for i in range(N_PARAMS) if i not in RHO_BLOCK]
BLOCK_INDICES = [RHO_BLOCK] + SINGLETON_BLOCKS

print(f"\nBlokkstruktur:")
print(f"  Rho-kluster (joint):   {[PARAM_NAMES[i] for i in RHO_BLOCK]}")
print(f"  Singleton-blokker:     {len(SINGLETON_BLOCKS)} parametre")
print(f"  Totalt blokker/trekk:  {len(BLOCK_INDICES)}")

# ── Prior-overrides (eksplisitt for logging) ──────────────────────────────────
# phi_O og rho_O er nå i global PARAM_PRIORS — ingen overrides nødvendig.
# Dokumentert i mcmc.py.

# ── MCMC-kjøring ──────────────────────────────────────────────────────────────
SEED        = 47
N_PROD      = 200_000
N_BURNIN    = 30_000
SAVE_PREFIX = str(ROOT / "data/results/chain_kj47_prod")

print(f"\n=== kj47 MCMC START ===")
print(f"  seed={SEED}, n_prod={N_PROD}, n_burnin={N_BURNIN}")
print(f"  phi_O estimert, rho_O strammet prior, rho_s fast=0.00")
print(f"  blokksampling: rho-kluster ({len(RHO_BLOCK)} dim) + {len(SINGLETON_BLOCKS)} singletons")
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
    use_reparam   = True,        # logit for psi_R (kj44+)
    block_indices = BLOCK_INDICES,
    build_fn      = _build_fn,
    prior_overrides = None,
)

print(f"\nMCMC fullført på {(time.time()-t0)/60:.1f} min")

# ── Endelig evaluering ────────────────────────────────────────────────────────
posterior_path = str(ROOT / "data/results/chain_kj47_prod_posterior.json")
try:
    with open(posterior_path) as f:
        summary = json.load(f)["summary"]
except FileNotFoundError:
    print("Posterior JSON ikke funnet — kjøring avbrutt tidlig?")
    raise SystemExit(1)

# Posterior mean → Parameters-instans
p_post = Parameters()
p_post.phi_PQ = PHI_PQ_FIXED
for name in PARAM_NAMES:
    if name in summary:
        setattr(p_post, name, float(summary[name]["mean"]))

# IRF ved posterior mean
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    G0, G1, Psi, Pi = _build_fn(p_post)
    T, R, d = solve(G0, G1, Psi, Pi, verbose=False)

irf = compute_irf(T, R, E_i, 0.0025, T_periods=13)

# NB-benchmark normering
peak_IR = float(np.max(irf[:, I_R]))
if peak_IR > 0:
    irf_n = irf / peak_IR
else:
    irf_n = irf

def _q(var, qs): return [float(irf_n[q-1, var]) for q in qs]

nb_Y   = [-0.12, -0.47, -0.40, -0.25]
nb_PI  = [-0.03, -0.14, -0.22, -0.22]
nb_IR  = [+1.00, +0.55, +0.10, -0.15]
nb_RER = [-1.50, -1.00, -0.50, -0.20]

mod_Y   = _q(Y,   [1,4,8,12])
mod_PI  = _q(PI,  [1,4,8,12])
mod_IR  = _q(I_R, [1,4,8,12])
mod_RER = _q(RER, [1,4,8,12])

pts = [(m,n) for mv,nv in zip([mod_Y,mod_PI,mod_IR,mod_RER],[nb_Y,nb_PI,nb_IR,nb_RER])
       for m,n in zip(mv,nv)]
rmse = float(np.sqrt(np.mean([(m-n)**2 for m,n in pts])))

psi_R_post  = float(summary.get('psi_R',  {}).get('mean', float('nan')))
phi_O_post  = float(summary.get('phi_O',  {}).get('mean', float('nan')))
rho_O_post  = float(summary.get('rho_O',  {}).get('mean', float('nan')))
sigma_H_post = float(summary.get('sigma_H', {}).get('mean', float('nan')))

print(f"\n=== kj47 ENDELIG EVALUERING (Fase 2, blokksampling + phi_O + rho_O) ===")
print(f"RMSE(korr NB)={rmse:.4f}")
print(f"  psi_R  = {psi_R_post:.4f}")
print(f"  phi_O  = {phi_O_post:.4f}  (K&M: 0.15)")
print(f"  rho_O  = {rho_O_post:.4f}  (K&M: 0.874, kj46: 0.244)")
print(f"  sigma_H= {sigma_H_post:.4f}")
print(f"  Y:   {[f'{v:.3f}' for v in mod_Y]}  (NB: {nb_Y})")
print(f"  PI:  {[f'{v:.3f}' for v in mod_PI]}  (NB: {nb_PI})")
print(f"  I_R: {[f'{v:.3f}' for v in mod_IR]}  (NB: {nb_IR})")
print(f"  RER: {[f'{v:.3f}' for v in mod_RER]}  (NB: {nb_RER})")
print(f"  I_R.q12(modell)={mod_IR[3]:.3f}  (NB: -0.15)")
print(f"  BK-stabil: {d.get('stable')}, max|eig(T)|={d.get('max_eig_T',float('nan')):.6f}")
print()
print(f"  phi_O-diagnose: {'HEVET (>K&M)' if phi_O_post>0.15 else 'LAVT (≤K&M)'}")
print(f"  rho_O-diagnose: {'HEVET (>0.50)' if rho_O_post>0.50 else 'LAVT (≤0.50)'}")
print()
print("kj47 fullført.")
