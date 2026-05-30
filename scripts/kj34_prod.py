"""
Kjøring 34 — Varm fortsettelse av kj33 (psi_R=0.88, utvidet burn-in).

Motivasjon (2026-05-30):
  kj33 (74k/200k): psi_R stabil ~0.903, men rho_A driftet 0.15→0.47 (ikke konv.).
  RMSE(16pt NB)=0.200 i tail [55k:74k] ✅ — men rho_* trenger mer tid.
  Strategi: start fra kj33-endepunkt, lengre burn-in (30k + 10 rekalibreringer).

ENDRINGER fra kj33:
  1. Warm start fra kj33 tail [55k:74k] mean (rho_A≈0.455) — ikke kj32
  2. burnin=30_000 (vs 15k), max_recalib=10 (vs 6) — mer rekalibrering
  3. Identiske priors som kj33

FORVENTET UTFALL:
  - rho_* konvergerer (starter fra riktig regime)
  - PSRF < 1.10 ved 40k–80k
  - RMSE(16pt NB) ≈ 0.200 (bekrefter kj33 tail-estimat)
  - B5: by4 ≈ 0.87× ✅

Lagres til: data/results/chain_kj34_prod*
"""

import sys
import numpy as np
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from nemo.estimation.mcmc import (
    PARAM_NAMES, N_PARAMS, KM,
    build_H, build_Sv,
    adaptive_mcmc_with_monitoring, log_posterior,
)
from nemo.model.equations import build_matrices_v3
from nemo.model.parameters import Parameters
import pandas as pd

rot = Path(__file__).parent.parent

datafil = rot / "data/processed/nemo_data_kpi_jae.csv"
if not datafil.exists():
    raise FileNotFoundError(f"{datafil} ikke funnet.")

from nemo.estimation.mcmc import OBS_NAMES
obs_df   = pd.read_csv(datafil, index_col=0, parse_dates=True)
obs_kols = ['pi_core_obs' if k == 'pi_obs' else k for k in OBS_NAMES]
pre  = obs_df[obs_df.index <= '2019-12-31'][obs_kols].values
post = obs_df[obs_df.index >= '2022-01-01'][obs_kols].values

print(f"Datafil: {datafil.name} (KPI-JAE)")
print(f"Pre={len(pre)} kv  Post={len(post)} kv")

H  = build_H()
Sv = build_Sv()

PHI_I1_KJ34 = 0.50
PHI_H1_KJ34 = 60.73
PSI_R_KJ34  = 0.88

prior_overrides = {
    'phi_I1': ('normal', PHI_I1_KJ34, 0.001, 0.40, 0.60),
    'phi_H1': ('normal', PHI_H1_KJ34, 0.001, 60.70, 60.76),
    'psi_R':  ('normal', PSI_R_KJ34, 0.005, 0.84, 0.92),
    'rho_C':  ('beta', 5.0, 3.0, 0.10, 0.99),
    'rho_O':  ('beta', 5.0, 3.0, 0.10, 0.99),
    'rho_Ys': ('beta', 5.0, 3.0, 0.10, 0.99),
    'rho_rp': ('beta', 5.0, 3.0, 0.10, 0.99),
    'rho_A':  ('beta', 5.0, 3.0, 0.01, 0.99),
    'rho_H':  ('beta', 5.0, 2.0, 0.30, 0.99),
}

print(f"\nN_PARAMS = {N_PARAMS}")

# Warm start fra kj33 tail [55k:74k] — rho_* nær konvergert
kj33_partial = rot / "data/results/chain_kj33_prod_partial.npy"
kj33_tail_json = rot / "data/results/chain_kj33_tail_posterior.json"

theta_start = None
post_std    = None

if kj33_partial.exists():
    print(f"Warm start fra kj33 tail [55k:74k]: {kj33_partial.name}")
    chain_kj33 = np.load(kj33_partial)
    tail = chain_kj33[55000:74000]
    theta_start = tail.mean(axis=0).copy()
    post_std    = tail.std(axis=0).copy()
    post_std    = np.maximum(post_std, 0.001)
    print(f"  kj33 tail: {len(tail)} samples, rho_A={theta_start[PARAM_NAMES.index('rho_A')]:.4f}")
else:
    # Fallback: kj33 tail JSON
    for posterior_fil, label in [
        (kj33_tail_json, "kj33_tail"),
        (rot / "data/results/chain_kj31_prod_posterior.json", "kj31"),
    ]:
        if posterior_fil.exists():
            print(f"Warm start fra {label}: {posterior_fil.name}")
            with open(posterior_fil) as f:
                d = json.load(f)
            key = 'posterior_mean' if 'posterior_mean' in d else 'summary'
            data = d[key]
            theta_start = np.zeros(N_PARAMS)
            post_std    = np.zeros(N_PARAMS)
            for i, n in enumerate(PARAM_NAMES):
                if n in data:
                    if isinstance(data[n], dict):
                        theta_start[i] = data[n]['mean']
                        post_std[i]    = max(data[n].get('std', 0.05), 0.001)
                    else:
                        theta_start[i] = float(data[n])
                        post_std[i]    = 0.05
                else:
                    theta_start[i] = KM.get(n, 0.5)
                    post_std[i]    = 0.05
            break

if theta_start is None:
    theta_start = np.array([KM.get(n, 0.5) for n in PARAM_NAMES])
    post_std    = np.array([0.05]*N_PARAMS)

# Kj34-spesifikke startverdier — hold tighte priors fast
phi_I1_idx = PARAM_NAMES.index('phi_I1')
phi_H1_idx = PARAM_NAMES.index('phi_H1')
psi_R_idx  = PARAM_NAMES.index('psi_R')

theta_start[phi_I1_idx] = PHI_I1_KJ34; post_std[phi_I1_idx] = 0.001
theta_start[phi_H1_idx] = PHI_H1_KJ34; post_std[phi_H1_idx] = 0.001
# psi_R: klipp til prior-grense [0.84, 0.92]
theta_start[psi_R_idx] = np.clip(theta_start[psi_R_idx], 0.85, 0.91)
post_std[psi_R_idx]    = 0.005

# phi_I2: begrens std
phi_I2_idx = PARAM_NAMES.index('phi_I2')
post_std[phi_I2_idx] = min(post_std[phi_I2_idx], 30.0)

# rho-klipp
for n in ['rho_C', 'rho_O', 'rho_Ys', 'rho_rp']:
    idx = PARAM_NAMES.index(n)
    theta_start[idx] = np.clip(theta_start[idx], 0.10, 0.99)
rho_H_idx = PARAM_NAMES.index('rho_H')
theta_start[rho_H_idx] = np.clip(theta_start[rho_H_idx], 0.30, 0.99)

print(f"\nKjøring 34 — varm fortsettelse kj33")
print(f"  phi_I1 = {theta_start[phi_I1_idx]:.4f}")
print(f"  psi_R  = {theta_start[psi_R_idx]:.4f}")
print(f"  rho_A  = {theta_start[PARAM_NAMES.index('rho_A')]:.4f}")
print(f"  rho_H  = {theta_start[rho_H_idx]:.4f}")

print(f"\nStartverdier:")
for i, n in enumerate(PARAM_NAMES):
    print(f"  {n:12s}: {theta_start[i]:.4f} (std={post_std[i]:.4f})")

# B5-sjekk
from nemo.model.equations import Y, PI, I_R, E_i
from nemo.solver.blanchard_kahn import solve, compute_irf
p = Parameters()
for i, n in enumerate(PARAM_NAMES):
    if hasattr(p, n): setattr(p, n, float(theta_start[i]))
G0, G1, Psi, Pi = build_matrices_v3(p)
T, R, d = solve(G0, G1, Psi, Pi, verbose=False)
irf_raw = compute_irf(T, R, E_i, 0.0025, T_periods=20)
peak = float(np.max(irf_raw[:, I_R]))
irf  = irf_raw / peak
by4  = irf[3, Y] / (-0.450)
bpi4 = irf[3, PI] / (-0.150)
print(f"\nB5 ved start: by4={by4:.4f}  bpi4={bpi4:.4f}  BESTÅTT={0.8<=by4<=1.5 and bpi4>=0.35}")
print(f"  I_R: q1={irf[0,I_R]:.4f}  q4={irf[3,I_R]:.4f}  q8={irf[7,I_R]:.4f}  (NB: 1.00, 0.60, 0.20)")

lp0 = log_posterior(theta_start, H, Sv, pre, post,
                    build_fn=None, prior_overrides=prior_overrides)
print(f"Startverdi log-posterior: {lp0:.2f}")
if not np.isfinite(lp0):
    raise ValueError(f"Ikke-endelig lp={lp0}. Avbryter.")

save_pref = str(rot / "data/results/chain_kj34_prod")
print(f"\nLagrer til: {save_pref}*")
print("Starter kjøring 34 (200k produksjon + 30k burnin + 10 rekalibreringer) ...")
print()

chain, lp_vec, meta = adaptive_mcmc_with_monitoring(
    Y_pre=pre, Y_post=post, H=H, Sv=Sv,
    theta_init=theta_start, post_std_init=post_std,
    n_production=200_000, burnin=30_000, adapt_every=500,
    check_every=10_000, max_recalib=10,
    psrf_thr=1.10, ess_pct_thr=0.02,
    scale_init=0.70, seed=34, verbose=True,
    save_prefix=save_pref,
    build_fn=None,
    prior_overrides=prior_overrides,
    use_reparam=False,
)

print(f"\nKjøring 34 fullført.")
