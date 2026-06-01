"""
Kjøring 33 — NB-kalibrert psi_R (dogmatisk prior for IRF-match).

Motivasjon (multi-kvartal NB-benchmark, 2026-05-29):
  kj31/kj32 har psi_R≈0.989/0.999 (data-drevet) → I_R halvtid ≈69 kv.
  NB Memo 3/2024 Figur 1 impliserer I_R halvtid ≈3–4 kv (psi_R≈0.84–0.88).
  Multi-kvartal RMSE: kj31 RMSE=0.353 → psi_R=0.88 RMSE=0.200 (−43%).

  psi_R-identifikasjonskonflikt:
    Data-LL: monotont stigende til 0.999 (ΔLL≈+1224 fra K&M=0.666)
    NB-benchmark: minimert ved psi_R=0.88 (I_R q4=0.62 vs NB 0.60 ✅)
    Eneste løsning: dogmatisk prior Normal(0.88, 0.005) — LL-kostnad≈437

ENDRINGER fra kj31:
  1. psi_R: Beta(2,2,[0.50,0.99]) → Normal(0.88, 0.005, [0.84, 0.92])
     Faglig begrunnelse: kalibrering mot NB-standardmodell (GEORG ω_r≈0.74–0.88)
     Exitstrategi: kj31 (data-drevet psi_R=0.989) er referanselinje
  2. phi_I1: beholdt 0.50 (ikke 0.40 fra kj32)
     Begrunnelse: phi_I1=0.50 + psi_R=0.88 → by4=0.81× (akkurat B5-pass)
                  phi_I1=0.40 + psi_R=0.88 → forventes by4<0.80× (B5 feiler)

FORVENTET UTFALL:
  - psi_R → ~0.88 (prior-dominert — data-LL kan ikke nå 0.999)
  - RMSE(16pt NB) → ~0.200 (mot kj31: 0.353)
  - RMSE(Kalman) → ~0.060–0.065 (liten forverring fra LL-kostnad)
  - B5: by4 ≈ 0.81× (marginal pass — monitoreres nøye)
  - B5-risiko: hvis kj31 sigma_* justerer seg, kan by4 falle under 0.80×

Warm start: kj32 → kj31 → kj26 (fallback)
Lagres til: data/results/chain_kj33_prod*
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

PHI_I1_KJ33 = 0.50   # beholdt fra kj31: B5-sikker med psi_R=0.88
PHI_H1_KJ33 = 60.73
PSI_R_KJ33  = 0.88   # NB-kalibrert (minimerer RMSE(16pt))

prior_overrides = {
    # kj33: phi_I1=0.50 (B5-kritisk ved psi_R=0.88)
    'phi_I1': ('normal', PHI_I1_KJ33, 0.001, 0.40, 0.60),
    'phi_H1': ('normal', PHI_H1_KJ33, 0.001, 60.70, 60.76),
    # kj33: psi_R dogmatisk kalibrert til NB-IRF-decay
    'psi_R':  ('normal', PSI_R_KJ33, 0.005, 0.84, 0.92),
    # rho-fix identisk kj31 (konvergerte godt)
    'rho_C':  ('beta', 5.0, 3.0, 0.10, 0.99),
    'rho_O':  ('beta', 5.0, 3.0, 0.10, 0.99),
    'rho_Ys': ('beta', 5.0, 3.0, 0.10, 0.99),
    'rho_rp': ('beta', 5.0, 3.0, 0.10, 0.99),
    'rho_A':  ('beta', 5.0, 3.0, 0.01, 0.99),
    'rho_H':  ('beta', 5.0, 2.0, 0.30, 0.99),
}

print(f"\nN_PARAMS = {N_PARAMS}")
theta_start = None
post_std    = None

for posterior_fil, label in [
    (rot / "data/results/chain_kj32_prod_posterior.json", "kj32"),
    (rot / "data/results/chain_kj31_prod_posterior.json", "kj31"),
    (rot / "data/results/chain_kj30_prod_posterior.json", "kj30"),
    (rot / "data/results/chain_kj26_prod_posterior.json", "kj26"),
]:
    if posterior_fil.exists():
        print(f"Warm start fra {label} posterior: {posterior_fil.name}")
        with open(posterior_fil) as f:
            post_json = json.load(f)['summary']
        theta_start = np.zeros(N_PARAMS)
        post_std    = np.zeros(N_PARAMS)
        for i, n in enumerate(PARAM_NAMES):
            if n in post_json:
                theta_start[i] = post_json[n]['mean']
                post_std[i]    = max(post_json[n]['std'], 0.001)
            elif n == 'phi_I1':
                theta_start[i] = PHI_I1_KJ33; post_std[i] = 0.001
            elif n == 'phi_H1':
                theta_start[i] = PHI_H1_KJ33; post_std[i] = 0.001
            else:
                theta_start[i] = KM.get(n, 0.5); post_std[i] = 0.05
        break

if theta_start is None:
    theta_start = np.array([KM.get(n, 0.5) for n in PARAM_NAMES])
    post_std    = np.array([0.05]*N_PARAMS)

# Sett kj33-spesifikke startverdier
phi_I1_idx = PARAM_NAMES.index('phi_I1')
psi_R_idx  = PARAM_NAMES.index('psi_R')
rho_H_idx  = PARAM_NAMES.index('rho_H')

theta_start[phi_I1_idx] = PHI_I1_KJ33
post_std[phi_I1_idx]    = 0.001

# psi_R: flytt fra kj31/kj32 posterior (≈0.989) til kj33 startverdi (0.88)
theta_start[psi_R_idx] = PSI_R_KJ33
post_std[psi_R_idx]    = 0.005

# rho_H: klipp til ny prior-grense
theta_start[rho_H_idx] = np.clip(theta_start[rho_H_idx], 0.30, 0.99)

# rho_C/O/Ys/rp: klipp til [0.10, 0.99]
for n in ['rho_C', 'rho_O', 'rho_Ys', 'rho_rp']:
    idx = PARAM_NAMES.index(n)
    if theta_start[idx] <= 0.10 or theta_start[idx] >= 0.99:
        theta_start[idx] = KM.get(n, 0.5); post_std[idx] = 0.10

phi_I2_idx = PARAM_NAMES.index('phi_I2')
if theta_start[phi_I2_idx] < 1.0:
    theta_start[phi_I2_idx] = 64.5; post_std[phi_I2_idx] = 20.0

# Rapport
print(f"\nKjøring 33 — NB-kalibrert psi_R")
print(f"  phi_I1 = {theta_start[phi_I1_idx]:.4f} (frosset Normal(0.50,0.001) — B5-sikker)")
print(f"  psi_R  = {theta_start[psi_R_idx]:.4f}  (Normal(0.88,0.005,[0.84,0.92]) — NB-kalibrert)")
print(f"  rho_H  = {theta_start[rho_H_idx]:.4f}  (Beta(5,2,[0.30,0.99]))")

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

save_pref = str(rot / "data/results/chain_kj33_prod")
print(f"\nLagrer til: {save_pref}*")
print("Starter kjøring 33 (200k produksjon + 15k burnin) ...")
print()

chain, lp_vec, meta = adaptive_mcmc_with_monitoring(
    Y_pre=pre, Y_post=post, H=H, Sv=Sv,
    theta_init=theta_start, post_std_init=post_std,
    n_production=200_000, burnin=15_000, adapt_every=500,
    check_every=10_000, max_recalib=6,
    psrf_thr=1.10, ess_pct_thr=0.02,
    scale_init=0.70, seed=33, verbose=True,
    save_prefix=save_pref,
    build_fn=None,
    prior_overrides=prior_overrides,
    use_reparam=False,
)

print(f"\nKjøring 33 fullført.")
