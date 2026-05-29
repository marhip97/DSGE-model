"""
Kjøring 32 — phi_I1=0.40 (LL-optimal) + psi_R fri i B5-sonen.

Endringer fra kj31 (phi_I1=0.50 frosset, psi_R=0.9894):

  FUNN FRA 2D LL-SWEEP (phi_I1 × psi_R, kj31 posterior):
    psi_R-LL er monotont stigende mot 1.0 for ALLE phi_I1-verdier —
    grenseidentifikasjonsproblem: data vil ha psi_R→1.0.
    B5-kriteriet (0.8 ≤ by4 ≤ 1.5) er den bindende skranken.

    Optimal B5-passerende hjørne:
      phi_I1=0.10: LL=-2435 (best LL) — by4≥2.4 (B5 FEILER alltid)
      phi_I1=0.30: LL=-3139 ved psi_R=0.999 — by4=1.769 (B5 FEILER ved psi_R≥0.989)
      phi_I1=0.40: LL=-3222 ved psi_R=0.999 — by4=1.444 ✅ B5-pass hele [0.85,0.999]
      phi_I1=0.50: LL=-3260 ved psi_R=0.999 — by4=1.242 ✅ (kj31)
    → phi_I1=0.40 er LL-optimal B5-passerende verdi (ΔLL≈+37 vs phi_I1=0.50).

  ENDRING 1: phi_I1=0.40 (fra 0.50 i kj31)
    Prior: Normal(0.40, 0.001, [0.35, 0.45]) — tight frysing ved LL-optimal verdi
    Begrunnelse: Constrained MLE — maksimerer LL gitt B5-kravet.
    Exitstrategi: Normal(0.50, 0.001) (kj31) er referansen.

  ENDRING 2: psi_R fri i B5-sonen — Beta(2,2,[0.85,0.999])
    Fra: Beta(2,2,[0.50,0.99]) kj31 (traff 0.99-grensen)
    Til: Beta(2,2,[0.85,0.999]) — symmetrisk, mode=0.925
    Formål: La data velge innenfor B5-sonen [0.85,0.999].
             [0.85,0.999] garanterer B5-pass for phi_I1=0.40 (fra sweep).
    Exitstrategi: kj31-prior Beta(2,2,[0.50,0.99]) er referansen.

  BEHOLDT fra kj31:
    rho_A/C/O/Ys/rp/H: samme Beta-priors (konvergerte godt i kj31)
    phi_H1=60.73 frosset
    build_matrices_v3 (NZ=49)

  Mål: B5 BESTÅTT OG RMSE < 0.118 OG PSRF < 1.10
  Faglig hypotese: data bekrefter phi_I1=0.40 + psi_R∈[0.92,0.999]
  Lagres til: data/results/chain_kj32_prod*
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

PHI_I1_KJ32 = 0.40   # LL-optimal B5-passerende verdi (vs kj31: 0.50)
PHI_H1_KJ32 = 60.73

prior_overrides = {
    # kj32: phi_I1 justert til 0.40
    'phi_I1': ('normal', PHI_I1_KJ32, 0.001, 0.35, 0.45),
    'phi_H1': ('normal', PHI_H1_KJ32, 0.001, 60.70, 60.76),
    # rho-fix fra kj30/kj31
    'rho_C':  ('beta', 5.0, 3.0, 0.10, 0.99),
    'rho_O':  ('beta', 5.0, 3.0, 0.10, 0.99),
    'rho_Ys': ('beta', 5.0, 3.0, 0.10, 0.99),
    'rho_rp': ('beta', 5.0, 3.0, 0.10, 0.99),
    'rho_A':  ('beta', 5.0, 3.0, 0.01, 0.99),
    'rho_H':  ('beta', 5.0, 2.0, 0.30, 0.99),
    # kj32: psi_R fri i B5-sonen
    'psi_R':  ('beta', 2.0, 2.0, 0.85, 0.999),
}

# Warm start: kj31 → kj30 → kj26
print(f"\nN_PARAMS = {N_PARAMS}")
theta_start = None
post_std    = None

for posterior_fil, label in [
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
                theta_start[i] = PHI_I1_KJ32; post_std[i] = 0.001
            elif n == 'phi_H1':
                theta_start[i] = PHI_H1_KJ32; post_std[i] = 0.001
            else:
                theta_start[i] = KM.get(n, 0.5); post_std[i] = 0.05
        break

if theta_start is None:
    theta_start = np.array([KM.get(n, 0.5) for n in PARAM_NAMES])
    post_std    = np.array([0.05]*N_PARAMS)

# Sett phi_I1=0.40 eksplisitt
phi_I1_idx = PARAM_NAMES.index('phi_I1')
theta_start[phi_I1_idx] = PHI_I1_KJ32
post_std[phi_I1_idx]    = 0.001

# psi_R: kj31 posterior=0.9894 — justér til midtpunkt [0.85,0.999]=0.925
psi_R_idx = PARAM_NAMES.index('psi_R')
theta_start[psi_R_idx] = 0.925
post_std[psi_R_idx]    = 0.02

# rho_H: ny prior [0.30, 0.99]
rho_H_idx = PARAM_NAMES.index('rho_H')
theta_start[rho_H_idx] = np.clip(theta_start[rho_H_idx], 0.30, 0.99)

# rho_C/O/Ys/rp: klipp til [0.10, 0.99]
for n in ['rho_C','rho_O','rho_Ys','rho_rp']:
    idx = PARAM_NAMES.index(n)
    if theta_start[idx] <= 0.10 or theta_start[idx] >= 0.99:
        theta_start[idx] = KM.get(n, 0.5); post_std[idx] = 0.10

phi_I2_idx = PARAM_NAMES.index('phi_I2')
if theta_start[phi_I2_idx] < 1.0:
    theta_start[phi_I2_idx] = 64.5; post_std[phi_I2_idx] = 20.0

# Rapport
print(f"\nKjøring 32 — LL-optimal + psi_R fri i B5-sonen")
print(f"  phi_I1 = {theta_start[phi_I1_idx]:.4f} (frosset Normal(0.40,0.001) — LL-optimal)")
print(f"  psi_R  = {theta_start[psi_R_idx]:.4f}  start (Beta(2,2,[0.85,0.999]))")
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

lp0 = log_posterior(theta_start, H, Sv, pre, post,
                    build_fn=None, prior_overrides=prior_overrides)
print(f"Startverdi log-posterior: {lp0:.2f}")
if not np.isfinite(lp0):
    raise ValueError(f"Ikke-endelig lp={lp0}. Avbryter.")

save_pref = str(rot / "data/results/chain_kj32_prod")
print(f"\nLagrer til: {save_pref}*")
print("Starter kjøring 32 (200k produksjon + 15k burnin) ...")
print()

chain, lp_vec, meta = adaptive_mcmc_with_monitoring(
    Y_pre=pre, Y_post=post, H=H, Sv=Sv,
    theta_init=theta_start, post_std_init=post_std,
    n_production=200_000, burnin=15_000, adapt_every=500,
    check_every=10_000, max_recalib=6,
    psrf_thr=1.10, ess_pct_thr=0.02,
    scale_init=0.70, seed=32, verbose=True,
    save_prefix=save_pref,
    build_fn=None,
    prior_overrides=prior_overrides,
    use_reparam=False,
)

print(f"\nKjøring 32 fullført.")
