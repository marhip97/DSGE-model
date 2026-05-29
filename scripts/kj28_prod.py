"""
Kjøring 28 — phi_I1 fri + rho_H/phi_H1 priors fikset.

Endringer fra kj27 (Alt B, PE fullmakt 2026-05-29):

  PARAMETERENDRINGER (20 param):
  - phi_I1: reaktivert som estimert parameter
    Prior: Normal(2.0, 5.0, [0.1, 25.0])
    Startverdi: 0.50 (LL-sweep: by4=1.01× ved phi_I1=0.50, psi_R=0.989)
    Begrunnelse: LL-sweep viser LL=-3235 ved phi_I1=0.30 vs LL=-3262 ved phi_I1=12.54.
    Data foretrekker phi_I1∈[0.3, 0.75] som OG passer B5-kriteriet.

  PRIOR-ENDRINGER:
  - rho_H: Beta(2,0.5,[0.01,0.9995]) → Beta(5,3,[0.30,0.95])
    Gammel prior hadde mode ved 0.9995 → drev rho_H→0.965 (kj26).
    Ny prior: mode=0.667≈K&M=0.694. Blokkerer kollaps til 0.15 (kj27-problem).
  - phi_H1: Normal(60.73,40,[0.5,200]) → Normal(60.73,5,[30,100])
    Strammet kraftig for å eliminere bimodal phi_H1/rho_H fra kj27.

  STRUKTUR: Alt B (NZ=51) beholdes — RMSE=0.059 var god.
  Warm start: kj27 posterior (19 param) + phi_I1=0.50 som 17. param.
  rho_H startverdi justert til 0.70 (K&M, innenfor ny prior [0.30,0.95]).
  phi_H1 startverdi justert til 60.73 (K&M, innenfor ny prior [30,100]).

  Mål: B5 BESTÅTT (by4∈[0.8,1.5], bpi4≥0.35) OG RMSE < 0.118
  Lagres til: data/results/chain_kj28_prod*
"""

import sys
import numpy as np
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from nemo.estimation.mcmc import (
    PARAM_NAMES, N_PARAMS, KM,
    build_H_altB, build_Sv,
    adaptive_mcmc_with_monitoring, log_posterior,
    SIGMA_RP_FIXED, SIGMA_A_FIXED, PHI_I1_KJ26_FIXED, PHI_U_FIXED,
    PHI_PQ_KJ26_FIXED,
)
from nemo.model.equations import build_matrices_altB
from nemo.model.parameters import Parameters
import pandas as pd

rot = Path(__file__).parent.parent

# ── Data — KPI-JAE ────────────────────────────────────────────────────────────
datafil = rot / "data/processed/nemo_data_kpi_jae.csv"
if not datafil.exists():
    raise FileNotFoundError(f"{datafil} ikke funnet.")

from nemo.estimation.mcmc import OBS_NAMES
obs_df   = pd.read_csv(datafil, index_col=0, parse_dates=True)
PI_OBS_COL = "pi_core_obs"
obs_kols = [PI_OBS_COL if k == 'pi_obs' else k for k in OBS_NAMES]
pre  = obs_df[obs_df.index <= '2019-12-31'][obs_kols].values
post = obs_df[obs_df.index >= '2022-01-01'][obs_kols].values

print(f"Datafil: {datafil.name} (KPI-JAE)")
print(f"Pre={len(pre)} kv  Post={len(post)} kv")

# ── Warm start fra kj27 posterior (19 param) + phi_I1=0.50 ───────────────────
posterior_fil = rot / "data/results/chain_kj27_prod_posterior.json"

print(f"\nN_PARAMS = {N_PARAMS}  (kj27: 19, kj28: 20 — phi_I1 reaktivert)")
assert N_PARAMS == 20, f"Forventet N_PARAMS=20, fikk {N_PARAMS}"

if posterior_fil.exists():
    print(f"Warm start fra kj27 posterior: {posterior_fil.name}")
    with open(posterior_fil) as f:
        post_json = json.load(f)['summary']
    theta_start = np.zeros(N_PARAMS)
    post_std    = np.zeros(N_PARAMS)
    for i, n in enumerate(PARAM_NAMES):
        if n in post_json:
            theta_start[i] = post_json[n]['mean']
            post_std[i]    = max(post_json[n]['std'], 0.001)
        elif n == 'phi_I1':
            theta_start[i] = 0.50      # B5-passing startverdi (by4=1.01× ved psi_R=0.99)
            post_std[i]    = 0.30
        else:
            theta_start[i] = KM.get(n, 0.5)
            post_std[i]    = 0.05
else:
    print("kj27 ikke funnet — bruker K&M")
    theta_start = np.array([KM.get(n, 0.5) for n in PARAM_NAMES])
    post_std    = np.array([max(KM.get(n,0.5)*0.3, 0.001) for n in PARAM_NAMES])

# ── Juster startverdier for endrede priors ────────────────────────────────────
rho_H_idx  = PARAM_NAMES.index('rho_H')
phi_H1_idx = PARAM_NAMES.index('phi_H1')
phi_I1_idx = PARAM_NAMES.index('phi_I1')
phi_I2_idx = PARAM_NAMES.index('phi_I2')

# rho_H: kj27 posterior=0.147, ny prior [0.30, 0.95] — startverdi innenfor grense
if theta_start[rho_H_idx] < 0.30:
    theta_start[rho_H_idx] = 0.694   # K&M startverdi
    post_std[rho_H_idx]    = 0.08

# phi_H1: kj27 posterior=94.75, ny prior [30, 100] — ok innenfor grense, men start nærmere K&M
theta_start[phi_H1_idx] = 60.73   # K&M startverdi (stram prior sentrert her)
post_std[phi_H1_idx]    = 5.0

# phi_I2: behold kj27 verdi
if theta_start[phi_I2_idx] < 1.0:
    theta_start[phi_I2_idx] = 64.5
    post_std[phi_I2_idx]    = 20.0

# ── Rapporter startverdier ────────────────────────────────────────────────────
print(f"\nKjøring 28: {N_PARAMS} parametere (Alt B, NZ=51)")
print(f"  φ_PQ fast={PHI_PQ_KJ26_FIXED} (K&M)  h_c fast=0.938")
print(f"  phi_I1 fri: Normal(2.0,5.0,[0.1,25])  startverdi={theta_start[phi_I1_idx]:.2f}")
print(f"  phi_H1 stram: Normal(60.73,5.0,[30,100])  startverdi={theta_start[phi_H1_idx]:.2f}")
print(f"  rho_H ny prior: Beta(5,3,[0.30,0.95])  startverdi={theta_start[rho_H_idx]:.3f}")
print(f"  psi_R: Beta(2,2,[0.50,0.99])  startverdi={theta_start[PARAM_NAMES.index('psi_R')]:.3f}")

print(f"\nStartverdier:")
for i, n in enumerate(PARAM_NAMES):
    print(f"  {n:12s}: {theta_start[i]:.4f} (std={post_std[i]:.4f})")

# ── Bygg Alt B H-matrise og Sv ───────────────────────────────────────────────
H  = build_H_altB()
Sv = build_Sv()

print(f"\nModell: build_matrices_altB  H={H.shape}  (NZ_ALTB=51, NE=13)")

# ── Sjekk startverdi log-posterior ───────────────────────────────────────────
lp0 = log_posterior(theta_start, H, Sv, pre, post,
                    build_fn=build_matrices_altB)
print(f"\nStartverdi log-posterior: {lp0:.2f}")
if not np.isfinite(lp0):
    print("  Ikke-endelig lp ved startverdi! Prøver K&M-parametere...")
    theta_start = np.array([KM.get(n, 0.5) for n in PARAM_NAMES])
    theta_start[phi_I1_idx] = 0.50
    theta_start[PARAM_NAMES.index('psi_R')] = 0.85
    lp0 = log_posterior(theta_start, H, Sv, pre, post,
                        build_fn=build_matrices_altB)
    print(f"  Fallback start: lp={lp0:.2f}")
    if not np.isfinite(lp0):
        raise ValueError(f"Startverdi gir ikke-endelig lp={lp0} selv med fallback.")

# ── MCMC ──────────────────────────────────────────────────────────────────────
save_pref = str(rot / "data/results/chain_kj28_prod")
print(f"\nLagrer til: {save_pref}*")
print("Starter kjøring 28 (200k produksjon + 15k burnin, Alt B, NZ=51, N=20) ...")
print()

chain, lp_vec, meta = adaptive_mcmc_with_monitoring(
    Y_pre=pre, Y_post=post, H=H, Sv=Sv,
    theta_init=theta_start, post_std_init=post_std,
    n_production=200_000, burnin=15_000, adapt_every=500,
    check_every=10_000, max_recalib=6,
    psrf_thr=1.10, ess_pct_thr=0.02,
    scale_init=0.70, seed=28, verbose=True,
    save_prefix=save_pref,
    build_fn=build_matrices_altB,
    use_reparam=False,
)

print(f"\nKjøring 28 fullført.")
