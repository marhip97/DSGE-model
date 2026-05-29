"""
Kjøring 27 — Alt B: boliginvesteringskanal + psi_R tak hevet + phi_H1 estimert.

Endringer fra kj26 (PE-godkjent 2026-05-29, fullmakt):

  STRUKTURELL (Alt B):
  - build_matrices_altB: separat INV_H-tilstand (NZ 49→51) med CEE Euler-ligning
    h_W/h_NW akkumulering kobles til INV_H (ikke Q_H)
    Ressursbetingelse: IY*INV + IHY*INV_H (bolig separert fra kapital)
    Motivasjon: phi_H1=60.73 var kalibrert i parameters.py men aldri brukt i equations.py.

  NY ESTIMERT PARAMETER (19 param):
  - phi_H1: Normal(60.73, 40, [0.5, 200]) — K&M Tabell 8: 60.73
    phi_H1-sweep viste: K&M-verdi → BNP q4=0.33×; phi_H1≈1 → 0.78× (nær B5)
    Data bestemmer kompensasjon for manglende kanaler i forenklet modell

  PRIOR-ENDRING (psi_R):
  - psi_R: Beta(2,2,[0.50, 0.99])  — kj26 traff tak 0.95 med std=0.001
    Global endring i PARAM_PRIORS (se mcmc.py linje ~222)

  Mål: B5 bestått OG RMSE < 0.118 (kj25/kj26-benchmark)
  Lagres til: data/results/chain_kj27_prod*
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

# ── Warm start fra kj26 posterior (18 param) + phi_H1=K&M ────────────────────
posterior_fil = rot / "data/results/chain_kj26_prod_posterior.json"

# kj26 hadde 18 parametere (uten phi_H1)
old_names_kj26 = ['rho_A','rho_C','rho_O','rho_Ys','rho_rp','rho_H',
                   'sigma_C','sigma_O','sigma_Ys','sigma_i','sigma_P','sigma_H',
                   'psi_R','psi_P1','psi_Y','gamma_p','phi_I2','rho_s']

print(f"\nN_PARAMS = {N_PARAMS}  (kj26: 18, kj27: 19 — phi_H1 ny)")

if posterior_fil.exists():
    print(f"Warm start fra kj26 posterior: {posterior_fil.name}")
    with open(posterior_fil) as f:
        post_json = json.load(f)['summary']
    theta_start = np.zeros(N_PARAMS)
    post_std    = np.zeros(N_PARAMS)
    for i, n in enumerate(PARAM_NAMES):
        if n in old_names_kj26 and n in post_json:
            theta_start[i] = post_json[n]['mean']
            post_std[i]    = max(post_json[n]['std'], 0.001)
        elif n == 'phi_H1':
            theta_start[i] = KM['phi_H1']   # K&M startverdi = 60.73
            post_std[i]    = 15.0            # bred initiell STD (utforsking)
        else:
            theta_start[i] = KM.get(n, 0.5)
            post_std[i]    = 0.05
    print(f"  18 felles param fra kj26 + phi_H1={theta_start[PARAM_NAMES.index('phi_H1')]:.2f} (K&M start)")
else:
    print("kj26 ikke funnet — bruker K&M")
    theta_start = np.array([KM.get(n, 0.5) for n in PARAM_NAMES])
    post_std    = np.array([max(KM.get(n,0.5)*0.3, 0.001) for n in PARAM_NAMES])

# ── Sjekk phi_I2 startverdi ───────────────────────────────────────────────────
phi_I2_idx = PARAM_NAMES.index('phi_I2')
if theta_start[phi_I2_idx] < 1.0:
    theta_start[phi_I2_idx] = 64.5
    post_std[phi_I2_idx]    = 20.0

# ── Rapporter startverdier ────────────────────────────────────────────────────
print(f"\nKjøring 27: {N_PARAMS} parametere (Alt B, NZ=51)")
print(f"  φ_I1 fast={PHI_I1_KJ26_FIXED} (K&M)  φ_PQ fast={PHI_PQ_KJ26_FIXED} (K&M)")
print(f"  psi_R: Beta(2,2,[0.50,0.99])  startverdi={theta_start[PARAM_NAMES.index('psi_R')]:.3f}")
print(f"  phi_H1 fri: Normal(60.73,40,[0.5,200])  startverdi={theta_start[PARAM_NAMES.index('phi_H1')]:.2f}")
print(f"  phi_I2 fri: Normal(50,50,[1,400])  startverdi={theta_start[phi_I2_idx]:.1f}")

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
    theta_start[PARAM_NAMES.index('psi_R')] = 0.74
    lp0 = log_posterior(theta_start, H, Sv, pre, post,
                        build_fn=build_matrices_altB)
    print(f"  K&M start: lp={lp0:.2f}")
    if not np.isfinite(lp0):
        raise ValueError(f"Startverdi gir ikke-endelig lp={lp0} selv med K&M-parametere.")

# ── MCMC ──────────────────────────────────────────────────────────────────────
save_pref = str(rot / "data/results/chain_kj27_prod")
print(f"\nLagrer til: {save_pref}*")
print("Starter kjøring 27 (200k produksjon + 15k burnin, Alt B, NZ=51) ...")
print()

chain, lp_vec, meta = adaptive_mcmc_with_monitoring(
    Y_pre=pre, Y_post=post, H=H, Sv=Sv,
    theta_init=theta_start, post_std_init=post_std,
    n_production=200_000, burnin=15_000, adapt_every=500,
    check_every=10_000, max_recalib=6,
    psrf_thr=1.10, ess_pct_thr=0.02,
    scale_init=0.70, seed=27, verbose=True,
    save_prefix=save_pref,
    build_fn=build_matrices_altB,
    use_reparam=False,
)

print(f"\nKjøring 27 fullført.")
