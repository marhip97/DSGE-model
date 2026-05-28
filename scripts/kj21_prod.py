"""
Kjøring 21 — A4b: fremoverskuende Taylor-regel E_t[π_{t+4}], KPI-JAE.

Metodikk (PE-godkjent 2026-05-28):
  - K&M §2.13 mimicking rule: i_R = ψ_R·i_{t-1} + (1-ψ_R)·[ψ_P1·E_t[π_{t+4}] + ψ_Y·y + ...] + ε_i
  - lambda_pi4 fast=0.0 (ren E_t[π_{t+4}]) — build_matrices_pi4chain, NZ=53
  - sigma_A fast=0.006 (PE-godkjent 2026-05-28) — kj20: sigma_A→0.049 (tak=0.050) → 0% aksept
  - phi_I1 fast=0.50 (kj20-PE-godkjent), rho_s fast=0.0 (ren UIP)
  - KPI-JAE (pi_core_obs) — build_H_pi4chain() (N_OBS×53)
  - Startverdi: kj20 posterior means (18 overlappende param, sigma_A utelatt)
  - phi_u justert til K&M=0.22 (kj20: phi_u→0.012 ved gulvet)
  - 200k produksjon, seed=21

Hypotese: samtid π_t i Taylor (kj18–kj20) svekker Taylor-prinsippet og driver
  psi_R→0.956. Med E_t[π_{t+4}] forventes psi_R å falle mot K&M=0.667,
  psi_P1 å stige mot K&M=0.292, og KPI q4-ratio å øke mot ≥0.35× NB.

Mål: BNP q4-ratio [0.8,1.5]× NB OG KPI q4-ratio ≥0.35× NB.

Lagres til: data/results/chain_kj21_prod*
"""

import sys
import numpy as np

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent / 'src'))

from nemo.estimation.mcmc import (
    PARAM_NAMES, N_PARAMS, KM,
    build_H_pi4chain, build_Sv, OBS_NAMES,
    adaptive_mcmc_with_monitoring, log_posterior,
    SIGMA_RP_FIXED, SIGMA_A_FIXED, PHI_I1_FIXED, LAMBDA_PI4_FIXED,
)
import pandas as pd

# ── Data — KPI-JAE ────────────────────────────────────────────────────────────
rot = __import__('pathlib').Path(__file__).parent.parent

datafil = rot / "data/processed/nemo_data_kpi_jae.csv"
if not datafil.exists():
    raise FileNotFoundError(f"{datafil} ikke funnet.")

obs_df = pd.read_csv(datafil, index_col=0, parse_dates=True)
PI_OBS_COL = "pi_core_obs"
if PI_OBS_COL not in obs_df.columns:
    raise ValueError(f"Kolonne {PI_OBS_COL} mangler i {datafil.name}")

obs_kols = [PI_OBS_COL if k == 'pi_obs' else k for k in OBS_NAMES]
pre  = obs_df[obs_df.index <= '2019-12-31'][obs_kols].values
post = obs_df[obs_df.index >= '2022-01-01'][obs_kols].values

print(f"Datafil: {datafil.name} (KPI-JAE)")
print(f"Pre={len(pre)} kv  Post={len(post)} kv  Totalt={len(pre)+len(post)} kv")

# ── Startverdi — kj20 posterior means (18 overlappende parametere) ───────────
# kj20 hadde 19 param (inkl. sigma_A); kj21 har 18 (sigma_A fryses fast=0.006)
kj20_fil = rot / "data/results/chain_kj20_prod.npy"
KJ20_NAMES = ['rho_A','rho_C','rho_O','rho_Ys','rho_rp','rho_H',
              'sigma_A','sigma_C','sigma_O','sigma_Ys','sigma_i','sigma_P','sigma_H',
              'psi_R','psi_P1','psi_Y','gamma_p','phi_I2','phi_u']  # kj20: 19 param

if kj20_fil.exists():
    print(f"\nLaster startverdi fra chain_kj20_prod.npy ...")
    kj20 = np.load(kj20_fil)
    n = kj20.shape[0]; half = kj20[n//2:]
    kj20_means = {KJ20_NAMES[i]: float(half[:,i].mean()) for i in range(len(KJ20_NAMES))}
    kj20_stds  = {KJ20_NAMES[i]: float(half[:,i].std())  for i in range(len(KJ20_NAMES))}
    theta_start = np.array([kj20_means.get(n2, KM.get(n2, 0.5)) for n2 in PARAM_NAMES])
    post_std    = np.array([max(kj20_stds.get(n2, 0.05), 1e-4) for n2 in PARAM_NAMES])
    print(f"  {N_PARAMS} parametere (sigma_A={SIGMA_A_FIXED} fast, phi_I1={PHI_I1_FIXED}, rho_s=0.0, lambda_pi4={LAMBDA_PI4_FIXED})")
else:
    print("Advarsel: chain_kj20_prod.npy ikke funnet — bruker K&M startverdi")
    theta_start = np.array([KM.get(n2, 0.5) for n2 in PARAM_NAMES])
    post_std    = np.array([0.05] * N_PARAMS)

# Korriger psi_R hvis over prior-grense (0.970)
psi_R_idx = PARAM_NAMES.index("psi_R")
if theta_start[psi_R_idx] > 0.969:
    theta_start[psi_R_idx] = 0.950
    print(f"  psi_R justert til 0.950 (kj20: {kj20_means.get('psi_R',0):.4f} > 0.970)")

# Korriger phi_u: kj20 konvergerte til gulvet (0.012 ≈ lb=0.010) → bruk K&M-startverdi
phi_u_idx = PARAM_NAMES.index("phi_u")
if theta_start[phi_u_idx] < 0.05:
    theta_start[phi_u_idx] = KM.get('phi_u', 0.22)
    post_std[phi_u_idx] = 0.05
    print(f"  phi_u justert til {theta_start[phi_u_idx]:.3f} (kj20: {kj20_means.get('phi_u',0):.4f} ved gulvet)")

print(f"\nKjøring 21: {N_PARAMS} parametere")
print(f"  sigma_rp fast={SIGMA_RP_FIXED}  sigma_A fast={SIGMA_A_FIXED}")
print(f"  phi_I1 fast={PHI_I1_FIXED}  rho_s fast=0.0")
print(f"  lambda_pi4 fast={LAMBDA_PI4_FIXED}  (ren E_t[pi_{{t+4}}], NZ=53)")
print(f"  psi_R prior: Beta(2,3,[0.01,0.970])")

print(f"\nStartverdier:")
for i, n2 in enumerate(PARAM_NAMES):
    print(f"  {n2:12s}: {theta_start[i]:.4f}  std={post_std[i]:.4f}")

# ── Sjekk startverdi ──────────────────────────────────────────────────────────
H  = build_H_pi4chain()
Sv = build_Sv()
lp0 = log_posterior(theta_start, H, Sv, pre, post)
print(f"\nStartverdi log-posterior: {lp0:.2f}")
if not np.isfinite(lp0):
    raise ValueError(f"Startverdi gir ikke-endelig lp={lp0}.")

# ── MCMC ──────────────────────────────────────────────────────────────────────
save_pref = str(rot / "data/results/chain_kj21_prod")
print(f"\nLagrer til: {save_pref}*")
print(f"Starter kjøring 21 (200k produksjon + 20k burnin) ...")
print()

chain, lp_vec, meta = adaptive_mcmc_with_monitoring(
    Y_pre=pre, Y_post=post, H=H, Sv=Sv,
    theta_init=theta_start, post_std_init=post_std,
    n_production=200_000, burnin=20_000, adapt_every=500,
    check_every=10_000, max_recalib=5,
    psrf_thr=1.10, ess_pct_thr=0.02,
    scale_init=0.676, seed=21, verbose=True,
    save_prefix=save_pref,
    use_reparam=False,
)

print(f"\nKjøring 21 fullført.")
