"""
Kjøring 21 — A4b: fremoverskuende Taylor-regel E_t[π_{t+4}], KPI-JAE.

Metodikk (PE-godkjent 2026-05-28):
  - K&M §2.13 mimicking rule: i_R = ψ_R·i_{t-1} + (1-ψ_R)·[ψ_P1·E_t[π_{t+4}] + ψ_Y·y + ...] + ε_i
  - lambda_pi4 fast=0.0 (ren E_t[π_{t+4}]) — build_matrices_pi4chain, NZ=53
  - phi_I1 fast=0.50 (kj20-PE-godkjent), rho_s fast=0.0 (ren UIP)
  - KPI-JAE (pi_core_obs) — build_H_pi4chain() (N_OBS×53)
  - Startverdi: kj20 posterior means (19 overlappende parametere)
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
    SIGMA_RP_FIXED, PHI_I1_FIXED, LAMBDA_PI4_FIXED,
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

# ── Startverdi — kj20 posterior means (19 parametere) ────────────────────────
kj20_fil = rot / "data/results/chain_kj20_prod.npy"
KJ20_FULL = PARAM_NAMES  # kj20 har samme 19 param som kj21

if kj20_fil.exists():
    print(f"\nLaster startverdi fra chain_kj20_prod.npy ...")
    kj20 = np.load(kj20_fil)
    n = kj20.shape[0]; half = kj20[n//2:]
    kj20_means = {KJ20_FULL[i]: float(half[:,i].mean()) for i in range(len(KJ20_FULL))}
    kj20_stds  = {KJ20_FULL[i]: float(half[:,i].std())  for i in range(len(KJ20_FULL))}
    theta_start = np.array([kj20_means.get(n2, KM.get(n2, 0.5)) for n2 in PARAM_NAMES])
    post_std    = np.array([max(kj20_stds.get(n2, 0.05), 1e-4) for n2 in PARAM_NAMES])
    print(f"  {N_PARAMS} parametere (phi_I1={PHI_I1_FIXED}, rho_s=0.0, lambda_pi4={LAMBDA_PI4_FIXED} — alle fast)")
else:
    print("Advarsel: chain_kj20_prod.npy ikke funnet — bruker K&M startverdi")
    theta_start = np.array([KM.get(n2, 0.5) for n2 in PARAM_NAMES])
    post_std    = np.array([0.05] * N_PARAMS)

# Korriger psi_R hvis over prior-grense (0.970)
psi_R_idx = PARAM_NAMES.index("psi_R")
if theta_start[psi_R_idx] > 0.969:
    theta_start[psi_R_idx] = 0.950
    print(f"  psi_R justert til 0.950 (kj20: {kj20_means.get('psi_R',0):.4f} > 0.970)")

print(f"\nKjøring 21: {N_PARAMS} parametere")
print(f"  sigma_rp fast={SIGMA_RP_FIXED}  phi_I1 fast={PHI_I1_FIXED}  rho_s fast=0.0")
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
