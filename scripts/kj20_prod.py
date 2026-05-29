"""
Kjøring 20 — phi_I1 fast=0.50, rho_s fast=0.0, KPI-JAE.

Metodikk (PE-godkjent 2026-05-26):
  - phi_I1 kalibrert fast=0.50 (sweep-diagnose: BNP-ratio 1.16× NB ved kj19-posterior)
    kj19 estimerte phi_I1→0.103 → BNP q4-ratio ~4.5× NB (for høy)
    phi_I1=0.5 tilsvarer kj9-estimat (standard KPI) som ga BNP -0.447% vs NB -0.450%
  - rho_s fast=0.0 (ren UIP) — kj19 avviste AR(1)-glatting (posterior: 0.009)
  - KPI-JAE (pi_core_obs) — kj16/kj18 viste nødvendig for KPI-ratio ≥0.35×
  - psi_R prior: Beta(2,3,[0.01,0.970]) — kj18-godkjent
  - Startverdi: kj19 posterior means (de 19 overlappende parameterne)
  - 200k produksjon, seed=20

Mål: BNP q4-ratio [0.8,1.5]× NB OG KPI q4-ratio ≥0.35× NB.

Lagres til: data/results/chain_kj20_prod*
"""

import sys
import numpy as np

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent / 'src'))

from nemo.estimation.mcmc import (
    PARAM_NAMES, N_PARAMS, KM,
    build_H, build_Sv, OBS_NAMES,
    adaptive_mcmc_with_monitoring, log_posterior,
    SIGMA_RP_FIXED, PHI_I1_FIXED,
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

# ── Startverdi — kj19 posterior means (19 overlappende param) ────────────────
kj19_fil = rot / "data/results/chain_kj19_prod.npy"
KJ19_NAMES = PARAM_NAMES + ['rho_s', 'phi_I1']  # kj19 hadde 21 param
# Rekonstruer: kj19 hadde [rho_A,...,phi_u,rho_s] = 21 param
# Nåværende PARAM_NAMES har 19 (uten rho_s, phi_I1)
KJ19_FULL = ['rho_A','rho_C','rho_O','rho_Ys','rho_rp','rho_H',
             'sigma_A','sigma_C','sigma_O','sigma_Ys','sigma_i','sigma_P','sigma_H',
             'psi_R','psi_P1','psi_Y','gamma_p','phi_I1','phi_I2','phi_u','rho_s']

if kj19_fil.exists():
    print(f"\nLaster startverdi fra chain_kj19_prod.npy ...")
    kj19 = np.load(kj19_fil)
    n = kj19.shape[0]; half = kj19[n//2:,:]
    kj19_means = {KJ19_FULL[i]: float(half[:,i].mean()) for i in range(len(KJ19_FULL))}
    kj19_stds  = {KJ19_FULL[i]: float(half[:,i].std())  for i in range(len(KJ19_FULL))}
    theta_start = np.array([kj19_means.get(n2, KM.get(n2, 0.5)) for n2 in PARAM_NAMES])
    post_std    = np.array([max(kj19_stds.get(n2, 0.05), 1e-4) for n2 in PARAM_NAMES])
    print(f"  19 parametere (uten phi_I1={PHI_I1_FIXED} og rho_s=0.0, begge fast)")
else:
    print("Advarsel: chain_kj19_prod.npy ikke funnet — bruker K&M startverdi")
    theta_start = np.array([KM.get(n2, 0.5) for n2 in PARAM_NAMES])
    post_std    = np.array([0.05] * N_PARAMS)

# Korriger psi_R hvis over ny prior-grense (0.970)
psi_R_idx = PARAM_NAMES.index("psi_R")
if theta_start[psi_R_idx] > 0.969:
    theta_start[psi_R_idx] = 0.950
    print(f"  psi_R justert til 0.950 (kj19: {kj19_means.get('psi_R',0):.4f} > 0.970)")

print(f"\nKjøring 20: {N_PARAMS} parametere")
print(f"  sigma_rp fast={SIGMA_RP_FIXED}  kappa_M fast={KM['kappa_M']}")
print(f"  phi_I1 fast={PHI_I1_FIXED}  rho_s fast=0.0")
print(f"  psi_R prior: Beta(2,3,[0.01,0.970])")

print(f"\nStartverdier:")
for i, n2 in enumerate(PARAM_NAMES):
    print(f"  {n2:12s}: {theta_start[i]:.4f}  std={post_std[i]:.4f}")

# ── Sjekk startverdi ──────────────────────────────────────────────────────────
H  = build_H()
Sv = build_Sv()
lp0 = log_posterior(theta_start, H, Sv, pre, post)
print(f"\nStartverdi log-posterior: {lp0:.2f}")
if not np.isfinite(lp0):
    raise ValueError(f"Startverdi gir ikke-endelig lp={lp0}.")

# ── MCMC ──────────────────────────────────────────────────────────────────────
save_pref = str(rot / "data/results/chain_kj20_prod")
print(f"\nLagrer til: {save_pref}*")
print(f"Starter kjøring 20 (200k produksjon + 20k burnin) ...")
print()

chain, lp_vec, meta = adaptive_mcmc_with_monitoring(
    Y_pre=pre, Y_post=post, H=H, Sv=Sv,
    theta_init=theta_start, post_std_init=post_std,
    n_production=200_000, burnin=20_000, adapt_every=500,
    check_every=10_000, max_recalib=5,
    psrf_thr=1.10, ess_pct_thr=0.02,
    scale_init=0.676, seed=20, verbose=True,
    save_prefix=save_pref,
    use_reparam=False,
)

print(f"\nKjøring 20 fullført.")
