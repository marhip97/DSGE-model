"""
Kjøring 18 — Kombinasjon: KPI-JAE + prior-justert psi_R, 200k produksjon.

Metodikk (PE-godkjent 2026-05-26):
  - KPI-JAE (pi_core_obs) som inflasjonsmål — eneste test (kj16) som passerte 0.35×
  - psi_R prior justert: Beta(2,3,[0.01,0.970]) — hindrer grense-atferd (kj16: 0.987)
  - Startverdi: kj16 posterior means (allerede i KPI-JAE-parameterrom)
  - 200k produksjon (mot 100k i kj16) for bedre posterior-dekning

Baseline kj16: KPI q4=0.42× NB (✅), men BNP q4=-209% (❌), psi_R=0.987.
Mål kj18:      KPI q4 ≥ 0.35× NB OG BNP q4 0.8–1.5× NB.

Beslutningspunkt:
  OK: kj18 → produksjonskjøring, Fase 1 kan starte
  BNP fortsatt ustabil: strukturell UIP-utvidelse (Fase 1B)

Lagres til: data/results/chain_kj18_prod*
"""

import sys
import numpy as np

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent / 'src'))

from nemo.estimation.mcmc import (
    PARAM_NAMES, N_PARAMS, KM,
    build_H, build_Sv, OBS_NAMES,
    adaptive_mcmc_with_monitoring, log_posterior,
    SIGMA_RP_FIXED,
)
import pandas as pd

# ── Data — KPI-JAE ────────────────────────────────────────────────────────────
rot = __import__('pathlib').Path(__file__).parent.parent

datafil = rot / "data/processed/nemo_data_kpi_jae.csv"
if not datafil.exists():
    raise FileNotFoundError(
        f"{datafil} ikke funnet. Kjør datapipeline med --kpi-jae først."
    )

print(f"Datafil: {datafil.name} (KPI-JAE)")
obs_df = pd.read_csv(datafil, index_col=0, parse_dates=True)

PI_OBS_COL = "pi_core_obs"
if PI_OBS_COL not in obs_df.columns:
    raise ValueError(f"Kolonne {PI_OBS_COL} mangler i {datafil.name}")

obs_kols = [PI_OBS_COL if k == "pi_obs" else k for k in OBS_NAMES]
pre  = obs_df[obs_df.index <= "2019-12-31"][obs_kols].values
post = obs_df[obs_df.index >= "2022-01-01"][obs_kols].values
print(f"Pre={len(pre)} kv  Post={len(post)} kv  Totalt={len(pre)+len(post)} kv")
print(f"pi-observasjon: {PI_OBS_COL}")

H  = build_H()
Sv = build_Sv()

# ── Startverdi fra kj16 ───────────────────────────────────────────────────────
kj16_chain_fil = rot / "data/results/chain_kj16_prod.npy"
print(f"\nLaster startverdi fra {kj16_chain_fil.name} ...")
kj16_chain = np.load(kj16_chain_fil)
kj16_means = {name: float(kj16_chain[:, i].mean()) for i, name in enumerate(PARAM_NAMES)}
kj16_stds  = {name: float(kj16_chain[:, i].std())  for i, name in enumerate(PARAM_NAMES)}

print(f"\nKjøring 18: {N_PARAMS} parametere")
print(f"  sigma_rp fast={SIGMA_RP_FIXED}  kappa_M fast={KM['kappa_M']}")
print(f"  psi_R prior: Beta(2,3,[0.01,0.970])  (PE-godkjent 2026-05-26)")

theta_start = np.array([kj16_means[n] for n in PARAM_NAMES])
post_std    = np.array([max(kj16_stds[n], 1e-4) for n in PARAM_NAMES])

# Korriger startverdi for psi_R — kj16 hadde 0.987, ny prior-grense er 0.970
psi_R_idx = PARAM_NAMES.index("psi_R")
if theta_start[psi_R_idx] > 0.969:
    theta_start[psi_R_idx] = 0.950
    print(f"  psi_R startverdi justert til 0.950 (kj16 hadde {kj16_means['psi_R']:.4f} > 0.970)")

print("\nStartverdier (kj16 posterior means):")
for i, name in enumerate(PARAM_NAMES):
    print(f"  {name:12s}: {theta_start[i]:.4f}  (std={post_std[i]:.4f})")

lp0 = log_posterior(theta_start, H, Sv, pre, post)
print(f"\nStartverdi log-posterior: {lp0:.2f}")
if not np.isfinite(lp0):
    print("ADVARSEL: ugyldig startpunkt — prøver kj12 posterior.")
    kj12_chain = np.load(rot / "data/results/chain_kj12_prod.npy")
    KJ12 = ['rho_A','rho_C','rho_O','rho_Ys','rho_rp','rho_H',
             'sigma_A','sigma_C','sigma_O','sigma_Ys','sigma_i','sigma_P','sigma_H',
             'psi_R','psi_P1','psi_Y','gamma_p','phi_I1','phi_I2','phi_u']
    kj12_means = {n: float(kj12_chain[:,i].mean()) for i,n in enumerate(KJ12)}
    theta_start = np.array([kj12_means[n] for n in PARAM_NAMES])
    theta_start[psi_R_idx] = min(theta_start[psi_R_idx], 0.950)
    post_std    = np.array([0.05] * N_PARAMS)
    lp0 = log_posterior(theta_start, H, Sv, pre, post)
    print(f"  kj12-start lp0={lp0:.2f}")

# ── Kjør ──────────────────────────────────────────────────────────────────────
save_pref = str(rot / "data/results/chain_kj18_prod")
print(f"\nLagrer til: {save_pref}*")
print("Starter kjøring 18 (200k produksjon + 20k burnin) ...\n")

chain, lp_vec, meta = adaptive_mcmc_with_monitoring(
    Y_pre=pre, Y_post=post, H=H, Sv=Sv,
    theta_init=theta_start, post_std_init=post_std,
    n_production=200_000, burnin=20_000, adapt_every=500,
    check_every=10_000, max_recalib=5,
    psrf_thr=1.10, ess_pct_thr=0.02,
    scale_init=0.676, seed=18, verbose=True,
    save_prefix=save_pref,
    use_reparam=False,
)
print("\nKjøring 18 fullført.")
