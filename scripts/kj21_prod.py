"""
Kjøring 21 — DIAGNOSE: psi_R fast=K&M=0.667, v3, KPI-JAE.

Metodikk (PE-godkjent 2026-05-28):
  - DIAGNOSE: psi_R fast=0.667 (K&M §2.13) — test om psi_R→0.956 er rotårsak
    til KPI q4-ratio 0.183× NB. Hypotese: med lavere renteglatting vil
    effektiv inflasjonskoff (1-psi_R)×psi_P1 bli 7× større og KPI-respons bedres.
  - pi4chain (lambda=0) mislyktes i kj21a/b: sigma_i→0 (degenerert modus)
  - sigma_A fast=0.006 (PE-godkjent kj21a)
  - phi_I1 fast=0.50, rho_s fast=0.0 (ren UIP)
  - v3-matriser (NZ=49), KPI-JAE
  - Startverdi: K&M-defaults (kj20-posterior er feil startpunkt etter strukturelle endringer)
  - 200k produksjon, seed=21

Mål: BNP q4-ratio [0.8,1.5]× NB OG KPI q4-ratio ≥0.35× NB.
     Diagnostikkmål: verifiser om psi_R=0.667 gir bedre IRF enn psi_R=0.956.

Lagres til: data/results/chain_kj21_prod*
"""

import sys
import numpy as np

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent / 'src'))

from nemo.estimation.mcmc import (
    PARAM_NAMES, N_PARAMS, KM,
    build_H, build_Sv, OBS_NAMES,
    adaptive_mcmc_with_monitoring, log_posterior,
    SIGMA_RP_FIXED, SIGMA_A_FIXED, PHI_I1_FIXED, PSI_R_FIXED,
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

# ── Startverdi — K&M-defaults (frisk start etter strukturelle endringer) ─────
# kj20-posterior er kalibrert under v3 med psi_R fri → feil startpunkt for kj21
# der psi_R er frosset til 0.667 (K&M).
print(f"\nBruker K&M-defaults som startverdi (N_PARAMS={N_PARAMS}, psi_R fast={PSI_R_FIXED})")
theta_start = np.array([KM.get(n2, 0.5) for n2 in PARAM_NAMES])
# Proporsjonale post_std: 30% av K&M-verdi, minimum 0.001 — unngår at sigma-parameter
# med K&M≈0.0003 (sigma_i) får for store forslag og straks bryter nedre grense.
post_std = np.array([max(KM.get(n2, 0.5) * 0.30, 0.001) for n2 in PARAM_NAMES])

print(f"\nKjøring 21: {N_PARAMS} parametere")
print(f"  sigma_rp fast={SIGMA_RP_FIXED}  sigma_A fast={SIGMA_A_FIXED}")
print(f"  psi_R fast={PSI_R_FIXED}  phi_I1 fast={PHI_I1_FIXED}  rho_s fast=0.0")
print(f"  v3-matriser (NZ=49)")

print(f"\nStartverdier (K&M):")
for i, n2 in enumerate(PARAM_NAMES):
    print(f"  {n2:12s}: {theta_start[i]:.4f}")

# ── Sjekk startverdi ──────────────────────────────────────────────────────────
H  = build_H()
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
