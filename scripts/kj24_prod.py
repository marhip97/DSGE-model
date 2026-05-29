"""
Kjøring 24 — phi_u fast=K&M, warm start fra kj23 156k-posterior, seed=24.

Endring fra kj23:
  - phi_u kalibreres fast=0.2192 (K&M Tabell 8, mikrodata)
    Begrunnelse: kj23 konvergerte til phi_u=1.72 (8× K&M) → BNP=2.33× NB ❌
    Med phi_u=K&M og psi_R=0.968: BNP=1.10× NB ✅, KPI=0.47× NB ✅ (PE-godkjent 2026-05-28)
  - N_PARAMS: 18 → 17 (phi_u fjernet fra estimering)
  - Warm start: kj23 156k-posterior means (lp≈-2663, vel konvergert)
  - seed=24

Kontekst (2026-05-28):
  - κ_P=0.0448 (korrekt markup-normering, K&M side 28)
  - phi_I1=0.50, sigma_A=0.006, sigma_rp=0.006, h_c=0.938 — alle faste
  - psi_R fri: Beta(2,3,[0.01,0.970])
  - Feasibility (K&M-phi_u, full kj23-posterior): BNP=1.10×✅, KPI=0.47×✅

Mål: BNP q4-ratio [0.8,1.5]× NB OG KPI q4-ratio ≥0.35× NB.

Lagres til: data/results/chain_kj24_prod*
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from nemo.estimation.mcmc import (
    PARAM_NAMES, N_PARAMS, KM,
    build_H, build_Sv, OBS_NAMES,
    adaptive_mcmc_with_monitoring, log_posterior,
    SIGMA_RP_FIXED, SIGMA_A_FIXED, PHI_I1_FIXED, PHI_U_FIXED,
)
from nemo.model.parameters import Parameters
import pandas as pd

rot = Path(__file__).parent.parent

# ── Data — KPI-JAE ────────────────────────────────────────────────────────────
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

# ── Warm start fra kj23 156k-posterior ────────────────────────────────────────
partial_fil = rot / "data/results/chain_kj23_prod_partial.npy"
if partial_fil.exists():
    print(f"\nWarm start fra kj23 156k-posterior: {partial_fil.name}")
    ch_kj23 = np.load(partial_fil)

    # kj23 hadde 18 param (inkl. phi_u sist); kj24 har 17 (phi_u fjernet)
    old_names_kj23 = ['rho_A','rho_C','rho_O','rho_Ys','rho_rp','rho_H',
                      'sigma_C','sigma_O','sigma_Ys','sigma_i','sigma_P','sigma_H',
                      'psi_R','psi_P1','psi_Y','gamma_p','phi_I2','phi_u']
    assert ch_kj23.shape[1] == 18, f"Forventet 18 kolumner, fikk {ch_kj23.shape[1]}"
    idx = [old_names_kj23.index(n) for n in PARAM_NAMES]
    ch17 = ch_kj23[:, idx]

    theta_start = ch17.mean(axis=0)
    post_std    = np.maximum(ch17.std(axis=0), 0.001)
    print(f"  {ch_kj23.shape[0]} trekk brukt til warm start")
else:
    print(f"\nkj23-partial ikke funnet — bruker K&M-defaults")
    theta_start = np.array([KM.get(n, 0.5) for n in PARAM_NAMES])
    post_std    = np.array([max(KM.get(n, 0.5) * 0.30, 0.001) for n in PARAM_NAMES])

print(f"\nKjøring 24: {N_PARAMS} parametere")
print(f"  sigma_rp fast={SIGMA_RP_FIXED}  sigma_A fast={SIGMA_A_FIXED}")
print(f"  phi_I1 fast={PHI_I1_FIXED}  phi_u fast={PHI_U_FIXED} (K&M Tabell 8)")
print(f"  rho_s fast=0.0")
print(f"  psi_R fri: Beta(2,3,[0.01,0.970])")
print(f"  kappa_P = {Parameters.kappa_P():.5f} (ε(ε-1)/φ_PQ)")

print(f"\nStartverdier (kj23 warm start):")
for i, n in enumerate(PARAM_NAMES):
    print(f"  {n:12s}: {theta_start[i]:.4f} (std={post_std[i]:.4f})")

# ── Sjekk startverdi ──────────────────────────────────────────────────────────
H  = build_H()
Sv = build_Sv()
lp0 = log_posterior(theta_start, H, Sv, pre, post)
print(f"\nStartverdi log-posterior: {lp0:.2f}")
if not np.isfinite(lp0):
    raise ValueError(f"Startverdi gir ikke-endelig lp={lp0}.")

# ── MCMC ──────────────────────────────────────────────────────────────────────
save_pref = str(rot / "data/results/chain_kj24_prod")
print(f"\nLagrer til: {save_pref}*")
print(f"Starter kjøring 24 (200k produksjon + 10k burnin, phi_u fast=K&M) ...")
print()

chain, lp_vec, meta = adaptive_mcmc_with_monitoring(
    Y_pre=pre, Y_post=post, H=H, Sv=Sv,
    theta_init=theta_start, post_std_init=post_std,
    n_production=200_000, burnin=10_000, adapt_every=500,
    check_every=10_000, max_recalib=5,
    psrf_thr=1.10, ess_pct_thr=0.02,
    scale_init=0.75, seed=24, verbose=True,
    save_prefix=save_pref,
    use_reparam=False,
)

print(f"\nKjøring 24 fullført.")
