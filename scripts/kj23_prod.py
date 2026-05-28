"""
Kjøring 23 — kappa_P-fix + warm start fra kj22, seed=23.

Identisk med kj22, men:
  - Warm start fra kj22 26k-posterior means (lp=-2658, allerede nær modus)
  - Kortere burn-in (10k i stedet for 20k) — starter nær modus
  - seed=23

Kontekst (2026-05-28):
  - kj22 ble avbrutt av container etter 26k/200k produksjonstrekk
  - B5-normalisering rettet: korrekt formel 4×Y[q4]/peak / (-0.45)
  - Med K&M base + kP=0.0448 + psi_R=0.95: BNP=1.046×, KPI=0.465× (begge OK)
  - kj22 26k-posterior: BNP=2.32× (for stor — phi_u=1.715 er ikke-konvergert)
  - Forventet ved konvergens: BNP∈[0.8,1.5]×, KPI≥0.35×

Mål: BNP q4-ratio [0.8,1.5]× NB OG KPI q4-ratio ≥0.35× NB.

Lagres til: data/results/chain_kj23_prod*
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from nemo.estimation.mcmc import (
    PARAM_NAMES, N_PARAMS, KM,
    build_H, build_Sv, OBS_NAMES,
    adaptive_mcmc_with_monitoring, log_posterior,
    SIGMA_RP_FIXED, SIGMA_A_FIXED, PHI_I1_FIXED,
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

# ── Warm start fra kj22 26k-posterior ─────────────────────────────────────────
partial_fil = rot / "data/results/chain_kj22_prod_partial.npy"
if partial_fil.exists():
    print(f"\nWarm start fra kj22 26k-posterior: {partial_fil.name}")
    ch_kj22 = np.load(partial_fil)
    theta_start = ch_kj22.mean(axis=0)
    post_std    = np.maximum(ch_kj22.std(axis=0), 0.001)
    print(f"  lp(startverdi): sjekkes nedenfor")
else:
    print(f"\nKj22-partial ikke funnet — bruker K&M-defaults")
    theta_start = np.array([KM.get(n, 0.5) for n in PARAM_NAMES])
    post_std    = np.array([max(KM.get(n, 0.5) * 0.30, 0.001) for n in PARAM_NAMES])

print(f"\nKjøring 23: {N_PARAMS} parametere")
print(f"  sigma_rp fast={SIGMA_RP_FIXED}  sigma_A fast={SIGMA_A_FIXED}")
print(f"  phi_I1 fast={PHI_I1_FIXED}  rho_s fast=0.0")
print(f"  psi_R fri: Beta(2,3,[0.01,0.970])")
print(f"  kappa_P = {Parameters.kappa_P():.5f} (ε(ε-1)/φ_PQ)")
print(f"  v3-matriser (NZ=49)")

print(f"\nStartverdier (kj22 warm start):")
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
save_pref = str(rot / "data/results/chain_kj23_prod")
print(f"\nLagrer til: {save_pref}*")
print(f"Starter kjøring 23 (200k produksjon + 10k burnin, warm start) ...")
print()

chain, lp_vec, meta = adaptive_mcmc_with_monitoring(
    Y_pre=pre, Y_post=post, H=H, Sv=Sv,
    theta_init=theta_start, post_std_init=post_std,
    n_production=200_000, burnin=10_000, adapt_every=500,
    check_every=10_000, max_recalib=5,
    psrf_thr=1.10, ess_pct_thr=0.02,
    scale_init=0.81, seed=23, verbose=True,
    save_prefix=save_pref,
    use_reparam=False,
)

print(f"\nKjøring 23 fullført.")
