"""
Kjøring 19 — Fase 1B: AR(1)-glatting av RER i UIP (rho_s estimert), KPI-JAE.

Metodikk (PE-godkjent 2026-05-26):
  - Fase 1B: ny rho_s-parameter for AR(1)-glatting av valutakurs i UIP
    rer_t = rho_s·rer_{t-1} + (1-rho_s)·[E_t[rer_{t+1}] + UIP-ledd]
  - KPI-JAE (pi_core_obs) — eneste observasjonsvalg som gir KPI q4 ≥ 0.35× NB (kj16/kj18)
  - psi_R prior: Beta(2,3,[0.01,0.970]) — samme som kj18
  - Startverdi: kj18 posterior means + rho_s=0.40 (prior-mean)
  - 200k produksjon, seed=19

Diagnose fra kj18 (KPI-JAE, uten rho_s):
  - KPI q4: 0.40× NB (OK ≥ 0.35×)
  - BNP q4: 4.55× NB (USTABIL)
  - Årsak: sigma_H=0.310, sigma_C=0.116 driver BNP-overreaksjon
  Rho_s demper umiddelbar RER-respons → forventet at BNP-ratio faller mot [0.8, 1.5]×.

Lagres til: data/results/chain_kj19_prod*
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

obs_df = pd.read_csv(datafil, index_col=0, parse_dates=True)

PI_OBS_COL = "pi_core_obs"
if PI_OBS_COL not in obs_df.columns:
    raise ValueError(f"Kolonne {PI_OBS_COL} mangler i {datafil.name}")

obs_kols = [PI_OBS_COL if k == 'pi_obs' else k for k in OBS_NAMES]
pre  = obs_df[obs_df.index <= '2019-12-31'][obs_kols].values
post = obs_df[obs_df.index >= '2022-01-01'][obs_kols].values

print(f"Datafil: {datafil.name} (KPI-JAE)")
print(f"Pre={len(pre)} kv  Post={len(post)} kv  Totalt={len(pre)+len(post)} kv")
print(f"pi-observasjon: {PI_OBS_COL}")

# ── Startverdi — kj18 posterior means + rho_s=0.40 ───────────────────────────
kj18_fil = rot / "data/results/chain_kj18_prod.npy"
PARAM_NAMES_KJ18 = [n for n in PARAM_NAMES if n != 'rho_s']  # kj18 hadde 20 param
rho_s_idx = PARAM_NAMES.index('rho_s')

if kj18_fil.exists():
    print(f"\nLaster startverdi fra chain_kj18_prod.npy ...")
    kj18 = np.load(kj18_fil)
    kj18_means = {PARAM_NAMES_KJ18[i]: float(kj18[:, i].mean())
                  for i in range(len(PARAM_NAMES_KJ18))}
    kj18_stds  = {PARAM_NAMES_KJ18[i]: float(kj18[:, i].std())
                  for i in range(len(PARAM_NAMES_KJ18))}
    theta_start = np.array([
        kj18_means.get(n, KM.get(n, 0.5)) if n != 'rho_s' else 0.40
        for n in PARAM_NAMES
    ])
    post_std = np.array([
        max(kj18_stds.get(n, 0.05), 1e-4) if n != 'rho_s' else 0.10
        for n in PARAM_NAMES
    ])
    print(f"  rho_s startverdi: 0.40 (prior-mean, ny parameter)")
else:
    print("Advarsel: chain_kj18_prod.npy ikke funnet — bruker K&M + rho_s=0.40")
    theta_start = np.array([KM.get(n, 0.5) if n != 'rho_s' else 0.40
                             for n in PARAM_NAMES])
    post_std = np.array([0.05] * N_PARAMS)

# Korriger psi_R hvis over ny prior-grense (0.970)
psi_R_idx = PARAM_NAMES.index("psi_R")
if theta_start[psi_R_idx] > 0.969:
    theta_start[psi_R_idx] = 0.950
    print(f"  psi_R startverdi justert til 0.950 (kj18 hadde {kj18_means.get('psi_R',0):.4f} > 0.970)")

print(f"\nKjøring 19: {N_PARAMS} parametere")
print(f"  sigma_rp fast={SIGMA_RP_FIXED}  kappa_M fast=0.03")
print(f"  rho_s prior: Beta(2,2,[0.001,0.99])")
print(f"  psi_R prior: Beta(2,3,[0.01,0.970])  (PE-godkjent 2026-05-26)")

print(f"\nStartverdier:")
for i, n in enumerate(PARAM_NAMES):
    src = "kj18" if n != 'rho_s' else "prior-mean"
    print(f"  {n:12s}: {theta_start[i]:.4f}  std={post_std[i]:.4f}  ({src})")

# ── Sjekk startverdi ──────────────────────────────────────────────────────────
H  = build_H()
Sv = build_Sv()
lp0 = log_posterior(theta_start, H, Sv, pre, post)
print(f"\nStartverdi log-posterior: {lp0:.2f}")
if not np.isfinite(lp0):
    raise ValueError(f"Startverdi gir ikke-endelig lp={lp0}. Sjekk data og parametre.")

# ── MCMC ──────────────────────────────────────────────────────────────────────
save_pref = str(rot / "data/results/chain_kj19_prod")
print(f"\nLagrer til: {save_pref}*")
print(f"Starter kjøring 19 (200k produksjon + 20k burnin) ...")
print()

chain, lp_vec, meta = adaptive_mcmc_with_monitoring(
    Y_pre=pre, Y_post=post, H=H, Sv=Sv,
    theta_init=theta_start, post_std_init=post_std,
    n_production=200_000, burnin=20_000, adapt_every=500,
    check_every=10_000, max_recalib=5,
    psrf_thr=1.10, ess_pct_thr=0.02,
    scale_init=0.676, seed=19, verbose=True,
    save_prefix=save_pref,
    use_reparam=False,
)

print(f"\nKjøring 19 fullført.")
