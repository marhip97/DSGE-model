"""
Kjøring 16 — Test B: KPI-JAE i stedet for total KPI, 100k eksplorerende.

Hypotese: Total KPI er kontaminert av energipriser og avgiftsendringer.
sigma_O (oljeprissjokk) og sigma_rp absorberer energiinflasjonen.
Lite KPI-signal er igjen til psi_P1 via Taylor-regelen → psi_P1 lav.
NB Memo bruker KPI-JAE (u/energi og avgifter, SSB tabell 10235) — renere
inflasjonsmål for pengepolitisk transmisjon.

Metodikk (PE-godkjent 2026-05-24):
  - pi_core_obs (KPI-JAE) brukes i stedet for pi_obs (total KPI)
  - H-matrise identisk med build_H() — kun data-kolonnen bytttes
  - Krever nemo_data_faktisk_v2.csv med pi_core_obs-kolonne

MERK: SSB-API (data.ssb.no) er blokkert i skymiljøet.
Datasettet må bygges lokalt:
    python -c "
    from nemo.data.pipeline import bygg_datasett
    bygg_datasett(utfil='data/processed/nemo_data_kpi_jae.csv', bruk_kpi_jae=True)
    "
Etter bygging: kjør dette scriptet med datafil=nemo_data_kpi_jae.csv.

Startverdi: kj12 posterior means, 20 parametere.

Beslutningspunkt B:
  KPI-ratio >= 0.35× NB → Test B hjelper, inkluder i kj18
  KPI-ratio <  0.35× NB → Datatype ikke avgjørende

Lagres til: data/results/chain_kj16_prod*
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

# ── Data ───────────────────────────────────────────────────────────────────────
rot = __import__('pathlib').Path(__file__).parent.parent

# Bruk datafil med pi_core_obs (KPI-JAE) — må bygges lokalt
datafil_kpi_jae = rot / "data/processed/nemo_data_kpi_jae.csv"
datafil_standard = (
    rot / "data/processed/nemo_data.csv"
    if (rot / "data/processed/nemo_data.csv").exists()
    else rot / "data/processed/nemo_data_faktisk_v2.csv"
)

if datafil_kpi_jae.exists():
    datafil = datafil_kpi_jae
    PI_OBS_COL = "pi_core_obs"
    print(f"Datafil: {datafil.name} (bruker KPI-JAE)")
else:
    datafil = datafil_standard
    PI_OBS_COL = "pi_obs"
    print(f"ADVARSEL: {datafil_kpi_jae.name} ikke funnet — bruker total KPI")
    print("Kj16 uten KPI-JAE gir ikke meningsfullt Test B-resultat.")
    print("Bygg data lokalt med SSB-API før kjøring.")

obs_df = pd.read_csv(datafil, index_col=0, parse_dates=True)

# Bygg obs-array med pi_core_obs i stedet for pi_obs
obs_kols = [PI_OBS_COL if k == "pi_obs" else k for k in OBS_NAMES]
pre  = obs_df[obs_df.index <= "2019-12-31"][obs_kols].values
post = obs_df[obs_df.index >= "2022-01-01"][obs_kols].values
print(f"Pre={len(pre)} kv  Post={len(post)} kv  Totalt={len(pre)+len(post)} kv")
print(f"pi-observasjon: {PI_OBS_COL}")

H  = build_H()
Sv = build_Sv()

# ── Startverdi fra kj12 ────────────────────────────────────────────────────────
KJ12_NAMES = ['rho_A','rho_C','rho_O','rho_Ys','rho_rp','rho_H',
              'sigma_A','sigma_C','sigma_O','sigma_Ys','sigma_i','sigma_P','sigma_H',
              'psi_R','psi_P1','psi_Y','gamma_p','phi_I1','phi_I2','phi_u']

kj12_chain_fil = rot / "data/results/chain_kj12_prod.npy"
print(f"\nLaster startverdi fra {kj12_chain_fil.name} ...")
kj12_chain = np.load(kj12_chain_fil)
kj12_means = {name: float(kj12_chain[:, i].mean()) for i, name in enumerate(KJ12_NAMES)}
kj12_stds  = {name: float(kj12_chain[:, i].std())  for i, name in enumerate(KJ12_NAMES)}

print(f"\nKjøring 16: {N_PARAMS} parametere (sigma_rp fast={SIGMA_RP_FIXED})")

theta_start = np.array([kj12_means[n] for n in PARAM_NAMES])
post_std    = np.array([max(kj12_stds[n], 1e-4) for n in PARAM_NAMES])

lp0 = log_posterior(theta_start, H, Sv, pre, post)
print(f"\nStartverdi log-posterior: {lp0:.2f}")
if not np.isfinite(lp0):
    print("ADVARSEL: ugyldig startpunkt — bruker K&M-verdier.")
    theta_start = np.array([KM.get(n, 0.5) for n in PARAM_NAMES])
    post_std    = np.array([0.05] * N_PARAMS)

# ── Kjør ───────────────────────────────────────────────────────────────────────
save_pref = str(rot / "data/results/chain_kj16_prod")
print(f"\nLagrer til: {save_pref}*")
print("Starter kjøring 16 (100k eksplorerende + 20k burnin) ...\n")

chain, lp_vec, meta = adaptive_mcmc_with_monitoring(
    Y_pre=pre, Y_post=post, H=H, Sv=Sv,
    theta_init=theta_start, post_std_init=post_std,
    n_production=100_000, burnin=20_000, adapt_every=500,
    check_every=10_000, max_recalib=5,
    psrf_thr=1.10, ess_pct_thr=0.02,
    scale_init=0.676, seed=16, verbose=True,
    save_prefix=save_pref,
    use_reparam=False,
)
print("\nKjøring 16 fullført.")
