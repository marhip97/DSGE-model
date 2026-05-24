"""
Kjøring 17 — Test C: Kun pre-COVID periode (≤2019Q4), 100k eksplorerende.

Hypotese: Post-COVID (2022Q1–2025Q3, 15 kv) inneholder kostnadspress-inflasjon
(sigma_P-sjokk) som konkurrerer med psi_P1-signalet for pengepolitisk transmisjon.
Med kun pre-COVID data (75 kv) er datagrunnlaget renere for å identifisere
psi_P1 (Taylor-regel KPI-koeffisient).

Metodikk (PE-godkjent 2026-05-24):
  - pre = 75 kvartaler (2001Q2–2019Q4) — uendret
  - post = tom (0 kvartaler) — ekskluderer COVID-gjenopphentingsperioden
  - Standard H og Sv (14 observasjoner, full sett)
  - Kalman-filteret håndterer tom post-array (NaN-safe)

Startverdi: kj12 posterior means, 20 parametere.

Beslutningspunkt C:
  KPI-ratio >= 0.35× NB → Post-COVID kontaminerer, bruk pre-only i kj18
  KPI-ratio <  0.35× NB → Pre+post gir bedre identifikasjon enn pre alene

Lagres til: data/results/chain_kj17_prod*
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

# ── Data — kun pre-COVID ───────────────────────────────────────────────────────
rot = __import__('pathlib').Path(__file__).parent.parent
datafil = (
    rot / "data/processed/nemo_data.csv"
    if (rot / "data/processed/nemo_data.csv").exists()
    else rot / "data/processed/nemo_data_faktisk_v2.csv"
)
print(f"Datafil: {datafil.name}")
obs_df = pd.read_csv(datafil, index_col=0, parse_dates=True)

pre  = obs_df[obs_df.index <= "2019-12-31"][OBS_NAMES].values
post = np.empty((0, len(OBS_NAMES)))   # tom post-periode — Test C
print(f"Pre={len(pre)} kv  Post={len(post)} kv  Totalt={len(pre)+len(post)} kv")
print("Test C: kun pre-COVID, post-array er tom")

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

print(f"\nKjøring 17: {N_PARAMS} parametere (sigma_rp fast={SIGMA_RP_FIXED})")

theta_start = np.array([kj12_means[n] for n in PARAM_NAMES])
post_std    = np.array([max(kj12_stds[n], 1e-4) for n in PARAM_NAMES])

lp0 = log_posterior(theta_start, H, Sv, pre, post)
print(f"\nStartverdi log-posterior: {lp0:.2f}")
if not np.isfinite(lp0):
    print("ADVARSEL: ugyldig startpunkt — bruker K&M-verdier.")
    theta_start = np.array([KM.get(n, 0.5) for n in PARAM_NAMES])
    post_std    = np.array([0.05] * N_PARAMS)

# ── Kjør ───────────────────────────────────────────────────────────────────────
save_pref = str(rot / "data/results/chain_kj17_prod")
print(f"\nLagrer til: {save_pref}*")
print("Starter kjøring 17 (100k eksplorerende + 20k burnin) ...\n")

chain, lp_vec, meta = adaptive_mcmc_with_monitoring(
    Y_pre=pre, Y_post=post, H=H, Sv=Sv,
    theta_init=theta_start, post_std_init=post_std,
    n_production=100_000, burnin=20_000, adapt_every=500,
    check_every=10_000, max_recalib=5,
    psrf_thr=1.10, ess_pct_thr=0.02,
    scale_init=0.676, seed=17, verbose=True,
    save_prefix=save_pref,
    use_reparam=False,
)
print("\nKjøring 17 fullført.")
