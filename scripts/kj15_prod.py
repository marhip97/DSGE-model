"""
Kjøring 15 — Test A: fjern i_3m_obs (dobbel rentevariabel), 100k eksplorerende.

Hypotese: H[7,I_R]=4 (i_R_obs) OG H[8,I_R]=4 (i_3m_obs) dobbelvekter rentebanen.
Sterk I_R-identifikasjon tvinger psi_R→0.95, noe som supprimerer psi_P1.
Med kun én rentevariabel (i_R_obs) bør psi_R synke og psi_P1 stige → KPI-ratio ↑.

Metodikk (PE-godkjent 2026-05-24):
  - 13 observasjoner (fjerner i_3m_obs fra 14-obs-settet)
  - OBS_NAMES_NO_I3M, build_H_no_i3m(), build_Sv_no_i3m()
  - kappa_M tilbake til K&M-fast (kj14 viste estimering gir KPI 0.13× NB)

Startverdi: kj12 posterior means, 20 parametere (laster fra chain).
Baseline: kj12 KPI q4 = 0.20× NB Memo 3/2024.

Beslutningspunkt A:
  KPI-ratio >= 0.35× → Test A hjelper, inkluder i kj18
  KPI-ratio <  0.35× → Dobbel rente er ikke årsaken

Lagres til: data/results/chain_kj15_prod*
"""

import sys
import numpy as np

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent / 'src'))

from nemo.estimation.mcmc import (
    PARAM_NAMES, N_PARAMS, KM,
    build_H_no_i3m, build_Sv_no_i3m, OBS_NAMES_NO_I3M,
    adaptive_mcmc_with_monitoring, log_posterior,
    SIGMA_RP_FIXED,
)
import pandas as pd

# ── Data ───────────────────────────────────────────────────────────────────────
rot = __import__('pathlib').Path(__file__).parent.parent
datafil = (
    rot / "data/processed/nemo_data.csv"
    if (rot / "data/processed/nemo_data.csv").exists()
    else rot / "data/processed/nemo_data_faktisk_v2.csv"
)
print(f"Datafil: {datafil.name}")
obs_df = pd.read_csv(datafil, index_col=0, parse_dates=True)
pre  = obs_df[obs_df.index <= "2019-12-31"][OBS_NAMES_NO_I3M].values
post = obs_df[obs_df.index >= "2022-01-01"][OBS_NAMES_NO_I3M].values
print(f"Pre={len(pre)} kv  Post={len(post)} kv  Totalt={len(pre)+len(post)} kv")
print(f"Observasjoner: {len(OBS_NAMES_NO_I3M)} (fjernet i_3m_obs)")

H  = build_H_no_i3m()
Sv = build_Sv_no_i3m()

# ── Startverdi fra kj12 ────────────────────────────────────────────────────────
# kj12 hadde 20 parametere (inkl. gamma_p, uten kappa_M) — identisk med kj15-basis.
KJ12_NAMES = ['rho_A','rho_C','rho_O','rho_Ys','rho_rp','rho_H',
              'sigma_A','sigma_C','sigma_O','sigma_Ys','sigma_i','sigma_P','sigma_H',
              'psi_R','psi_P1','psi_Y','gamma_p','phi_I1','phi_I2','phi_u']

kj12_chain_fil = rot / "data/results/chain_kj12_prod.npy"
print(f"\nLaster startverdi fra {kj12_chain_fil.name} ...")
kj12_chain = np.load(kj12_chain_fil)   # shape (200000, 20)
kj12_means = {name: float(kj12_chain[:, i].mean()) for i, name in enumerate(KJ12_NAMES)}
kj12_stds  = {name: float(kj12_chain[:, i].std())  for i, name in enumerate(KJ12_NAMES)}

print(f"\nKjøring 15: {N_PARAMS} parametere (sigma_rp fast={SIGMA_RP_FIXED}, kappa_M fast={KM['kappa_M']})")

theta_start = np.zeros(N_PARAMS)
post_std    = np.zeros(N_PARAMS)

for i, name in enumerate(PARAM_NAMES):
    theta_start[i] = kj12_means[name]
    post_std[i]    = max(kj12_stds[name], 1e-4)

print("\nStartverdier (kj12 posterior means):")
for i, name in enumerate(PARAM_NAMES):
    print(f"  {name:12s}: {theta_start[i]:.4f}  (std={post_std[i]:.4f})")
print(f"  {'sigma_rp':12s}: {SIGMA_RP_FIXED:.4f}  (FAST)")
print(f"  {'kappa_M':12s}: {KM['kappa_M']:.4f}  (FAST K&M)")

# ── Verifiser startpunkt ───────────────────────────────────────────────────────
lp0 = log_posterior(theta_start, H, Sv, pre, post)
print(f"\nStartverdi log-posterior: {lp0:.2f}")
if not np.isfinite(lp0):
    print("ADVARSEL: ugyldig startpunkt — bruker K&M-verdier.")
    theta_start = np.array([KM.get(n, 0.5) for n in PARAM_NAMES])
    post_std    = np.array([0.05] * N_PARAMS)

# ── Kjør ───────────────────────────────────────────────────────────────────────
save_pref = str(rot / "data/results/chain_kj15_prod")
print(f"\nLagrer til: {save_pref}*")
print("Starter kjøring 15 (100k eksplorerende + 20k burnin) ...\n")

chain, lp_vec, meta = adaptive_mcmc_with_monitoring(
    Y_pre=pre, Y_post=post, H=H, Sv=Sv,
    theta_init=theta_start, post_std_init=post_std,
    n_production=100_000, burnin=20_000, adapt_every=500,
    check_every=10_000, max_recalib=5,
    psrf_thr=1.10, ess_pct_thr=0.02,
    scale_init=0.676, seed=15, verbose=True,
    save_prefix=save_pref,
    use_reparam=False,
)
print("\nKjøring 15 fullført.")
