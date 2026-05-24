"""
Kjøring 14 — produksjonskjøring, 21 parametere.

Endringer fra kj13 (21 param med phi_PQ):
  - phi_PQ fjernet igjen (svakt identifisert i kj13, [104,1089], KPI 0.21× NB)
  - kappa_M er nå fri: importpriskanal-koeffisient i NK Phillips-kurve
    (PE-godkjent 2026-05-24, Steg B). Prior: Normal(0.03, 0.03, [0.005, 0.20]).
    K&M: κ_M=0.03. G0[0,RER]=G0[0,PI_STAR]=-kappa_M.
    Høyere κ_M → sterkere RER→KPI-kanal → kan bedre KPI-amplituden.
  - Startpunkt lp=3573.99 (+20 vs kj13-slutt) — umiddelbar forbedring

Startverdi: kj13 posterior means for felles 20 parametere (ekskl. phi_PQ);
            kappa_M kaldt start = 0.030 (K&M).

Lagres til: data/results/chain_kj14_prod*
"""

import sys, json, os
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
datafil = (
    rot / "data/processed/nemo_data.csv"
    if (rot / "data/processed/nemo_data.csv").exists()
    else rot / "data/processed/nemo_data_faktisk_v2.csv"
)
print(f"Datafil: {datafil.name}")
obs_df = pd.read_csv(datafil, index_col=0, parse_dates=True)
pre  = obs_df[obs_df.index <= "2019-12-31"][OBS_NAMES].values
post = obs_df[obs_df.index >= "2022-01-01"][OBS_NAMES].values
print(f"Pre={len(pre)} kv  Post={len(post)} kv  Totalt={len(pre)+len(post)} kv")

H  = build_H()
Sv = build_Sv()

# ── Startverdi fra kj13 ────────────────────────────────────────────────────────
# kj13 hadde phi_PQ (idx 20); kj14 bytter phi_PQ med kappa_M.
# Laster fra chain direkte for å unngå sigma_A-summary-bug.
KJ13_NAMES = ['rho_A','rho_C','rho_O','rho_Ys','rho_rp','rho_H',
              'sigma_A','sigma_C','sigma_O','sigma_Ys','sigma_i','sigma_P','sigma_H',
              'psi_R','psi_P1','psi_Y','gamma_p','phi_I1','phi_I2','phi_u','phi_PQ']

kj13_chain_fil = rot / "data/results/chain_kj13_prod.npy"
print(f"\nLaster startverdi fra {kj13_chain_fil.name} ...")
kj13_chain = np.load(kj13_chain_fil)   # shape (200000, 21)
kj13_means = {name: float(kj13_chain[:, i].mean()) for i, name in enumerate(KJ13_NAMES)}
kj13_stds  = {name: float(kj13_chain[:, i].std())  for i, name in enumerate(KJ13_NAMES)}

print(f"\nKjøring 14: {N_PARAMS} parametere (sigma_rp fast={SIGMA_RP_FIXED})")

theta_start = np.zeros(N_PARAMS)
post_std    = np.zeros(N_PARAMS)

for i, name in enumerate(PARAM_NAMES):
    if name in kj13_means:
        theta_start[i] = kj13_means[name]
        post_std[i]    = max(kj13_stds[name], 1e-4)
    else:
        # Ny parameter: kappa_M — kaldt start ved K&M-verdi
        theta_start[i] = KM.get(name, 0.03)
        post_std[i]    = 0.01   # startskala ~33% av K&M-verdi

print("\nStartverdier:")
for i, name in enumerate(PARAM_NAMES):
    marker = " ← ny" if name not in kj13_means else ""
    print(f"  {name:12s}: {theta_start[i]:.4f}  (std={post_std[i]:.4f}){marker}")
print(f"  {'sigma_rp':12s}: {SIGMA_RP_FIXED:.4f}  (FAST — ikke estimert)")

# ── Verifiser startpunkt ───────────────────────────────────────────────────────
lp0 = log_posterior(theta_start, H, Sv, pre, post)
print(f"\nStartverdi log-posterior: {lp0:.2f}")
if not np.isfinite(lp0):
    print("ADVARSEL: ugyldig startpunkt — bruker K&M-verdier.")
    theta_start = np.array([KM.get(n, 0.5) for n in PARAM_NAMES])
    post_std    = np.array([0.05] * N_PARAMS)

# ── Kjør ───────────────────────────────────────────────────────────────────────
save_pref = str(rot / "data/results/chain_kj14_prod")
print(f"\nLagrer til: {save_pref}*")
print("Starter kjøring 14 (200k produksjon + 20k burnin) ...\n")

chain, lp_vec, meta = adaptive_mcmc_with_monitoring(
    Y_pre=pre, Y_post=post, H=H, Sv=Sv,
    theta_init=theta_start, post_std_init=post_std,
    n_production=200_000, burnin=20_000, adapt_every=500,
    check_every=10_000, max_recalib=5,
    psrf_thr=1.10, ess_pct_thr=0.02,
    scale_init=0.676, seed=14, verbose=True,
    save_prefix=save_pref,
    use_reparam=False,
)
print("\nKjøring 14 fullført.")
