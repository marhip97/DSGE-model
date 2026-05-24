"""
Kjøring 13 — produksjonskjøring, 21 parametere.

Endringer fra kj12 (20 param):
  - phi_PQ er nå fri: Rotemberg-prisjusteringskostnad
    (PE-godkjent 2026-05-24). Prior: Normal(669,300) / [50, 2000].
    kappa_P = (eps_P-1)/phi_PQ = 5/phi_PQ.
  - kj12 viste KPI-ratio=0.20× NB Memo 3/2024 → κ_P=0.0075 (phi_PQ=669) for flat.
  - Hvis data vil ha lavere phi_PQ → brattere Phillips-kurve → sterkere KPI-respons.

Startverdi: kj12 posterior means for felles 20 parametere; phi_PQ kaldt start = 669 (K&M).

Lagres til: data/results/chain_kj13_prod*
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

# ── Startverdi fra kj12 ────────────────────────────────────────────────────────
# Laster fra chain-filen direkte (ikke JSON) fordi kj12 JSON har en
# summary-overwrite-bug: sigma_A=0.006 (fast) i stedet for posteriorverdien ~0.013.
kj12_chain_fil = rot / "data/results/chain_kj12_prod.npy"
print(f"\nLaster startverdi fra {kj12_chain_fil.name} ...")
kj12_chain = np.load(kj12_chain_fil)   # shape (200000, 20) — 20 param (uten phi_PQ)

# kj12 PARAM_NAMES (20 param, phi_PQ ikke med)
KJ12_PARAM_NAMES = [n for n in PARAM_NAMES if n != 'phi_PQ']
kj12_means = {name: float(kj12_chain[:, i].mean()) for i, name in enumerate(KJ12_PARAM_NAMES)}
kj12_stds  = {name: float(kj12_chain[:, i].std())  for i, name in enumerate(KJ12_PARAM_NAMES)}

print(f"\nKjøring 13: {N_PARAMS} parametere (sigma_rp fast={SIGMA_RP_FIXED})")

theta_start = np.zeros(N_PARAMS)
post_std    = np.zeros(N_PARAMS)

for i, name in enumerate(PARAM_NAMES):
    if name in kj12_means:
        theta_start[i] = kj12_means[name]
        post_std[i]    = max(kj12_stds[name], 1e-4)
    else:
        # Ny parameter: phi_PQ — kaldt start ved K&M-verdi
        theta_start[i] = KM.get(name, 669.0)
        post_std[i]    = 100.0   # startskala ~15% av K&M-verdi

print("\nStartverdier:")
for i, name in enumerate(PARAM_NAMES):
    marker = " ← ny" if name not in kj12_means else ""
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
save_pref = str(rot / "data/results/chain_kj13_prod")
print(f"\nLagrer til: {save_pref}*")
print("Starter kjøring 13 (200k produksjon + 20k burnin) ...\n")

chain, lp_vec, meta = adaptive_mcmc_with_monitoring(
    Y_pre=pre, Y_post=post, H=H, Sv=Sv,
    theta_init=theta_start, post_std_init=post_std,
    n_production=200_000, burnin=20_000, adapt_every=500,
    check_every=10_000, max_recalib=5,
    psrf_thr=1.10, ess_pct_thr=0.02,
    scale_init=0.676, seed=13, verbose=True,
    save_prefix=save_pref,
    use_reparam=False,
)
print("\nKjøring 13 fullført.")
