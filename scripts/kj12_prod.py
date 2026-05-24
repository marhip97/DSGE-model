"""
Kjøring 12 — produksjonskjøring, 20 parametere.

Endringer fra kj10 (19 param):
  - gamma_p er nå fri: Calvo-prisindeksasjon i hybrid NK Phillips-kurve
    (PE-godkjent 2026-05-24). Prior: Beta(3,3) sentrert ~0.5, [0, 0.95].
    K&M Tabell 8: γ_p ≈ 0.35. Fikser timing i KPI-respons (gradvis vs. umiddelbar).
  - psi_R tilbake til estimering med utvidet øvre grense 0.990 (fra 0.92).
    kj11 viste likelihood-fall 97 log-enheter med K&M=0.667 → data vil ha høy
    renteglatting; ny grense gir rom uten å begrense.
  - sigma_rp=0.006 fast (PE-godkjent 2026-05-24, kj10)

Startverdi: kj10 posterior means for felles parametere; gamma_p kaldt start = 0.35 (K&M).

Lagres til: data/results/chain_kj12_prod*
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

# ── Startverdi fra kj10 ────────────────────────────────────────────────────────
kj10_fil = rot / "data/results/chain_kj10_prod_posterior.json"
print(f"\nLaster startverdi fra {kj10_fil.name}...")
with open(kj10_fil) as f:
    kj10 = json.load(f)
kj10_summ = kj10["summary"]

print(f"\nKjøring 12: {N_PARAMS} parametere (sigma_rp fast={SIGMA_RP_FIXED})")

theta_start = np.zeros(N_PARAMS)
post_std    = np.zeros(N_PARAMS)

for i, name in enumerate(PARAM_NAMES):
    if name in kj10_summ:
        theta_start[i] = kj10_summ[name]["mean"]
        post_std[i]    = max(kj10_summ[name]["std"], 1e-4)
    else:
        # Ny parameter: gamma_p — kaldt start ved K&M-verdi
        theta_start[i] = KM.get(name, 0.35)
        post_std[i]    = 0.05

print("\nStartverdier:")
for i, name in enumerate(PARAM_NAMES):
    marker = " ← ny" if name not in kj10_summ else ""
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
save_pref = str(rot / "data/results/chain_kj12_prod")
print(f"\nLagrer til: {save_pref}*")
print("Starter kjøring 12 (200k produksjon + 20k burnin) ...\n")

chain, lp_vec, meta = adaptive_mcmc_with_monitoring(
    Y_pre=pre, Y_post=post, H=H, Sv=Sv,
    theta_init=theta_start, post_std_init=post_std,
    n_production=200_000, burnin=20_000, adapt_every=500,
    check_every=10_000, max_recalib=5,
    psrf_thr=1.10, ess_pct_thr=0.02,
    scale_init=0.676, seed=12, verbose=True,
    save_prefix=save_pref,
    use_reparam=False,
)
print("\nKjøring 12 fullført.")
