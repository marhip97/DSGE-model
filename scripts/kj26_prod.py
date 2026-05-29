"""
Kjøring 26 — K&M-korreksjon: φ_I1=12.54 + φ_PQ=669 + psi_R fri + rho_s genuint estimert.

Endringer fra kj25 (PE-godkjent 2026-05-29, fullmakt):
  - φ_I1: 0.50 → K&M=12.54 (25× korrigert). nemo_complete_documentation_2019.pdf s.59.
    Effekt: investeringene er mye tregere → psi_R kan konvergere lavere uten B5-brudd.
  - φ_PQ: 300 → K&M=669 (2× korrigert). κ_P = ε(ε-1)/φ_PQ = 30/669 = 0.0448 (K&M).
    Effekt: flatere Phillips-kurve, nærmere K&M original.
  - psi_R: reaktivert som estimert parameter, Beta(2,2,[0.50,0.95]), sentrert ~0.73.
    kj25 festet psi_R=0.90. Med K&M φ_I1 er B5-grensen ukjent — MCMC bestemmer.
  - rho_s: kj25 satte alltid rho_s=0 i log_posterior (bug). kj26 estimerer genuint.
    Prior: Beta(2,2,[0.05,0.90]) uendret fra kj25.
  - phi_I2: Prior åpnet Normal(50,50,[1,400]) (kj25: Normal(8,4,[0.5,40])).
    K&M=165.66, kj25 estimat=11.58 — data bestemmer.
  - N_PARAMS: 17→18 (psi_R reaktivert)

Mål: B5 bestått OG full kvartalsmatch RMSE < 0.118 (kj25-benchmark).

Lagres til: data/results/chain_kj26_prod*
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from nemo.estimation.mcmc import (
    PARAM_NAMES, N_PARAMS, KM,
    build_H, build_Sv, OBS_NAMES,
    adaptive_mcmc_with_monitoring, log_posterior,
    SIGMA_RP_FIXED, SIGMA_A_FIXED, PHI_I1_KJ26_FIXED, PHI_U_FIXED,
    PHI_PQ_KJ26_FIXED,
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
obs_kols = [PI_OBS_COL if k == 'pi_obs' else k for k in OBS_NAMES]
pre  = obs_df[obs_df.index <= '2019-12-31'][obs_kols].values
post = obs_df[obs_df.index >= '2022-01-01'][obs_kols].values

print(f"Datafil: {datafil.name} (KPI-JAE)")
print(f"Pre={len(pre)} kv  Post={len(post)} kv")

# ── Warm start fra kj25 posterior + psi_R=0.74 ───────────────────────────────
partial_fil = rot / "data/results/chain_kj25_prod_partial.npy"
posterior_fil = rot / "data/results/chain_kj25_prod_posterior.json"

# kj25 hadde 17 parametere (uten psi_R)
old_names_kj25 = ['rho_A','rho_C','rho_O','rho_Ys','rho_rp','rho_H',
                   'sigma_C','sigma_O','sigma_Ys','sigma_i','sigma_P','sigma_H',
                   'psi_P1','psi_Y','gamma_p','phi_I2','rho_s']

import json

if posterior_fil.exists():
    print(f"\nWarm start fra kj25 posterior JSON: {posterior_fil.name}")
    with open(posterior_fil) as f:
        post_json = json.load(f)['summary']
    theta_start = np.zeros(N_PARAMS)
    post_std    = np.zeros(N_PARAMS)
    for i, n in enumerate(PARAM_NAMES):
        if n in old_names_kj25 and n in post_json:
            theta_start[i] = post_json[n]['mean']
            post_std[i]    = max(post_json[n]['std'], 0.001)
        elif n == 'psi_R':
            # Startverdi: K&M mimicking rule ω_R=0.6663, kj26 estimerer fritt
            theta_start[i] = 0.74
            post_std[i]    = 0.05
        else:
            theta_start[i] = KM.get(n, 0.5)
            post_std[i]    = 0.05
    print(f"  17 felles param fra kj25 posterior + psi_R=0.74 (K&M mimicking rule)")
elif partial_fil.exists():
    print(f"\nWarm start fra kj25 partial chain: {partial_fil.name}")
    ch25 = np.load(partial_fil)
    theta_start = np.zeros(N_PARAMS)
    post_std    = np.zeros(N_PARAMS)
    for i, n in enumerate(PARAM_NAMES):
        if n in old_names_kj25:
            j = old_names_kj25.index(n)
            theta_start[i] = ch25[:,j].mean()
            post_std[i]    = max(ch25[:,j].std(), 0.001)
        elif n == 'psi_R':
            theta_start[i] = 0.74
            post_std[i]    = 0.05
    print(f"  kj25: {ch25.shape[0]} trekk, 17 felles param + psi_R=0.74")
else:
    print("kj25 ikke funnet — bruker K&M")
    theta_start = np.array([KM.get(n, 0.5) for n in PARAM_NAMES])
    post_std    = np.array([max(KM.get(n, 0.5)*0.3, 0.001) for n in PARAM_NAMES])

# Sjekk at phi_I2 startverdi er innenfor ny prior [1, 400]
phi_I2_idx = PARAM_NAMES.index('phi_I2')
if theta_start[phi_I2_idx] < 1.0:
    theta_start[phi_I2_idx] = 11.6   # kj25 posterior mean
    post_std[phi_I2_idx]    = 3.0

print(f"\nKjøring 26: {N_PARAMS} parametere")
print(f"  φ_I1 fast={PHI_I1_KJ26_FIXED} (K&M)  φ_PQ fast={PHI_PQ_KJ26_FIXED} (K&M)")
print(f"  κ_P = {6*5/PHI_PQ_KJ26_FIXED:.5f} (K&M=0.04484)")
print(f"  φ_u fast={PHI_U_FIXED}  σ_A fast={SIGMA_A_FIXED}")
print(f"  psi_R fri: Beta(2,2,[0.50,0.95])  startverdi={theta_start[PARAM_NAMES.index('psi_R')]:.3f}")
print(f"  rho_s fri: Beta(2,2,[0.05,0.90])  startverdi={theta_start[PARAM_NAMES.index('rho_s')]:.3f}")
print(f"  phi_I2 fri: Normal(50,50,[1,400])  startverdi={theta_start[phi_I2_idx]:.1f}")

print(f"\nStartverdier:")
for i, n in enumerate(PARAM_NAMES):
    print(f"  {n:12s}: {theta_start[i]:.4f} (std={post_std[i]:.4f})")

# ── Sjekk startverdi ──────────────────────────────────────────────────────────
H  = build_H()
Sv = build_Sv()
lp0 = log_posterior(theta_start, H, Sv, pre, post)
print(f"\nStartverdi log-posterior: {lp0:.2f}")
if not np.isfinite(lp0):
    # Prøv psi_R=0.85 hvis 0.74 gir ustabilitet
    print("  psi_R=0.74 gir ustabilt — prøver psi_R=0.85")
    theta_start[PARAM_NAMES.index('psi_R')] = 0.85
    lp0 = log_posterior(theta_start, H, Sv, pre, post)
    print(f"  psi_R=0.85: lp={lp0:.2f}")
    if not np.isfinite(lp0):
        raise ValueError(f"Startverdi gir ikke-endelig lp={lp0} selv med psi_R=0.85.")

# ── MCMC ──────────────────────────────────────────────────────────────────────
save_pref = str(rot / "data/results/chain_kj26_prod")
print(f"\nLagrer til: {save_pref}*")
print("Starter kjøring 26 (200k produksjon + 15k burnin, K&M φ_I1/φ_PQ, psi_R/rho_s fri) ...")
print()

chain, lp_vec, meta = adaptive_mcmc_with_monitoring(
    Y_pre=pre, Y_post=post, H=H, Sv=Sv,
    theta_init=theta_start, post_std_init=post_std,
    n_production=200_000, burnin=15_000, adapt_every=500,
    check_every=10_000, max_recalib=6,
    psrf_thr=1.10, ess_pct_thr=0.02,
    scale_init=0.70, seed=26, verbose=True,
    save_prefix=save_pref,
    use_reparam=False,
)

print(f"\nKjøring 26 fullført.")
