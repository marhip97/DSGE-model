"""
Kjøring 25 — full kvartalsmatch: psi_R=0.90 fast, rho_s fri, phi_PQ=300, seed=25.

Endringer fra kj24 (PE-godkjent 2026-05-28):
  - psi_R festes til 0.90 (halvtid 6kv vs 18kv). Full-kvartalsmatch RMSE: 0.258→0.170 (-34%).
    B5 bestått ved psi_R=0.90: BNP=0.84×✅, KPI=0.81×✅ (psi_R fjernet fra PARAM_NAMES).
  - rho_s reaktiveres: Beta(2,2,[0.05,0.90]). Diagnostikk: rho_s=0.50 halverer RER-avvik.
    kj19 fant rho_s≈0.009 med gammel spec; ny model (κ_P-fix, phi_u-fix) gir bedre identifikasjon.
  - phi_PQ festes til 300 (κ_P=0.10, 2× K&M). KPI q4: -0.072→-0.141% (nær NB -0.15%).
  - N_PARAMS=17 (psi_R ut, rho_s inn, phi_PQ fast)
  - Warm start: kj24 200k-posterior (16 felles param) + rho_s=0.50

Diagnostikk (sweepresultater med kj24 øvrige param):
  psi_R=0.93+rho_s=0.50+φPQ=300: RMSE=0.170 (beste totalmatch)
  psi_R=0.90+rho_s=0.50+φPQ=200: RMSE=0.170 (nesten identisk)

Mål: B5 bestått OG full kvartalsmatch RMSE < 0.20.

Lagres til: data/results/chain_kj25_prod*
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
    PHI_PQ_FIXED, PSI_R_KJ25_FIXED,
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

# ── Warm start fra kj24 + rho_s=0.50 ─────────────────────────────────────────
partial_fil = rot / "data/results/chain_kj24_prod.npy"
old_names_kj24 = ['rho_A','rho_C','rho_O','rho_Ys','rho_rp','rho_H',
                   'sigma_C','sigma_O','sigma_Ys','sigma_i','sigma_P','sigma_H',
                   'psi_R','psi_P1','psi_Y','gamma_p','phi_I2']

if partial_fil.exists():
    print(f"\nWarm start fra kj24: {partial_fil.name}")
    ch24 = np.load(partial_fil)
    theta_start = np.zeros(N_PARAMS)
    post_std    = np.zeros(N_PARAMS)
    for i, n in enumerate(PARAM_NAMES):
        if n in old_names_kj24:
            j = old_names_kj24.index(n)
            theta_start[i] = ch24[:,j].mean()
            post_std[i]    = max(ch24[:,j].std(), 0.001)
        elif n == 'rho_s':
            theta_start[i] = 0.50
            post_std[i]    = 0.10
    print(f"  kj24: {ch24.shape[0]} trekk, 16 felles param + rho_s=0.50")
else:
    print("kj24 ikke funnet — bruker K&M")
    theta_start = np.array([KM.get(n, 0.5) for n in PARAM_NAMES])
    post_std    = np.array([max(KM.get(n, 0.5)*0.3, 0.001) for n in PARAM_NAMES])

print(f"\nKjøring 25: {N_PARAMS} parametere")
print(f"  psi_R fast={PSI_R_KJ25_FIXED}  phi_PQ fast={PHI_PQ_FIXED} (κ_P={6*5/PHI_PQ_FIXED:.4f})")
print(f"  phi_I1 fast={PHI_I1_FIXED}  phi_u fast={PHI_U_FIXED}  sigma_A fast={SIGMA_A_FIXED}")
print(f"  rho_s fri: Beta(2,2,[0.05,0.90])  startverdi={theta_start[PARAM_NAMES.index('rho_s')]:.3f}")
print(f"  kappa_P = {Parameters.kappa_P.__func__(type('P',(),{'phi_PQ':PHI_PQ_FIXED,'eps_P':6.0})()):.5f} (med phi_PQ={PHI_PQ_FIXED})")

print(f"\nStartverdier:")
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
save_pref = str(rot / "data/results/chain_kj25_prod")
print(f"\nLagrer til: {save_pref}*")
print(f"Starter kjøring 25 (200k produksjon + 15k burnin, psi_R=0.90 fast, rho_s fri) ...")
print()

chain, lp_vec, meta = adaptive_mcmc_with_monitoring(
    Y_pre=pre, Y_post=post, H=H, Sv=Sv,
    theta_init=theta_start, post_std_init=post_std,
    n_production=200_000, burnin=15_000, adapt_every=500,
    check_every=10_000, max_recalib=6,
    psrf_thr=1.10, ess_pct_thr=0.02,
    scale_init=0.70, seed=25, verbose=True,
    save_prefix=save_pref,
    use_reparam=False,
)

print(f"\nKjøring 25 fullført.")
