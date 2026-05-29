"""
Kjøring 29 — v3 + phi_I1=0.50 (fast via tight prior) + rho_H/phi_H1 fikset.

Endringer fra kj28 (Alt B krasjet, phi_I1→0.10+psi_R→0.99 feiler B5):

  MODELL: build_matrices_v3 (NZ=49) — tilbake til stabil v3
    Begrunnelse: Alt B + fri phi_I1 → phi_I1→0.10, psi_R→0.99 → by4>1.5 (B5 feil).
    v3 + phi_I1=0.50 + psi_R=0.99 gir by4=1.20× (B5 BESTÅTT ✓).
    Alt B bevares som exit-mulighet i build_matrices_altB.

  phi_I1 = 0.50 via tight prior (ikke K&M=12.54):
    Prior: Normal(0.50, 0.001, [0.40, 0.60]) — deltafunksjon rundt 0.50
    LL-sweep: phi_I1=0.50 gir LL=-3303 vs K&M=12.54: LL=-3262 (+41 enheter)
    B5: phi_I1=0.50 + psi_R=0.99 → by4=1.20×, bpi4=0.55× ✓

  phi_H1 fryst (v3 bruker ikke phi_H1):
    Prior: Normal(60.73, 0.001, [60.70, 60.76]) — hindrer vektorsøk i tom retning

  PRIOR-ENDRINGER globalt (fra kj28):
  - rho_H: Beta(5,3,[0.30,0.95]) — allerede fikset i mcmc.py
  Ingen ytterligere endringer i mcmc.py.

  N_PARAMS = 20 (uendret — phi_I1 og phi_H1 er i PARAM_NAMES men
                 frosset via prior_overrides)

  Mål: B5 BESTÅTT OG RMSE < 0.118 OG PSRF < 1.10
  Lagres til: data/results/chain_kj29_prod*
"""

import sys
import numpy as np
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from nemo.estimation.mcmc import (
    PARAM_NAMES, N_PARAMS, KM,
    build_Sv,
    adaptive_mcmc_with_monitoring, log_posterior,
    SIGMA_RP_FIXED, SIGMA_A_FIXED, PHI_U_FIXED, PHI_PQ_KJ26_FIXED,
)
from nemo.model.equations import build_matrices_v3  # NZ=49
from nemo.model.parameters import Parameters
import pandas as pd

rot = Path(__file__).parent.parent

# ── Data — KPI-JAE ────────────────────────────────────────────────────────────
datafil = rot / "data/processed/nemo_data_kpi_jae.csv"
if not datafil.exists():
    raise FileNotFoundError(f"{datafil} ikke funnet.")

from nemo.estimation.mcmc import OBS_NAMES
obs_df   = pd.read_csv(datafil, index_col=0, parse_dates=True)
PI_OBS_COL = "pi_core_obs"
obs_kols = [PI_OBS_COL if k == 'pi_obs' else k for k in OBS_NAMES]
pre  = obs_df[obs_df.index <= '2019-12-31'][obs_kols].values
post = obs_df[obs_df.index >= '2022-01-01'][obs_kols].values

print(f"Datafil: {datafil.name} (KPI-JAE)")
print(f"Pre={len(pre)} kv  Post={len(post)} kv")

# ── H-matrise for v3 (NZ=49) ─────────────────────────────────────────────────
from nemo.estimation.mcmc import build_H
H  = build_H()   # 14×49 for v3
Sv = build_Sv()

# ── prior_overrides: phi_I1=0.50 og phi_H1=60.73 frosset ────────────────────
PHI_I1_KJ29 = 0.50
PHI_H1_KJ29 = 60.73
prior_overrides = {
    'phi_I1': ('normal', PHI_I1_KJ29, 0.001, 0.40, 0.60),   # deltafunksjon
    'phi_H1': ('normal', PHI_H1_KJ29, 0.001, 60.70, 60.76),  # v3 bruker ikke phi_H1
}

# ── Warm start fra kj26 posterior (18 param base) ────────────────────────────
posterior_fil = rot / "data/results/chain_kj26_prod_posterior.json"
kj26_names = ['rho_A','rho_C','rho_O','rho_Ys','rho_rp','rho_H',
              'sigma_C','sigma_O','sigma_Ys','sigma_i','sigma_P','sigma_H',
              'psi_R','psi_P1','psi_Y','gamma_p','phi_I2','rho_s']

print(f"\nN_PARAMS = {N_PARAMS}  (phi_I1 og phi_H1 frosset via tight prior)")

if posterior_fil.exists():
    print(f"Warm start fra kj26 posterior: {posterior_fil.name}")
    with open(posterior_fil) as f:
        post_json = json.load(f)['summary']
    theta_start = np.zeros(N_PARAMS)
    post_std    = np.zeros(N_PARAMS)
    for i, n in enumerate(PARAM_NAMES):
        if n in post_json:
            theta_start[i] = post_json[n]['mean']
            post_std[i]    = max(post_json[n]['std'], 0.001)
        elif n == 'phi_I1':
            theta_start[i] = PHI_I1_KJ29
            post_std[i]    = 0.001
        elif n == 'phi_H1':
            theta_start[i] = PHI_H1_KJ29
            post_std[i]    = 0.001
        else:
            theta_start[i] = KM.get(n, 0.5)
            post_std[i]    = 0.05
else:
    print("kj26 ikke funnet — bruker K&M + phi_I1=0.50")
    theta_start = np.array([KM.get(n, 0.5) for n in PARAM_NAMES])
    theta_start[PARAM_NAMES.index('phi_I1')] = PHI_I1_KJ29
    theta_start[PARAM_NAMES.index('phi_H1')] = PHI_H1_KJ29
    post_std    = np.array([0.05]*N_PARAMS)

# ── Juster rho_H startverdi til ny prior [0.30, 0.95] ────────────────────────
rho_H_idx = PARAM_NAMES.index('rho_H')
if theta_start[rho_H_idx] < 0.30 or theta_start[rho_H_idx] > 0.95:
    theta_start[rho_H_idx] = 0.694   # K&M
    post_std[rho_H_idx]    = 0.08

phi_I2_idx = PARAM_NAMES.index('phi_I2')
if theta_start[phi_I2_idx] < 1.0:
    theta_start[phi_I2_idx] = 64.5
    post_std[phi_I2_idx]    = 20.0

# ── Rapporter ─────────────────────────────────────────────────────────────────
print(f"\nKjøring 29: {N_PARAMS} param (v3, NZ=49)")
print(f"  phi_I1 = {theta_start[PARAM_NAMES.index('phi_I1')]:.4f} (frosset via Normal(0.50,0.001))")
print(f"  phi_H1 = {theta_start[PARAM_NAMES.index('phi_H1')]:.4f} (frosset, v3 bruker ikke denne)")
print(f"  rho_H  = {theta_start[rho_H_idx]:.4f}  (prior: Beta(5,3,[0.30,0.95]))")
print(f"  psi_R  = {theta_start[PARAM_NAMES.index('psi_R')]:.4f}  (prior: Beta(2,2,[0.50,0.99]))")

print(f"\nStartverdier:")
for i, n in enumerate(PARAM_NAMES):
    print(f"  {n:12s}: {theta_start[i]:.4f} (std={post_std[i]:.4f})")

# ── Startverdi B5-sjekk ───────────────────────────────────────────────────────
from nemo.model.equations import Y, PI, I_R, E_i
from nemo.solver.blanchard_kahn import solve, compute_irf
p = Parameters()
for i, n in enumerate(PARAM_NAMES):
    if hasattr(p, n): setattr(p, n, float(theta_start[i]))
G0, G1, Psi, Pi = build_matrices_v3(p)
T, R, d = solve(G0, G1, Psi, Pi, verbose=False)
irf_raw = compute_irf(T, R, E_i, 0.0025, T_periods=20)
peak = float(np.max(irf_raw[:, I_R]))
irf  = irf_raw / peak
by4  = irf[3, Y] / (-0.450)
bpi4 = irf[3, PI] / (-0.150)
print(f"\nB5 ved start: by4={by4:.4f}  bpi4={bpi4:.4f}  BESTÅTT={0.8<=by4<=1.5 and bpi4>=0.35}")

# ── Startverdi log-posterior ─────────────────────────────────────────────────
lp0 = log_posterior(theta_start, H, Sv, pre, post,
                    build_fn=None, prior_overrides=prior_overrides)
print(f"Startverdi log-posterior: {lp0:.2f}")
if not np.isfinite(lp0):
    raise ValueError(f"Ikke-endelig lp={lp0} ved startverdi. Avbryter.")

# ── MCMC ──────────────────────────────────────────────────────────────────────
save_pref = str(rot / "data/results/chain_kj29_prod")
print(f"\nLagrer til: {save_pref}*")
print("Starter kjøring 29 (200k produksjon + 15k burnin, v3, NZ=49, N=20) ...")
print()

chain, lp_vec, meta = adaptive_mcmc_with_monitoring(
    Y_pre=pre, Y_post=post, H=H, Sv=Sv,
    theta_init=theta_start, post_std_init=post_std,
    n_production=200_000, burnin=15_000, adapt_every=500,
    check_every=10_000, max_recalib=6,
    psrf_thr=1.10, ess_pct_thr=0.02,
    scale_init=0.70, seed=29, verbose=True,
    save_prefix=save_pref,
    build_fn=None,
    prior_overrides=prior_overrides,
    use_reparam=False,
)

print(f"\nKjøring 29 fullført.")
