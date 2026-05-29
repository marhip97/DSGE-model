"""
Kjøring 31 — v3 + rho_A/C/O/Ys/rp frosset ved K&M + phi_I1=0.50 frosset.

Endringer fra kj30 (oscillerende PSRF=1.09–1.19, ESS=26–50 — svak identifikasjon):

  DIAGNOSE: rho_A/C/O/Ys/rp er svakt identifiserte i DSGE-modellen.
    Kalman-filteret observerer lineære kombinasjoner av sjokk; det er
    vanskelig å skille f.eks. høy rho_C*sigma_C fra høy rho_O*sigma_O.
    Posteriorflaten er flat og korrelert i rho_*-rommet.
    Uansett prior (Beta(2,0.5) eller Beta(5,3)): ESS≈40–50 (behov: 200).
    Resultatet er at PSRF oscillerer 1.09↔1.19 uten konvergens.

  FIX: Frys rho_A/C/O/Ys/rp ved K&M-verdier via tight Normal priors.
    Forankring: K&M (2019) estimerer disse på norske data (Tabell 1, side 15).
    Verdiene er:
      rho_A  = 0.804  (Technology shock)
      rho_C  = 0.725  (Consumption preference)
      rho_O  = 0.874  (Oil price)
      rho_Ys = 0.783  (Foreign output)
      rho_rp = 0.737  (Risk premium)
    Prior: Normal(K&M, 0.001, [K&M-0.05, K&M+0.05]) — praktisk delta-funksjon.
    Justifikasjon: PROSJEKTPLAN.md — K&M-parameterisering er referansen.
    Kun prior_overrides — global PARAM_PRIORS uendret.

  Effektiv dimensjon: 20 param (N_PARAMS=20), men 7 effektivt frosset
    (phi_I1, phi_H1, rho_A, rho_C, rho_O, rho_Ys, rho_rp).
    Frie: rho_H, sigma_*, psi_R, psi_P1, psi_Y, gamma_p, phi_I2, rho_s (13 fri).

  BEHOLDT FRA kj30:
    phi_I1=0.50, phi_H1=60.73 frosset via tight prior
    build_matrices_v3 (NZ=49)
    rho_H: Beta(5,3,[0.30,0.95])

  Warm start: kj30 posterior → kj29 → kj26 → K&M fallback.

  Mål: B5 BESTÅTT OG RMSE < 0.118 OG PSRF < 1.10 OG ESS/n > 0.02
  Lagres til: data/results/chain_kj31_prod*
"""

import sys
import numpy as np
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from nemo.estimation.mcmc import (
    PARAM_NAMES, N_PARAMS, KM,
    build_H, build_Sv,
    adaptive_mcmc_with_monitoring, log_posterior,
    SIGMA_RP_FIXED, SIGMA_A_FIXED, PHI_U_FIXED, PHI_PQ_KJ26_FIXED,
)
from nemo.model.equations import build_matrices_v3
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
H  = build_H()
Sv = build_Sv()

# ── K&M-verdier for frosne rho-parametere (Tabell 1, K&M 2019) ──────────────
RHO_KM = {
    'rho_A':  KM['rho_A'],   # 0.804
    'rho_C':  KM['rho_C'],   # 0.725
    'rho_O':  KM['rho_O'],   # 0.874
    'rho_Ys': KM['rho_Ys'],  # 0.783
    'rho_rp': KM['rho_rp'],  # 0.737
}

# ── prior_overrides: rho_A/C/O/Ys/rp frosset + phi_I1/H1 frosset ─────────────
PHI_I1_KJ31 = 0.50
PHI_H1_KJ31 = 60.73
prior_overrides = {
    'phi_I1': ('normal', PHI_I1_KJ31, 0.001, 0.40, 0.60),
    'phi_H1': ('normal', PHI_H1_KJ31, 0.001, 60.70, 60.76),
}
for name, km_val in RHO_KM.items():
    prior_overrides[name] = ('normal', km_val, 0.001,
                             round(km_val - 0.05, 3),
                             round(km_val + 0.05, 3))

# ── Warm start: kj30 → kj29 → kj26 → K&M fallback ──────────────────────────
print(f"\nN_PARAMS = {N_PARAMS}  (7 frosset: rho_A/C/O/Ys/rp + phi_I1 + phi_H1)")

theta_start = None
post_std    = None

for posterior_fil, label in [
    (rot / "data/results/chain_kj30_prod_posterior.json", "kj30"),
    (rot / "data/results/chain_kj29_prod_posterior.json", "kj29"),
    (rot / "data/results/chain_kj26_prod_posterior.json", "kj26"),
]:
    if posterior_fil.exists():
        print(f"Warm start fra {label} posterior: {posterior_fil.name}")
        with open(posterior_fil) as f:
            post_json = json.load(f)['summary']
        theta_start = np.zeros(N_PARAMS)
        post_std    = np.zeros(N_PARAMS)
        for i, n in enumerate(PARAM_NAMES):
            if n in post_json:
                theta_start[i] = post_json[n]['mean']
                post_std[i]    = max(post_json[n]['std'], 0.001)
            elif n == 'phi_I1':
                theta_start[i] = PHI_I1_KJ31; post_std[i] = 0.001
            elif n == 'phi_H1':
                theta_start[i] = PHI_H1_KJ31; post_std[i] = 0.001
            else:
                theta_start[i] = KM.get(n, 0.5); post_std[i] = 0.05
        break

if theta_start is None:
    print("Ingen posterior funnet — bruker K&M")
    theta_start = np.array([KM.get(n, 0.5) for n in PARAM_NAMES])
    theta_start[PARAM_NAMES.index('phi_I1')] = PHI_I1_KJ31
    theta_start[PARAM_NAMES.index('phi_H1')] = PHI_H1_KJ31
    post_std = np.array([0.05]*N_PARAMS)

# ── Sett frosne rho-verdier eksplisitt ved K&M ────────────────────────────────
for name, km_val in RHO_KM.items():
    i = PARAM_NAMES.index(name)
    theta_start[i] = km_val
    post_std[i]    = 0.001

# ── Juster øvrige startverdier ────────────────────────────────────────────────
rho_H_idx  = PARAM_NAMES.index('rho_H')
phi_I2_idx = PARAM_NAMES.index('phi_I2')

if theta_start[rho_H_idx] < 0.30 or theta_start[rho_H_idx] > 0.95:
    theta_start[rho_H_idx] = 0.694; post_std[rho_H_idx] = 0.08

if theta_start[phi_I2_idx] < 1.0:
    theta_start[phi_I2_idx] = 64.5; post_std[phi_I2_idx] = 20.0

# ── Rapporter ─────────────────────────────────────────────────────────────────
print(f"\nKjøring 31: {N_PARAMS} param (v3, NZ=49) — 7 frosset, 13 fri")
for name, km_val in RHO_KM.items():
    print(f"  {name:8s} = {km_val:.4f} (frosset Normal({km_val:.3f},0.001), K&M Tabell 1)")
print(f"  phi_I1   = {theta_start[PARAM_NAMES.index('phi_I1')]:.4f} (frosset Normal(0.50,0.001))")
print(f"  phi_H1   = {theta_start[PARAM_NAMES.index('phi_H1')]:.4f} (fryst)")
print(f"  rho_H    = {theta_start[rho_H_idx]:.4f}  (fri, Beta(5,3,[0.30,0.95]))")
print(f"  psi_R    = {theta_start[PARAM_NAMES.index('psi_R')]:.4f}  (fri, Beta(2,2,[0.50,0.99]))")

print(f"\nStartverdier:")
for i, n in enumerate(PARAM_NAMES):
    print(f"  {n:12s}: {theta_start[i]:.4f} (std={post_std[i]:.4f})")

# ── B5-sjekk ─────────────────────────────────────────────────────────────────
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
save_pref = str(rot / "data/results/chain_kj31_prod")
print(f"\nLagrer til: {save_pref}*")
print("Starter kjøring 31 (200k produksjon + 15k burnin, v3, NZ=49, N=20) ...")
print()

chain, lp_vec, meta = adaptive_mcmc_with_monitoring(
    Y_pre=pre, Y_post=post, H=H, Sv=Sv,
    theta_init=theta_start, post_std_init=post_std,
    n_production=200_000, burnin=15_000, adapt_every=500,
    check_every=10_000, max_recalib=6,
    psrf_thr=1.10, ess_pct_thr=0.02,
    scale_init=0.70, seed=31, verbose=True,
    save_prefix=save_pref,
    build_fn=None,
    prior_overrides=prior_overrides,
    use_reparam=False,
)

print(f"\nKjøring 31 fullført.")
