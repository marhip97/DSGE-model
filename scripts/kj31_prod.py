"""
Kjøring 31 — v3 + targeted rho_A/H prior-fix (kj30 residual convergence).

Endringer fra kj30 (max PSRF=1.695 — rho_A + rho_H feiler):

  DIAGNOSE kj30 (200k, 17/20 OK):
    Beta(5,3) fix fra kj30 virket for rho_C/O/Ys/rp:
      rho_C: ESS=816, PSRF=1.034 ✓ (K&M=0.725, posterior=0.229)
      rho_O: ESS=1140, PSRF=1.010 ✓ (K&M=0.874, posterior=0.240)
      rho_Ys: ESS=1250, PSRF=1.019 ✓ (K&M=0.783, posterior=0.346)
      rho_rp: ESS=3306, PSRF=1.004 ✓ (K&M=0.737, posterior=0.652)
    Gjenstående problemer:
      rho_A: PSRF=1.695 — Beta(2,2,[0.01,0.9995]) ikke fikset i kj30.
             Posterior=0.091, K&M=0.804 (data foretrekker mye lavere).
      rho_H: PSRF=1.202 — posterior=0.915 treffer øvre grense 0.95.
             Beta(5,3,[0.30,0.95]) → kjede spretter fra grensen.

  FIX 1: rho_A — Beta(5,3,[0.01,0.99]) via prior_overrides
    Fra: Beta(2,2,[0.01,0.9995]) — flat, mode=0.5, lar kjede vandre bredt
    Til: Beta(5,3,[0.01,0.99]) — mode=0.667, konsentrert, øvre grense 0.99
    Merk: data foretrekker rho_A≈0.09, langt fra K&M=0.804. Prioren er
    informatise nok til å hindre boundary-adferd men bred nok til å la
    data dominere.

  FIX 2: rho_H — Beta(5,2,[0.30,0.99]) via prior_overrides
    Fra: Beta(5,3,[0.30,0.95]) — mode=0.667, øvre grense 0.95
    Til: Beta(5,2,[0.30,0.99]) — mode=0.80, øvre grense 0.99
    kj30: rho_H posterior=0.915 traff 0.95-grensen → PSRF=1.202.
    Ny prior: mode=0.80 (nær kj30 posterior), tillater 0.30–0.99.
    Note: Beta(5,2) mode = (5-1)/(5+2-2) = 4/5 = 0.80.

  BEHOLDT FRA kj30:
    rho_C/O/Ys/rp: Beta(5,3,[0.10,0.99]) — allerede konvergerte ✓
    phi_I1=0.50, phi_H1=60.73 frosset
    build_matrices_v3 (NZ=49)

  Effektivt fri: rho_A, rho_C, rho_O, rho_Ys, rho_rp, rho_H,
                 sigma_*, psi_R, psi_P1, psi_Y, gamma_p, phi_I2, rho_s (18 fri)

  Warm start: kj30 posterior.

  Mål: B5 BESTÅTT OG RMSE < 0.118 OG PSRF < 1.10 (alle 20 parametere)
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

# ── prior_overrides ───────────────────────────────────────────────────────────
PHI_I1_KJ31 = 0.50
PHI_H1_KJ31 = 60.73
prior_overrides = {
    # Frosne (fra kj29)
    'phi_I1': ('normal', PHI_I1_KJ31, 0.001,  0.40,  0.60),
    'phi_H1': ('normal', PHI_H1_KJ31, 0.001, 60.70, 60.76),
    # kj30-fix (virket — beholdes)
    'rho_C':  ('beta', 5.0, 3.0, 0.10, 0.99),
    'rho_O':  ('beta', 5.0, 3.0, 0.10, 0.99),
    'rho_Ys': ('beta', 5.0, 3.0, 0.10, 0.99),
    'rho_rp': ('beta', 5.0, 3.0, 0.10, 0.99),
    # Ny kj31-fix
    'rho_A':  ('beta', 5.0, 3.0, 0.01, 0.99),  # Beta(2,2)→Beta(5,3), øvre 0.9995→0.99
    'rho_H':  ('beta', 5.0, 2.0, 0.30, 0.99),  # Beta(5,3,[0.30,0.95])→Beta(5,2,[0.30,0.99])
}

# ── Warm start: kj30 → kj29 → kj26 → K&M fallback ──────────────────────────
print(f"\nN_PARAMS = {N_PARAMS}  (phi_I1 + phi_H1 frosset; alle rho via prior_overrides)")

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

# ── Juster startverdier til prior-grenser ────────────────────────────────────
rho_A_idx  = PARAM_NAMES.index('rho_A')
rho_H_idx  = PARAM_NAMES.index('rho_H')
phi_I2_idx = PARAM_NAMES.index('phi_I2')

# rho_A: ny prior [0.01, 0.99] — kj30 posterior=0.091, innenfor grense
theta_start[rho_A_idx] = np.clip(theta_start[rho_A_idx], 0.011, 0.989)

# rho_H: ny prior [0.30, 0.99] — kj30 posterior=0.915, innenfor ny grense ✓
if theta_start[rho_H_idx] < 0.30 or theta_start[rho_H_idx] > 0.99:
    theta_start[rho_H_idx] = 0.915; post_std[rho_H_idx] = 0.02

# rho_C/O/Ys/rp: behold kj30 posterior men klipp til [0.10, 0.99]
for n, lo, hi in [('rho_C',0.10,0.99),('rho_O',0.10,0.99),
                  ('rho_Ys',0.10,0.99),('rho_rp',0.10,0.99)]:
    idx = PARAM_NAMES.index(n)
    v = theta_start[idx]
    if v <= lo or v >= hi:
        theta_start[idx] = KM.get(n, 0.5); post_std[idx] = 0.10

if theta_start[phi_I2_idx] < 1.0:
    theta_start[phi_I2_idx] = 64.5; post_std[phi_I2_idx] = 20.0

# ── Rapporter ─────────────────────────────────────────────────────────────────
print(f"\nKjøring 31: {N_PARAMS} param (v3, NZ=49)")
print(f"  rho_A  = {theta_start[rho_A_idx]:.4f}  (Beta(5,3,[0.01,0.99]) — ny fra kj31)")
print(f"  rho_H  = {theta_start[rho_H_idx]:.4f}  (Beta(5,2,[0.30,0.99]) — utvidet øvre grense)")
print(f"  phi_I1 = {theta_start[PARAM_NAMES.index('phi_I1')]:.4f} (frosset Normal(0.50,0.001))")
print(f"  phi_H1 = {theta_start[PARAM_NAMES.index('phi_H1')]:.4f} (fryst)")
print(f"  psi_R  = {theta_start[PARAM_NAMES.index('psi_R')]:.4f}  (Beta(2,2,[0.50,0.99]))")

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
