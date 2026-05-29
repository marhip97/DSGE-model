"""
Kjøring 30 — v3 + phi_I1=0.50 frosset + rho_C/O/Ys/rp prior-fix.

Endringer fra kj29 (max_recalib med PSRF=1.28 — rho_* konvergensproblem):

  PROBLEM: rho_C/O/Ys/rp har Beta(2,0.5,[0.01,0.9995]) i PARAM_PRIORS.
    Beta(2,0.5): β<1 → PDF ubegrenset ved x=1 → mode ved x=1 (grensen).
    Fører til boundary-treff, høy autokorrelasjon, ESS=44 (behov: 200).
    kj29 oscillerte runde 5→6: PSRF 1.087→1.280 uten konvergens.

  FIX (via prior_overrides — global PARAM_PRIORS uendret):
    rho_C/O/Ys/rp: Beta(2,0.5,[0.01,0.9995]) → Beta(5,3,[0.10,0.99])
    Mode=0.667, std≈0.16, øvre grense 0.99 (ikke 0.9995 — unngår degenerert grense).
    K&M-verdier: rho_C=0.725, rho_O=0.874, rho_Ys=0.783, rho_rp=0.737.
    Beta(5,3) dekker K&M-verdiene innenfor ±1σ og hindrer boundary-adferd.

  BEHOLDT FRA kj29:
    phi_I1=0.50 via tight prior Normal(0.50,0.001,[0.40,0.60]) — B5 BESTÅTT
    phi_H1=60.73 fryst — v3 bruker ikke phi_H1
    build_matrices_v3 (NZ=49) — stabil løsning
    rho_H: Beta(5,3,[0.30,0.95]) (fra mcmc.py kj28-fix) — uendret

  Warm start: kj29 posterior (faller tilbake til kj26 posterior).

  Mål: B5 BESTÅTT OG RMSE < 0.118 OG PSRF < 1.10 OG ESS/n > 0.02
  Lagres til: data/results/chain_kj30_prod*
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
H  = build_H()   # 14×49
Sv = build_Sv()

# ── prior_overrides: phi_I1=0.50, phi_H1=60.73 frosset + rho-fix ─────────────
PHI_I1_KJ30 = 0.50
PHI_H1_KJ30 = 60.73
prior_overrides = {
    'phi_I1': ('normal', PHI_I1_KJ30, 0.001,  0.40,  0.60),   # delta ved 0.50
    'phi_H1': ('normal', PHI_H1_KJ30, 0.001, 60.70, 60.76),   # fryst, v3 bruker ikke
    # Beta(2,0.5) → Beta(5,3): mode 0→0.667, hindrer boundary-adferd
    'rho_C':  ('beta',   5.0, 3.0, 0.10, 0.99),
    'rho_O':  ('beta',   5.0, 3.0, 0.10, 0.99),
    'rho_Ys': ('beta',   5.0, 3.0, 0.10, 0.99),
    'rho_rp': ('beta',   5.0, 3.0, 0.10, 0.99),
}

# ── Warm start: kj29 → kj26 → K&M fallback ──────────────────────────────────
print(f"\nN_PARAMS = {N_PARAMS}  (phi_I1 og phi_H1 frosset via tight prior)")

theta_start = None
post_std    = None

for posterior_fil, label in [
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
                theta_start[i] = PHI_I1_KJ30
                post_std[i]    = 0.001
            elif n == 'phi_H1':
                theta_start[i] = PHI_H1_KJ30
                post_std[i]    = 0.001
            else:
                theta_start[i] = KM.get(n, 0.5)
                post_std[i]    = 0.05
        break

if theta_start is None:
    print("Ingen posterior funnet — bruker K&M + phi_I1=0.50")
    theta_start = np.array([KM.get(n, 0.5) for n in PARAM_NAMES])
    theta_start[PARAM_NAMES.index('phi_I1')] = PHI_I1_KJ30
    theta_start[PARAM_NAMES.index('phi_H1')] = PHI_H1_KJ30
    post_std    = np.array([0.05]*N_PARAMS)

# ── Juster startverdier til prior-grenser ────────────────────────────────────
rho_H_idx  = PARAM_NAMES.index('rho_H')
rho_C_idx  = PARAM_NAMES.index('rho_C')
rho_O_idx  = PARAM_NAMES.index('rho_O')
rho_Ys_idx = PARAM_NAMES.index('rho_Ys')
rho_rp_idx = PARAM_NAMES.index('rho_rp')
phi_I2_idx = PARAM_NAMES.index('phi_I2')

# rho_H: prior [0.30, 0.95]
if theta_start[rho_H_idx] < 0.30 or theta_start[rho_H_idx] > 0.95:
    theta_start[rho_H_idx] = 0.694
    post_std[rho_H_idx]    = 0.08

# rho_C/O/Ys/rp: ny prior [0.10, 0.99] — Beta(5,3) mode=0.667
# NB: kj29 drev rho_C/O til ~0.05 (utenfor ny range). Bruk K&M startverdi
# heller enn å klippe til grensen (grense → x=0 → logpdf=-inf).
RHO_OVERRIDES = {
    rho_C_idx:  (0.10, 0.99, KM.get('rho_C',  0.725)),
    rho_O_idx:  (0.10, 0.99, KM.get('rho_O',  0.874)),
    rho_Ys_idx: (0.10, 0.99, KM.get('rho_Ys', 0.783)),
    rho_rp_idx: (0.10, 0.99, KM.get('rho_rp', 0.737)),
}
for idx, (lo, hi, km_val) in RHO_OVERRIDES.items():
    v = theta_start[idx]
    if v <= lo or v >= hi:
        # Utenfor ny prior-grense — bruk K&M-verdi (innenfor [0.10, 0.99])
        theta_start[idx] = np.clip(km_val, lo + 1e-4, hi - 1e-4)
        post_std[idx]    = 0.10

if theta_start[phi_I2_idx] < 1.0:
    theta_start[phi_I2_idx] = 64.5
    post_std[phi_I2_idx]    = 20.0

# ── Rapporter ─────────────────────────────────────────────────────────────────
print(f"\nKjøring 30: {N_PARAMS} param (v3, NZ=49)")
print(f"  phi_I1 = {theta_start[PARAM_NAMES.index('phi_I1')]:.4f} (frosset Normal(0.50,0.001))")
print(f"  phi_H1 = {theta_start[PARAM_NAMES.index('phi_H1')]:.4f} (fryst)")
print(f"  rho_H  = {theta_start[rho_H_idx]:.4f}  (Beta(5,3,[0.30,0.95]))")
print(f"  rho_C  = {theta_start[rho_C_idx]:.4f}  (Beta(5,3,[0.10,0.99]) — ny)")
print(f"  rho_O  = {theta_start[rho_O_idx]:.4f}  (Beta(5,3,[0.10,0.99]) — ny)")
print(f"  rho_Ys = {theta_start[rho_Ys_idx]:.4f}  (Beta(5,3,[0.10,0.99]) — ny)")
print(f"  rho_rp = {theta_start[rho_rp_idx]:.4f}  (Beta(5,3,[0.10,0.99]) — ny)")
print(f"  psi_R  = {theta_start[PARAM_NAMES.index('psi_R')]:.4f}  (Beta(2,2,[0.50,0.99]))")

print(f"\nStartverdier:")
for i, n in enumerate(PARAM_NAMES):
    print(f"  {n:12s}: {theta_start[i]:.4f} (std={post_std[i]:.4f})")

# ── B5-sjekk ved startverdi ───────────────────────────────────────────────────
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
save_pref = str(rot / "data/results/chain_kj30_prod")
print(f"\nLagrer til: {save_pref}*")
print("Starter kjøring 30 (200k produksjon + 15k burnin, v3, NZ=49, N=20) ...")
print()

chain, lp_vec, meta = adaptive_mcmc_with_monitoring(
    Y_pre=pre, Y_post=post, H=H, Sv=Sv,
    theta_init=theta_start, post_std_init=post_std,
    n_production=200_000, burnin=15_000, adapt_every=500,
    check_every=10_000, max_recalib=6,
    psrf_thr=1.10, ess_pct_thr=0.02,
    scale_init=0.70, seed=30, verbose=True,
    save_prefix=save_pref,
    build_fn=None,
    prior_overrides=prior_overrides,
    use_reparam=False,
)

print(f"\nKjøring 30 fullført.")
