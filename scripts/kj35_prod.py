"""
Kjøring 35 — Sandkasse: rho_s og gamma_p NB-kalibrert (Fase 0.75).

Motivasjon (sandkasse-diagnostikk, 2026-05-30):
  Parametersweep identifiserer rho_s (AR-glatting av RER) som dominerende parameter
  for RMSE(16pt NB)-reduksjon:
    rho_s=0.50: RMSE=0.136 (kj34: 0.200, −32%)
    rho_s=0.70: RMSE=0.122 (−39%)
    rho_s=0.70 + gamma_p=0.65: RMSE=0.121 (−40%)

  Analytisk begrunnelse:
  - rho_s kontrollerer hastighet på valutakursjustering i UIP (Justiniano & Preston 2010)
  - Empirisk: rho_s ≈ 0.40–0.60 for åpen økonomi
  - kj33/kj34 estimerte rho_s ≈ 0.056 (data-drevet, nær ren UIP)
  - NB-benchmark krever langsom RER-normalisering → rho_s ≈ 0.50

  gamma_p (Calvo-prisindeksasjon):
  - kj33/kj34 estimerte gamma_p ≈ 0.136 (meget lav)
  - Smets & Wouters (2007) euro: gamma_p ≈ 0.55
  - Bidrar til PI-persistens i q8–q12 (liten men reell effekt)

ENDRINGER fra kj34:
  1. rho_s: Beta(2,2,[0.05,0.90]) → Normal(0.50, 0.05, [0.30, 0.75]) — NB-kalibrert
  2. gamma_p: Beta(3,3,[0.0,0.95]) → Normal(0.65, 0.05, [0.40, 0.85]) — persistens
  (psi_R, phi_I1, phi_H1, rho_* priors uendret fra kj34)

FORVENTET UTFALL:
  - RMSE(16pt NB) ≈ 0.121–0.133 (analytisk verified)
  - B5 by4: sjekkes ved start — sannsynligvis OK siden Y-responsen ikke endres mye
  - PSRF < 1.10 (warm start fra kj34 / kj33 tail)

Warm start: kj34 → kj33 tail → kj31
Lagres til: data/results/chain_kj35_prod*
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
)
from nemo.model.equations import build_matrices_v3
from nemo.model.parameters import Parameters
import pandas as pd

rot = Path(__file__).parent.parent

datafil = rot / "data/processed/nemo_data_kpi_jae.csv"
if not datafil.exists():
    raise FileNotFoundError(f"{datafil} ikke funnet.")

from nemo.estimation.mcmc import OBS_NAMES
obs_df   = pd.read_csv(datafil, index_col=0, parse_dates=True)
obs_kols = ['pi_core_obs' if k == 'pi_obs' else k for k in OBS_NAMES]
pre  = obs_df[obs_df.index <= '2019-12-31'][obs_kols].values
post = obs_df[obs_df.index >= '2022-01-01'][obs_kols].values

print(f"Datafil: {datafil.name} (KPI-JAE)")
print(f"Pre={len(pre)} kv  Post={len(post)} kv")

H  = build_H()
Sv = build_Sv()

RHO_S_KJ35   = 0.50   # NB-kalibrert (Justiniano & Preston 2010: 0.40–0.60)
GAMMA_P_KJ35 = 0.65   # Smets & Wouters (2007) euroområdet ≈ 0.55
PHI_I1_KJ35  = 0.50
PHI_H1_KJ35  = 60.73
PSI_R_KJ35   = 0.88

prior_overrides = {
    # kj35: rho_s dogmatisk NB-kalibrert (sweepresultat: RMSE 0.200→0.136)
    'rho_s':   ('normal', RHO_S_KJ35, 0.05, 0.30, 0.75),
    # kj35: gamma_p dogmatisk for inflasjonspersistens (S&W 2007)
    'gamma_p': ('normal', GAMMA_P_KJ35, 0.05, 0.40, 0.85),
    # Identisk kj34:
    'phi_I1':  ('normal', PHI_I1_KJ35, 0.001, 0.40, 0.60),
    'phi_H1':  ('normal', PHI_H1_KJ35, 0.001, 60.70, 60.76),
    'psi_R':   ('normal', PSI_R_KJ35, 0.005, 0.84, 0.92),
    'rho_C':   ('beta', 5.0, 3.0, 0.10, 0.99),
    'rho_O':   ('beta', 5.0, 3.0, 0.10, 0.99),
    'rho_Ys':  ('beta', 5.0, 3.0, 0.10, 0.99),
    'rho_rp':  ('beta', 5.0, 3.0, 0.10, 0.99),
    'rho_A':   ('beta', 5.0, 3.0, 0.01, 0.99),
    'rho_H':   ('beta', 5.0, 2.0, 0.30, 0.99),
}

print(f"\nN_PARAMS = {N_PARAMS}")
theta_start = None
post_std    = None

# Warm start: kj34 → kj33 tail JSON → kj33 npy → kj31
for src, label in [
    (rot / "data/results/chain_kj34_prod_posterior.json", "kj34"),
    (rot / "data/results/chain_kj33_tail_posterior.json", "kj33 tail"),
    (rot / "data/results/chain_kj31_prod_posterior.json", "kj31"),
]:
    if not src.exists():
        continue
    print(f"Warm start fra {label}: {src.name}")
    with open(src) as f:
        d = json.load(f)
    key = 'posterior_mean' if 'posterior_mean' in d else 'summary'
    data = d[key]
    theta_start = np.zeros(N_PARAMS)
    post_std    = np.zeros(N_PARAMS)
    for i, n in enumerate(PARAM_NAMES):
        if n in data:
            v = data[n]
            if isinstance(v, dict):
                theta_start[i] = v['mean']
                post_std[i]    = max(v.get('std', 0.05), 0.001)
            else:
                theta_start[i] = float(v)
                post_std[i]    = 0.05
        else:
            theta_start[i] = KM.get(n, 0.5)
            post_std[i]    = 0.05
    break

# Fallback: npy
if theta_start is None:
    npy = rot / "data/results/chain_kj33_prod_partial.npy"
    if npy.exists():
        chain = np.load(npy)
        tail = chain[55000:74000]
        theta_start = tail.mean(0)
        post_std    = np.maximum(tail.std(0), 0.001)
        print(f"Warm start fra kj33 tail .npy")

if theta_start is None:
    theta_start = np.array([KM.get(n, 0.5) for n in PARAM_NAMES])
    post_std    = np.array([0.05]*N_PARAMS)

# kj35-spesifikke startverdier
rho_s_idx   = PARAM_NAMES.index('rho_s')
gamma_p_idx = PARAM_NAMES.index('gamma_p')
phi_I1_idx  = PARAM_NAMES.index('phi_I1')
phi_H1_idx  = PARAM_NAMES.index('phi_H1')
psi_R_idx   = PARAM_NAMES.index('psi_R')

theta_start[rho_s_idx]   = RHO_S_KJ35;   post_std[rho_s_idx]   = 0.05
theta_start[gamma_p_idx] = GAMMA_P_KJ35; post_std[gamma_p_idx] = 0.05
theta_start[phi_I1_idx]  = PHI_I1_KJ35;  post_std[phi_I1_idx]  = 0.001
theta_start[phi_H1_idx]  = PHI_H1_KJ35;  post_std[phi_H1_idx]  = 0.001
theta_start[psi_R_idx]   = np.clip(theta_start[psi_R_idx], 0.85, 0.91)
post_std[psi_R_idx]      = 0.005

# rho-klipp
rho_H_idx = PARAM_NAMES.index('rho_H')
theta_start[rho_H_idx] = np.clip(theta_start[rho_H_idx], 0.30, 0.99)
for n in ['rho_C', 'rho_O', 'rho_Ys', 'rho_rp']:
    idx = PARAM_NAMES.index(n)
    theta_start[idx] = np.clip(theta_start[idx], 0.10, 0.99)

phi_I2_idx = PARAM_NAMES.index('phi_I2')
post_std[phi_I2_idx] = min(post_std[phi_I2_idx], 30.0)

print(f"\nKjøring 35 — Sandkasse: rho_s og gamma_p NB-kalibrert")
print(f"  rho_s  = {theta_start[rho_s_idx]:.4f}  (Normal(0.50,0.05) — Justiniano & Preston 2010)")
print(f"  gamma_p= {theta_start[gamma_p_idx]:.4f}  (Normal(0.65,0.05) — Smets & Wouters 2007)")
print(f"  psi_R  = {theta_start[psi_R_idx]:.4f}  (Normal(0.88,0.005) — NB-kalibrert)")
print(f"  phi_I1 = {theta_start[phi_I1_idx]:.4f}  (frosset)")

print(f"\nStartverdier:")
for i, n in enumerate(PARAM_NAMES):
    print(f"  {n:12s}: {theta_start[i]:.4f} (std={post_std[i]:.4f})")

# B5-sjekk og analytisk RMSE-prediksjon
from nemo.model.equations import Y, PI, I_R, RER, E_i
from nemo.solver.blanchard_kahn import solve, compute_irf
import warnings

NB_FIGUR1 = {
    "Y":   {"q1": -0.20, "q4": -0.45, "q8": -0.35, "q12": -0.15},
    "PI":  {"q1": -0.05, "q4": -0.15, "q8": -0.20, "q12": -0.10},
    "I_R": {"q1": +1.00, "q4": +0.60, "q8": +0.20, "q12": +0.05},
    "RER": {"q1": -0.50, "q4": -0.40, "q8": -0.20, "q12": -0.05},
}
VI = {"Y": Y, "PI": PI, "I_R": I_R, "RER": RER}
HZ = {"q1": 0, "q4": 3, "q8": 7, "q12": 11}

p = Parameters()
for i, n in enumerate(PARAM_NAMES):
    if hasattr(p, n): setattr(p, n, float(theta_start[i]))
G0, G1, Psi, Pi = build_matrices_v3(p)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    T, R, d = solve(G0, G1, Psi, Pi, verbose=False)
irf_raw = compute_irf(T, R, E_i, 0.0025, T_periods=20)
peak = float(np.max(irf_raw[:, I_R]))
irf  = irf_raw / peak
by4  = irf[3, Y] / (-0.450)
bpi4 = irf[3, PI] / (-0.150)
sq   = [(irf[h, v] - NB_FIGUR1[vn][qn])**2
        for vn, v in VI.items() for qn, h in HZ.items()]
rmse_start = np.sqrt(np.mean(sq))

print(f"\nVed startverdi:")
print(f"  B5: by4={by4:.4f}  bpi4={bpi4:.4f}  BESTÅTT={0.8<=by4<=1.5 and bpi4>=0.35}")
print(f"  RMSE(16pt NB)={rmse_start:.4f}")
print(f"  I_R: q1={irf[0,I_R]:.3f}  q4={irf[3,I_R]:.3f}  q8={irf[7,I_R]:.3f}  (NB: 1.00, 0.60, 0.20)")
print(f"  PI:  q4={irf[3,PI]:.3f}  q8={irf[7,PI]:.3f}  (NB: -0.15, -0.20)")
print(f"  RER: q4={irf[3,RER]:.3f}  q12={irf[11,RER]:.3f}  (NB: -0.40, -0.05)")

lp0 = log_posterior(theta_start, H, Sv, pre, post,
                    build_fn=None, prior_overrides=prior_overrides)
print(f"\nStartverdi log-posterior: {lp0:.2f}")
if not np.isfinite(lp0):
    raise ValueError(f"Ikke-endelig lp={lp0}. Avbryter.")

save_pref = str(rot / "data/results/chain_kj35_prod")
print(f"\nLagrer til: {save_pref}*")
print("Starter kjøring 35 (200k produksjon + 30k burnin + 10 rekalibreringer) ...")
print()

chain, lp_vec, meta = adaptive_mcmc_with_monitoring(
    Y_pre=pre, Y_post=post, H=H, Sv=Sv,
    theta_init=theta_start, post_std_init=post_std,
    n_production=200_000, burnin=30_000, adapt_every=500,
    check_every=10_000, max_recalib=10,
    psrf_thr=1.10, ess_pct_thr=0.02,
    scale_init=0.70, seed=35, verbose=True,
    save_prefix=save_pref,
    build_fn=None,
    prior_overrides=prior_overrides,
    use_reparam=False,
)

print(f"\nKjøring 35 fullført.")
