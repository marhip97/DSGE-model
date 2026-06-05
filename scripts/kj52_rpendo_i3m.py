"""
kj52 — Fase 3: endogen risikopremie + i_3m anker pengemarkedspremien

PE-godkjent 2026-06-04 (oppfølging av kj50/kj51). kj51 avdekket
observasjonsekvivalens: persistensen migrerer mellom psi_R, endogen FX-premie og
eksogen risikopremie fordi de ikke er separat identifiserbare med 14 observabler.

kj52 utnytter en EKSISTERENDE serie bedre: i_3m_obs (NIBOR 3M) re-mappes fra
redundant I_R-observasjon til i_R + pengemarkedspremie (EPS_PREM). EPS_PREM inngår
både i innskuddsrenten og UIP — å anke den fjerner en fri RER-persistens-absorber.

Design: psi_R FRI (som kj50), build_matrices_rpendo, build_H_rpendo_i3m.
Tester om den ekstra observasjonsinformasjonen (a) hindrer psi_R-driften til 0.99
og (b) skarpere identifiserer premiekanalene. Warm start: kj50 posterior.
N_PARAMS=21.

Bruk:
  python scripts/kj52_rpendo_i3m.py [n_production] [n_burnin]
"""

from __future__ import annotations
import sys, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

from nemo.estimation.mcmc import (
    PARAM_NAMES, N_PARAMS, KM, PARAM_PRIORS,
    build_H_rpendo_i3m, build_Sv, OBS_NAMES,
    adaptive_mcmc_with_monitoring,
)
from nemo.model.equations import build_matrices_rpendo, I_R, Y, PI, RER, EPS_PREM, E_i
from nemo.model.parameters import Parameters
from nemo.solver.blanchard_kahn import solve, compute_irf

assert 'kappa_rp_endo' in PARAM_NAMES and 'rho_rp_endo' in PARAM_NAMES
assert 'psi_R' in PARAM_NAMES, "psi_R skal være FRI i kj52"
assert 'sigma_prem' in PARAM_NAMES, "sigma_prem skal estimeres i kj52"
assert N_PARAMS == 22, f"N_PARAMS={N_PARAMS}, forventet 22"

LAMBDA_PI4_FIXED = 0.0
PHI_PQ_FIXED     = 150.0
def _build_fn(p, theta_H: float = 0.05):
    p.phi_PQ = PHI_PQ_FIXED
    return build_matrices_rpendo(p, theta_H=theta_H, lambda_pi4=LAMBDA_PI4_FIXED)

# ── Re-mappet observasjonslikning: i_3m = i_R + pengemarkedspremie ────────────
H  = build_H_rpendo_i3m()
Sv = build_Sv()
assert H[8, EPS_PREM] == 4.0, "i_3m_obs skal anke EPS_PREM"
print(f"i_3m-mapping: H[8,I_R]={H[8,I_R]}, H[8,EPS_PREM]={H[8,EPS_PREM]} (re-mappet)")

obs_df = pd.read_csv(ROOT / "data/processed/nemo_data_kpi_jae.csv", index_col=0, parse_dates=True)
obs_kols = ['pi_core_obs' if k == 'pi_obs' else k for k in OBS_NAMES]
Y_pre  = obs_df[obs_df.index <= '2019-12-31'][obs_kols].values
Y_post = obs_df[obs_df.index >= '2022-01-01'][obs_kols].values
print(f"Data: Y_pre={Y_pre.shape}, Y_post={Y_post.shape}")

# ── Warm start fra kj50 ────────────────────────────────────────────────────────
with open(ROOT / "data/results/chain_kj50_prod_posterior.json") as f:
    kj50 = json.load(f)["summary"]
theta_start   = np.array([KM.get(n, 0.5) for n in PARAM_NAMES])
post_std_init = np.ones(N_PARAMS) * 0.05
for i, name in enumerate(PARAM_NAMES):
    if name in kj50:
        theta_start[i]   = float(kj50[name]["mean"])
        post_std_init[i] = max(float(kj50[name]["std"]), 1e-4)
# sigma_prem finnes ikke i kj50 — start på prior-skala
theta_start[PARAM_NAMES.index('sigma_prem')]   = 0.0010
post_std_init[PARAM_NAMES.index('sigma_prem')] = 0.0005
for i, name in enumerate(PARAM_NAMES):
    lb, ub = float(PARAM_PRIORS[name][-2]), float(PARAM_PRIORS[name][-1])
    theta_start[i] = float(np.clip(theta_start[i], lb + 1e-6, ub - 1e-6))

print("\nWarm start (kj50 posterior, psi_R fri):")
for i, n in enumerate(PARAM_NAMES):
    print(f"  {n:14s} = {theta_start[i]:.4f}  [std={post_std_init[i]:.4f}]")

N_PROD   = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
N_BURNIN = int(sys.argv[2]) if len(sys.argv) > 2 else 30_000
SEED     = 52
SAVE_PREFIX = str(ROOT / "data/results/chain_kj52_prod")

print(f"\n=== kj52 MCMC START (n_prod={N_PROD}, burnin={N_BURNIN}, i_3m→premie) ===")
t0 = time.time()
chain, lp_chain, meta = adaptive_mcmc_with_monitoring(
    Y_pre, Y_post, H, Sv,
    theta_init=theta_start, post_std_init=post_std_init,
    n_production=N_PROD, burnin=N_BURNIN,
    adapt_every=500, check_every=10_000, max_recalib=5,
    psrf_thr=1.10, ess_pct_thr=0.02, scale_init=0.676,
    seed=SEED, verbose=True, save_prefix=SAVE_PREFIX,
    use_reparam=True, block_indices=None,
    build_fn=_build_fn, prior_overrides=None,
)
print(f"\nMCMC fullført på {(time.time()-t0)/60:.1f} min")

# ── Evaluering ─────────────────────────────────────────────────────────────────
try:
    with open(str(ROOT / "data/results/chain_kj52_prod_posterior.json")) as f:
        summary = json.load(f)["summary"]
except FileNotFoundError:
    print("Posterior JSON ikke funnet — kjøring avbrutt tidlig?"); raise SystemExit(1)

p_post = Parameters(); p_post.phi_PQ = PHI_PQ_FIXED
for name in PARAM_NAMES:
    if name in summary: setattr(p_post, name, float(summary[name]["mean"]))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    G0, G1, Psi, Pi = _build_fn(p_post); T, R, d = solve(G0, G1, Psi, Pi, verbose=False)
irf = compute_irf(T, R, E_i, 0.0025, T_periods=13); irf = irf / np.max(irf[:, I_R])
nb = {"Y":[-0.12,-0.47,-0.40,-0.25],"PI":[-0.03,-0.14,-0.22,-0.22],
      "I_R":[1.00,0.55,0.10,-0.15],"RER":[-1.50,-1.00,-0.50,-0.20]}
VAR={"Y":Y,"PI":PI,"I_R":I_R,"RER":RER}
def _q(v): return [float(irf[h, VAR[v]]) for h in [0,3,7,11]]
mod={k:_q(k) for k in nb}; pts=[(m,n) for k in nb for m,n in zip(mod[k],nb[k])]
rmse=float(np.sqrt(np.mean([(m-n)**2 for m,n in pts])))
print(f"\n=== kj52 ENDELIG EVALUERING (i_3m anker pengemarkedspremie) ===")
print(f"RMSE(16pt NB) = {rmse:.4f}  (kj41: 0.295, kj50: 0.374, kj51: 0.281)")
for k in ["I_R","RER","Y","PI"]:
    print(f"  {k:4}: {[f'{v:+.3f}' for v in mod[k]]}  NB {nb[k]}")
for pp in ['kappa_rp_endo','rho_rp_endo','psi_R','rho_rp','rho_prem']:
    s=summary.get(pp,{})
    print(f"  {pp:14}= {s.get('mean',float('nan')):.4f} ± {s.get('std',float('nan')):.4f}"
          f"  (ESS={s.get('ess',float('nan')):.0f}, PSRF={s.get('psrf',float('nan')):.3f})")
print(f"  BK-stabil: {d.get('stable')}, max|eig|={d.get('max_eig_T',float('nan')):.6f}")
print("kj52 fullført.")
