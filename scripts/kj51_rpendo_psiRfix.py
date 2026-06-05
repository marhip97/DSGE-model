"""
kj51 — Fase 3: endogen risikopremie med psi_R pinnet (oppfølging av kj50)

PE-godkjent 2026-06-04 (oppfølging): kj50 viste at frigjøring av UIP-premien
presser psi_R → 0.99 (begrensning 7), som overpersisterer renten og forverrer
NB-fittet. kj51 pinner psi_R = 0.949 (kj41-verdi) via dogmatisk prior for å
ISOLERE premiens bidrag uten renteglatting-drift.

Mekanisme: prior_overrides={'psi_R': Normal(0.949, 0.0005)} — effektivt fast,
men holder infrastruktur (reparam/PARAM_NAMES/tester) uendret og reversibel.

Modell: build_matrices_rpendo (NZ=51), N_PARAMS=21. Warm start: kj50 posterior
(men psi_R settes til 0.949). Øvrig kalibrering = kj41/kj50.

Bruk:
  python scripts/kj51_rpendo_psiRfix.py [n_production] [n_burnin]
"""

from __future__ import annotations
import sys, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

from nemo.estimation.mcmc import (
    PARAM_NAMES, N_PARAMS, KM, PARAM_PRIORS,
    build_H_rpendo, build_Sv, OBS_NAMES,
    adaptive_mcmc_with_monitoring,
)
from nemo.model.equations import build_matrices_rpendo, I_R, Y, PI, RER, E_i
from nemo.model.parameters import Parameters
from nemo.solver.blanchard_kahn import solve, compute_irf

assert 'kappa_rp_endo' in PARAM_NAMES and 'rho_rp_endo' in PARAM_NAMES
assert 'psi_R' in PARAM_NAMES, "psi_R må være i PARAM_NAMES (pinnes via prior, ikke fjernes)"
assert N_PARAMS == 21, f"N_PARAMS={N_PARAMS}, forventet 21"

# ── psi_R pinnet via dogmatisk prior (= kj41-verdi) ───────────────────────────
PSI_R_PIN = 0.949
PRIOR_OVERRIDES = {'psi_R': ('normal', PSI_R_PIN, 0.0005, 0.90, 0.99)}

LAMBDA_PI4_FIXED = 0.0
PHI_PQ_FIXED     = 150.0
def _build_fn(p, theta_H: float = 0.05):
    p.phi_PQ = PHI_PQ_FIXED
    return build_matrices_rpendo(p, theta_H=theta_H, lambda_pi4=LAMBDA_PI4_FIXED)

H  = build_H_rpendo()
Sv = build_Sv()
obs_df = pd.read_csv(ROOT / "data/processed/nemo_data_kpi_jae.csv", index_col=0, parse_dates=True)
obs_kols = ['pi_core_obs' if k == 'pi_obs' else k for k in OBS_NAMES]
Y_pre  = obs_df[obs_df.index <= '2019-12-31'][obs_kols].values
Y_post = obs_df[obs_df.index >= '2022-01-01'][obs_kols].values
print(f"Data: Y_pre={Y_pre.shape}, Y_post={Y_post.shape}")

# ── Warm start fra kj50 (men psi_R pinnet) ────────────────────────────────────
with open(ROOT / "data/results/chain_kj50_prod_posterior.json") as f:
    kj50 = json.load(f)["summary"]
theta_start   = np.array([KM.get(n, 0.5) for n in PARAM_NAMES])
post_std_init = np.ones(N_PARAMS) * 0.05
for i, name in enumerate(PARAM_NAMES):
    if name in kj50:
        theta_start[i]   = float(kj50[name]["mean"])
        post_std_init[i] = max(float(kj50[name]["std"]), 1e-4)
theta_start[PARAM_NAMES.index('psi_R')]   = PSI_R_PIN
post_std_init[PARAM_NAMES.index('psi_R')] = 0.0005
for i, name in enumerate(PARAM_NAMES):
    lb, ub = float(PARAM_PRIORS[name][-2]), float(PARAM_PRIORS[name][-1])
    theta_start[i] = float(np.clip(theta_start[i], lb + 1e-6, ub - 1e-6))

print(f"\nWarm start (kj50 posterior, psi_R pinnet={PSI_R_PIN}):")
for i, n in enumerate(PARAM_NAMES):
    print(f"  {n:14s} = {theta_start[i]:.4f}  [std={post_std_init[i]:.4f}]")

N_PROD   = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
N_BURNIN = int(sys.argv[2]) if len(sys.argv) > 2 else 30_000
SEED     = 51
SAVE_PREFIX = str(ROOT / "data/results/chain_kj51_prod")

print(f"\n=== kj51 MCMC START (n_prod={N_PROD}, burnin={N_BURNIN}, psi_R pin={PSI_R_PIN}) ===")
t0 = time.time()
chain, lp_chain, meta = adaptive_mcmc_with_monitoring(
    Y_pre, Y_post, H, Sv,
    theta_init=theta_start, post_std_init=post_std_init,
    n_production=N_PROD, burnin=N_BURNIN,
    adapt_every=500, check_every=10_000, max_recalib=5,
    psrf_thr=1.10, ess_pct_thr=0.02, scale_init=0.676,
    seed=SEED, verbose=True, save_prefix=SAVE_PREFIX,
    use_reparam=True, block_indices=None,
    build_fn=_build_fn, prior_overrides=PRIOR_OVERRIDES,
)
print(f"\nMCMC fullført på {(time.time()-t0)/60:.1f} min")

# ── Evaluering ─────────────────────────────────────────────────────────────────
try:
    with open(str(ROOT / "data/results/chain_kj51_prod_posterior.json")) as f:
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
print(f"\n=== kj51 ENDELIG EVALUERING (psi_R pinnet={PSI_R_PIN}) ===")
print(f"RMSE(16pt NB) = {rmse:.4f}  (kj41: 0.295, kj50: 0.374)")
for k in ["I_R","RER","Y","PI"]:
    print(f"  {k:4}: {[f'{v:+.3f}' for v in mod[k]]}  NB {nb[k]}")
print(f"  kappa_rp_endo = {summary.get('kappa_rp_endo',{}).get('mean',float('nan')):.4f}"
      f" ± {summary.get('kappa_rp_endo',{}).get('std',float('nan')):.4f}")
print(f"  rho_rp_endo   = {summary.get('rho_rp_endo',{}).get('mean',float('nan')):.4f}")
print(f"  psi_R         = {summary.get('psi_R',{}).get('mean',float('nan')):.4f} (pinnet {PSI_R_PIN})")
print(f"  BK-stabil: {d.get('stable')}, max|eig|={d.get('max_eig_T',float('nan')):.6f}")
print("kj51 fullført.")
