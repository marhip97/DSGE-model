"""
[STAT] kj41 FEVD + Historisk dekomposisjon

Bruker kj41-posterioret (psi_R=0.9490, beste kalibrering etter kj41–43)
for å beregne:
  1. FEVD ved q4, q8, q20 for Y, PI, I_R, RER
  2. Historisk dekomposisjon (sjokk-bidrag per kvartal)

Resultater lagres i data/results/kj41_fevd.json og kj41_hd.json.
"""

from __future__ import annotations
import json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nemo.estimation.mcmc import (
    PARAM_NAMES, SIGMA_A_FIXED, SIGMA_RP_FIXED,
    build_H, build_Sv, OBS_NAMES,
)
from nemo.model.equations import (
    build_matrices_v3_forward,
    Y, PI, I_R, RER, NZ, NE,
    E_A, E_C, E_P, E_O, E_Ys, E_rp, E_i, E_H,
)
from nemo.model.parameters import Parameters
from nemo.solver.blanchard_kahn import solve, compute_irf

RESULTS = ROOT / "data" / "results"

# ── Sjokkindekser og navn ─────────────────────────────────────────────────────
SHOCK_IDX = [E_A, E_C, E_P, E_O, E_Ys, E_rp, E_i, E_H]
SHOCK_NAMES = {
    E_A: "TFP", E_C: "Konsum", E_P: "Prismarkup",
    E_O: "Oljepris", E_Ys: "Utenl.ettersp.",
    E_rp: "Risikopremie", E_i: "Pengepolitikk", E_H: "Bolig",
}
VAR_NAMES = {"Y": Y, "PI": PI, "I_R": I_R, "RER": RER}

Parameters.phi_PQ = 150.0
LAMBDA_PI4 = 0.0

# ── Last posterior ─────────────────────────────────────────────────────────────
post = json.load(open(RESULTS / "chain_kj41_prod_posterior.json"))["summary"]
theta = np.array([post[n]["mean"] for n in PARAM_NAMES])

p = Parameters()
for i, n in enumerate(PARAM_NAMES):
    if hasattr(p, n):
        setattr(p, n, float(theta[i]))
p.lambda_pi4 = LAMBDA_PI4

# ── Løs BK ────────────────────────────────────────────────────────────────────
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    G0, G1, Psi, Pi_ = build_matrices_v3_forward(p, lambda_pi4=LAMBDA_PI4)
    T, R, diag = solve(G0, G1, Psi, Pi_, verbose=False)

assert diag.get("stable", False), "BK-stabilitet ikke oppfylt!"
print(f"BK stabil ✅  max|eig|={diag.get('max_eig', float('nan')):.4f}")

# ── Sigma-vektor ──────────────────────────────────────────────────────────────
def _sigma(name: str) -> float:
    if name == "sigma_A":  return SIGMA_A_FIXED
    if name == "sigma_rp": return SIGMA_RP_FIXED
    return float(post[name]["mean"])

sigma_vec = np.zeros(NE)
smap = {E_A:"sigma_A", E_C:"sigma_C", E_P:"sigma_P", E_O:"sigma_O",
        E_Ys:"sigma_Ys", E_rp:"sigma_rp", E_i:"sigma_i", E_H:"sigma_H"}
for eidx, sname in smap.items():
    sigma_vec[eidx] = _sigma(sname)

print("\nSjokk-standardavvik (posterior mean):")
for eidx in SHOCK_IDX:
    print(f"  {SHOCK_NAMES[eidx]:20s} σ={sigma_vec[eidx]:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. FEVD
# ─────────────────────────────────────────────────────────────────────────────
HORIZONS = [4, 8, 20]
H_PERIODS = max(HORIZONS) + 1

# Beregn kumulativ bidragsvariance per sjokk per variabel
cumvar: dict[str, dict[str, np.ndarray]] = {vn: {} for vn in VAR_NAMES}

for eidx in SHOCK_IDX:
    irf = compute_irf(T, R, eidx, sigma_vec[eidx], T_periods=H_PERIODS)
    for vn, vidx in VAR_NAMES.items():
        cumvar[vn][SHOCK_NAMES[eidx]] = np.cumsum(irf[:, vidx]**2)

fevd: dict = {}
for vn in VAR_NAMES:
    fevd[vn] = {}
    for h in HORIZONS:
        tot = sum(cumvar[vn][s][h] for s in cumvar[vn])
        fevd[vn][f"q{h}"] = {
            s: round(float(cumvar[vn][s][h] / max(tot, 1e-15) * 100), 1)
            for s in cumvar[vn]
        }

print("\n── FEVD (% av prognosevariansfeil) ──────────────────────────────────")
for vn in VAR_NAMES:
    print(f"\n  {vn}:")
    shocks_sorted = sorted(cumvar[vn].keys(),
                           key=lambda s: cumvar[vn][s][20], reverse=True)
    header = f"  {'Sjokk':22s}" + "".join(f"  q{h:2d}" for h in HORIZONS)
    print(header)
    for s in shocks_sorted:
        row = f"  {s:22s}"
        for h in HORIZONS:
            row += f"  {fevd[vn][f'q{h}'][s]:5.1f}"
        print(row)

json.dump(fevd, open(RESULTS / "kj41_fevd.json", "w"), indent=2)
print(f"\nFEVD lagret: {RESULTS/'kj41_fevd.json'}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Historisk dekomposisjon
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Historisk dekomposisjon ──────────────────────────────────────────")

datafil  = ROOT / "data/processed/nemo_data_kpi_jae.csv"
obs_df   = pd.read_csv(datafil, index_col=0, parse_dates=True)
obs_kols = ['pi_core_obs' if k == 'pi_obs' else k for k in OBS_NAMES]

pre  = obs_df[obs_df.index <= '2019-12-31'][obs_kols].values
post_data = obs_df[obs_df.index >= '2022-01-01'][obs_kols].values

Hmat = build_H()
Sv   = build_Sv()

Q = R @ np.diag(sigma_vec**2) @ R.T

def kalman_filter_states(Y_obs: np.ndarray) -> np.ndarray:
    """Kalman-filter (kun foroverpass) — returner filtrerte tilstander (T_obs × NZ).

    Initialiseres med stasjonær kovarians fra Lyapunov-ligningen for numerisk stabilitet.
    """
    from scipy.linalg import solve_discrete_lyapunov
    T_obs = len(Y_obs)
    z_filt = np.zeros((T_obs, NZ))

    try:
        P = solve_discrete_lyapunov(T, Q)
    except Exception:
        P = np.eye(NZ) * 1.0

    z = np.zeros(NZ)

    for t in range(T_obs):
        z_pr = T @ z
        P_pr = T @ P @ T.T + Q
        P_pr = (P_pr + P_pr.T) / 2

        obs = Y_obs[t]
        valid = np.where(np.isfinite(obs))[0]
        if len(valid) > 0:
            H_v   = Hmat[valid, :]
            Sv_v  = Sv[np.ix_(valid, valid)]
            S_v   = H_v @ P_pr @ H_v.T + Sv_v
            S_v   = (S_v + S_v.T) / 2
            K     = np.linalg.solve(S_v.T, (P_pr @ H_v.T).T).T
            innov = obs[valid] - H_v @ z_pr
            z     = z_pr + K @ innov
            P     = (np.eye(NZ) - K @ H_v) @ P_pr
            P     = (P + P.T) / 2
        else:
            z, P = z_pr, P_pr

        z_filt[t] = z

    return z_filt

# Pre-COVID
z_pre  = kalman_filter_states(pre)
# Post-COVID
z_post = kalman_filter_states(post_data)

dates_pre  = obs_df[obs_df.index <= '2019-12-31'].index
dates_post = obs_df[obs_df.index >= '2022-01-01'].index

def shock_contributions(z_smooth: np.ndarray) -> dict:
    """Beregn sjokk-bidrag til observerbare via glattede tilstander.

    Strukturelle sjokk gjenopprettes ved minste-norm inversjon:
        innov_t = R @ eps_t  →  eps_hat_t = R^+ @ innov_t
    Bidrag av sjokk k til variabel j: R[j,k] * eps_hat_t[k]
    """
    T_obs  = len(z_smooth)
    R_sub  = R[:, SHOCK_IDX]            # NZ × n_shock
    R_pinv = np.linalg.pinv(R_sub)      # n_shock × NZ  (pseudo-invers)

    contrib: dict[str, dict[str, list]] = {vn: {SHOCK_NAMES[e]: [] for e in SHOCK_IDX}
                                            for vn in VAR_NAMES}
    for t in range(1, T_obs):
        innov    = z_smooth[t] - T @ z_smooth[t-1]   # NZ
        eps_hat  = R_pinv @ innov                      # n_shock (strukturelle sjokk)
        for vn, vidx in VAR_NAMES.items():
            for k, eidx in enumerate(SHOCK_IDX):
                contrib[vn][SHOCK_NAMES[eidx]].append(
                    float(R_sub[vidx, k] * eps_hat[k])
                )
    return contrib

contrib_pre  = shock_contributions(z_pre)
contrib_post = shock_contributions(z_post)

hd = {
    "pre":  {vn: {s: v for s, v in contrib_pre[vn].items()}  for vn in VAR_NAMES},
    "post": {vn: {s: v for s, v in contrib_post[vn].items()} for vn in VAR_NAMES},
    "dates_pre":  [str(d.date()) for d in dates_pre[1:]],
    "dates_post": [str(d.date()) for d in dates_post[1:]],
}

json.dump(hd, open(RESULTS / "kj41_hd.json", "w"), indent=2)
print(f"HD lagret (foreløpig — krever full RTS-smoother for endelig versjon): {RESULTS/'kj41_hd.json'}")
print("\nNB: Kalman-filter HD er foreløpig. Historisk dekomposisjon med full RTS-smoother")
print("    implementeres i kj41_analyse_v2.py.")

print("\nkj41_fevd_hd fullført.")
