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

from nemo.analysis.analyse import rts_smoother

datafil  = ROOT / "data/processed/nemo_data_kpi_jae.csv"
obs_df   = pd.read_csv(datafil, index_col=0, parse_dates=True)
obs_kols = ['pi_core_obs' if k == 'pi_obs' else k for k in OBS_NAMES]

Hmat = build_H()
Sv   = build_Sv()

Q = R @ np.diag(sigma_vec**2) @ R.T

def kalman_filter_states(Y_obs: np.ndarray) -> np.ndarray:
    """Kalman-filter (kun foroverpass) — returner filtrerte tilstander (T_obs × NZ)."""
    from scipy.linalg import solve_discrete_lyapunov
    T_obs = len(Y_obs)
    z_filt = np.zeros((T_obs, NZ))
    try:
        P = solve_discrete_lyapunov(T, Q)
    except Exception:
        P = np.eye(NZ) * 1.0
    z = np.zeros(NZ)
    for t in range(T_obs):
        z_pr = T @ z; P_pr = (T @ P @ T.T + Q); P_pr = (P_pr + P_pr.T)/2
        obs = Y_obs[t]; valid = np.where(np.isfinite(obs))[0]
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

# Kombiner pre+post (med NaN-hull for COVID-perioden) for full RTS-smoother
dates_all = obs_df.index
obs_all   = obs_df[obs_kols].values
# Merk: COVID-perioden (2020Q1–2021Q4) er NaN i data — smoother håndterer dette
Q_mat = np.diag(sigma_vec**2)

print("  Kjører RTS-smoother (full periode inkl. COVID-hull)...")
z_all = rts_smoother(T, R, Q_mat, obs_all, Hmat, Sv)
print(f"  RTS-smoother ferdig: {len(z_all)} perioder.")

# Split tilbake til pre/post
pre_mask  = obs_df.index <= '2019-12-31'
post_mask = obs_df.index >= '2022-01-01'
z_pre  = z_all[pre_mask]
z_post = z_all[post_mask]

dates_pre  = obs_df[pre_mask].index
dates_post = obs_df[post_mask].index

dates_pre  = obs_df[obs_df.index <= '2019-12-31'].index
dates_post = obs_df[obs_df.index >= '2022-01-01'].index

def shock_contributions(z_smooth: np.ndarray, h_rows: dict,
                        scales: dict) -> dict:
    """Beregn periode-for-periode sjokk-innovasjonsbidrag (for sum/cumulering)."""
    T_obs = len(z_smooth)
    contrib: dict[str, dict[str, list]] = {vn: {SHOCK_NAMES[e]: [] for e in SHOCK_IDX}
                                            for vn in VAR_NAMES}
    for t in range(1, T_obs):
        eps_t = z_smooth[t] - T @ z_smooth[t-1]
        for vn in VAR_NAMES:
            h_row = h_rows[vn]
            sc    = scales[vn]
            for eidx in SHOCK_IDX:
                r_col = R[:, eidx]
                contrib[vn][SHOCK_NAMES[eidx]].append(
                    sc * float(h_row @ (r_col * eps_t[eidx]))
                )
    return contrib


def level_hd(z_smooth: np.ndarray, h_rows: dict, scales: dict,
             R_pinv: np.ndarray) -> dict:
    """Full nivå-HD via rekursiv propagasjon gjennom T-matrisen.

    Sjokk-amplitude hentes via pseudo-invers: ε_t = R⁺ @ (z_t - T @ z_{t-1}),
    slik at sum_j R[:,j]*ε_j(t) = proj_{col(R)}(residual) ≈ residual.
    Dette gir nøyaktig additivitet: Σ_j bidrag_j + initial ≈ H @ z_t.
    """
    T_obs  = len(z_smooth)
    result: dict[str, dict[str, list]] = {
        vn: {**{SHOCK_NAMES[e]: [] for e in SHOCK_IDX}, "initial": []}
        for vn in VAR_NAMES
    }

    c_state  = {eidx: np.zeros(NZ) for eidx in SHOCK_IDX}
    z0_state = z_smooth[0].copy()

    for t in range(1, T_obs):
        residual   = z_smooth[t] - T @ z_smooth[t-1]
        eps_shocks = R_pinv @ residual          # (NE,) sjokk-amplituder
        z0_state   = T @ z0_state

        # Oppdater tilstandsbidrag for alle sjokk (EN gang per tidssteg)
        for eidx in SHOCK_IDX:
            c_state[eidx] = T @ c_state[eidx] + R[:, eidx] * eps_shocks[eidx]

        for vn in VAR_NAMES:
            h_row = h_rows[vn]
            sc    = scales[vn]
            result[vn]["initial"].append(sc * float(h_row @ z0_state))
            for eidx in SHOCK_IDX:
                result[vn][SHOCK_NAMES[eidx]].append(
                    sc * float(h_row @ c_state[eidx])
                )
    return result

# H-rader fra observasjonsmatrisen (obs-rom → korrekte enheter)
# OBS_NAMES-rekkefølge: dy_obs(0), dc_obs(1), ..., pi_obs(5), ..., i_R_obs(7), ..., ds_obs(9)
_obs_map = {
    'Y':   OBS_NAMES.index('dy_obs'),
    'PI':  [k for k, n in enumerate(OBS_NAMES) if 'pi' in n][0],
    'I_R': OBS_NAMES.index('i_R_obs'),
    'RER': OBS_NAMES.index('ds_obs'),
}
h_rows   = {vn: Hmat[oidx] for vn, oidx in _obs_map.items()}
# Skalering til annualiserte enheter for I_R (kvartal × 4)
_scales  = {'Y': 1.0, 'PI': 1.0, 'I_R': 4.0, 'RER': 1.0}

contrib_pre  = shock_contributions(z_pre,  h_rows, _scales)
contrib_post = shock_contributions(z_post, h_rows, _scales)

# Pseudo-invers av R for nøyaktig sjokk-attributering
R_pinv = np.linalg.pinv(R)  # (NE × NZ)

# Nivå-HD via rekursiv propagasjon
print("  Beregner nivå-HD (rekursiv propagasjon gjennom T)...")
lvl_pre  = level_hd(z_pre,  h_rows, _scales, R_pinv)
lvl_post = level_hd(z_post, h_rows, _scales, R_pinv)

hd = {
    # Periode-for-periode innovasjonsbidrag (brukes til å sjekke additivitet)
    "innov_pre":  {vn: {s: v for s, v in contrib_pre[vn].items()}  for vn in VAR_NAMES},
    "innov_post": {vn: {s: v for s, v in contrib_post[vn].items()} for vn in VAR_NAMES},
    # Nivå-bidrag akkumulert via T-propagasjon (dette er HD-plottet)
    "level_pre":  {vn: dict(lvl_pre[vn])  for vn in VAR_NAMES},
    "level_post": {vn: dict(lvl_post[vn]) for vn in VAR_NAMES},
    "dates_pre":  [str(d.date()) for d in dates_pre[1:]],
    "dates_post": [str(d.date()) for d in dates_post[1:]],
}

json.dump(hd, open(RESULTS / "kj41_hd.json", "w"), indent=2)
print(f"HD lagret: {RESULTS/'kj41_hd.json'}")

# Sanity-sjekk: sjokk-sum + initial ≈ H @ z_smooth (observert nivå)
print("\n  Sanity-sjekk nivå-HD (post-COVID I_R, siste kvartal):")
last_t = -1
total_contrib = sum(
    lvl_post["I_R"].get(SHOCK_NAMES[e], [0])[last_t] for e in SHOCK_IDX
) + lvl_post["I_R"]["initial"][last_t]
actual = _scales["I_R"] * float(h_rows["I_R"] @ z_post[last_t])
print(f"    Sum sjokk+initial: {total_contrib:.4f}  Faktisk H@z: {actual:.4f}  "
      f"Diff: {abs(total_contrib-actual):.2e}")

print("\n  Post-COVID I_R nivåbidrag siste kvartal (annualisert %):")
for s, vals in sorted(lvl_post["I_R"].items(),
                      key=lambda x: abs(x[1][last_t]), reverse=True):
    print(f"    {s:22s}  {vals[last_t]:+.4f}%")

print("\nkj41_fevd_hd fullført.")
