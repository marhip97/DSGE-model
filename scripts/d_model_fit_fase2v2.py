"""
[ARK] D-rapporten for Fase 2v2 (NZ=49, phi_u estimert).

Leser chain_fase2v2_prod_posterior.json og produserer:
  - Parameter-tabell med mean/std/p05/p95/ESS/PSRF
  - Sammenligning med Fase 2 (NZ=48) posterior
  - Kalman-filter R² per observert variabel
  - Lagrer docs/D_modellfit_fase2v2.md
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROT / "src"))

from nemo.estimation.mcmc import PARAM_NAMES, N_PARAMS, build_H, build_Sv, KM

POST_PATH = ROT / "data" / "results" / "chain_fase2v2_prod_posterior.json"
PREV_PATH = ROT / "data" / "results" / "chain_fase2_prod_posterior.json"

if not POST_PATH.exists():
    print(f"FEIL: {POST_PATH} finnes ikke. Kjør fase2v2_production.py først.")
    sys.exit(1)

with open(POST_PATH) as f:
    post = json.load(f)

summ = post['summary']
meta = post['meta']

print(f"Fase 2v2 posterior lest: {meta['n_samples']} trekk, {meta['time_min']:.1f} min")
print(f"  PSRF_max = {meta['psrf_max']:.4f}")
print(f"  ESS_min  = {meta['ess_min']:.0f} ({meta['ess_min']/meta['n_samples']*100:.2f}%)")

# Last Fase 2 posterior for sammenligning
prev_summ = {}
if PREV_PATH.exists():
    with open(PREV_PATH) as f:
        prev = json.load(f)
    prev_summ = prev['summary']

# ── Parameter-tabell ────────────────────────────────────────────────────────
rows = []
for n in PARAM_NAMES:
    s = summ[n]
    p = prev_summ.get(n, {})
    km_val = KM.get(n, float('nan'))
    rows.append({
        'Param':   n,
        'K&M':     f"{km_val:.3f}" if not np.isnan(km_val) else '—',
        'Fase2 mean': f"{p.get('mean', float('nan')):.4f}" if p else '—',
        'Fase2v2 mean': f"{s['mean']:.4f}",
        'std':     f"{s['std']:.4f}",
        'p05':     f"{s['p05']:.4f}",
        'p95':     f"{s['p95']:.4f}",
        'ESS':     f"{s['ess']:.0f}",
        'PSRF':    f"{s['psrf']:.3f}",
    })
df_par = pd.DataFrame(rows)

# ── Kalman R² ───────────────────────────────────────────────────────────────
data_sti = ROT / "data" / "processed" / "nemo_data_faktisk_v2.csv"
df = pd.read_csv(data_sti, index_col=0)
df.index = pd.to_datetime(df.index)
pre  = df[df.index <= "2019-12-31"].values
post_data = df[df.index >= "2022-01-01"].values
obs_names = list(df.columns)

H  = build_H()
Sv = build_Sv()

theta_mean = np.array([summ[n]['mean'] for n in PARAM_NAMES])

from nemo.model.equations import build_matrices_v3
from nemo.model.parameters import Parameters as NemoParams
import importlib, nemo.model.parameters as _pm

p = NemoParams()
for i, n in enumerate(PARAM_NAMES):
    if hasattr(p, n):
        setattr(p, n, theta_mean[i])

try:
    mats = build_matrices_v3(p)
    T, R_mat = mats['T'], mats['R']

    def kalman_r2(Y):
        nobs, nvar = Y.shape
        T_obs = nobs
        y_hat = np.zeros_like(Y)
        try:
            from nemo.analysis.analyse import kalman_filter as kf
            ll, states = kf(T, R_mat, H, Sv, Y, return_states=True)
            y_hat = (H @ states.T).T
        except Exception:
            pass
        ss_res = np.sum((Y - y_hat)**2, axis=0)
        ss_tot = np.sum((Y - Y.mean(axis=0))**2, axis=0)
        r2 = 1 - ss_res / np.where(ss_tot > 0, ss_tot, np.nan)
        return r2

    r2_pre  = kalman_r2(pre)
    r2_post = kalman_r2(post_data)

    r2_rows = [{'Variabel': obs_names[j],
                'R² pre (≤2019)': f"{r2_pre[j]:.3f}" if j < len(r2_pre) else '—',
                'R² post (≥2022)': f"{r2_post[j]:.3f}" if j < len(r2_post) else '—'}
               for j in range(len(obs_names))]
    df_r2 = pd.DataFrame(r2_rows)
    has_r2 = True
except Exception as e:
    print(f"  NB: Kalman R² feilet: {e}")
    df_r2 = pd.DataFrame()
    has_r2 = False

# ── Skriv rapport ────────────────────────────────────────────────────────────
lines = [
    "# D — Modellpassform Fase 2v2 (NZ=49, phi_u estimert)",
    "",
    f"**Dato:** 2026-05-16  ",
    f"**Trekk:** {meta['n_samples']:,}  ",
    f"**Tid:** {meta['time_min']:.1f} min  ",
    f"**PSRF_max:** {meta['psrf_max']:.4f}  ",
    f"**ESS_min:** {meta['ess_min']:.0f} ({meta['ess_min']/meta['n_samples']*100:.2f}% av trekk)  ",
    "",
    "## Parametertabell",
    "",
    df_par.to_markdown(index=False),
    "",
]

if has_r2:
    lines += [
        "## Kalman R² per observert variabel",
        "",
        df_r2.to_markdown(index=False),
        "",
    ]

# Nøkkelparameter-kommentarer
hc  = summ['h_c']['mean']
phiu = summ['phi_u']['mean']
psiR = summ['psi_R']['mean']
srp = summ['sigma_rp']['mean']

lines += [
    "## Nøkkelparametre",
    "",
    f"- **h_c** = {hc:.4f} (p95={summ['h_c']['p95']:.4f}) — {'⚠ treffer priorbegrensning' if hc > 0.985 else '✓ under priorbegrensning'}",
    f"- **phi_u** = {phiu:.4f} (K&M: 0.2192) — kapitalutnyttelse",
    f"- **psi_R** = {psiR:.4f} — renteglatting",
    f"- **sigma_rp** = {srp:.4f} (K&M: 0.006) — {'⚠ overestimert' if srp > 0.010 else '✓ nær K&M'}",
    "",
]

out_path = ROT / "docs" / "D_modellfit_fase2v2.md"
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("\n".join(lines))
print(f"\nRapport lagret: {out_path}")
print(df_par[['Param','Fase2v2 mean','std','ESS','PSRF']].to_string(index=False))
