"""
[STAT] Fase 2 — produksjonskjøring MCMC (10k burnin + 100k produksjon).

Bruker:
  - Empirisk kovariansproposal (Haario AM) etter burn-in
  - 19 parametre (inkl. phi_I1, phi_I2)
  - COVID-hull (pre ≤2019Q4, post ≥2022Q1)

Forventet tid: ~35-40 min. Resultat: data/results/chain_fase2_prod*.{npy,json}.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROT / "src"))

from nemo.estimation.mcmc import (
    PARAM_NAMES, N_PARAMS, KM,
    build_H, build_Sv, log_posterior,
    adaptive_mcmc_with_monitoring, compute_psrf, compute_ess,
)

print(f"Fase 2 produksjonskjøring — {N_PARAMS} parametre")
print(f"  {PARAM_NAMES}")

data_sti = ROT / "data" / "processed" / "nemo_data_faktisk_v2.csv"
df = pd.read_csv(data_sti, index_col=0)
df.index = pd.to_datetime(df.index)
pre  = df[df.index <= "2019-12-31"].values
post = df[df.index >= "2022-01-01"].values
print(f"  Pre-COVID: {len(pre)} kv  Post-COVID: {len(post)} kv")

H  = build_H()
Sv = build_Sv()

# Startverdier: K&M (eller forrige posterior hvis tilgjengelig)
theta_start = np.array([KM.get(n, 0.5) for n in PARAM_NAMES])
post_std    = np.array([0.05]*N_PARAMS)
for i, n in enumerate(PARAM_NAMES):
    if n == 'phi_I1': post_std[i] = 1.0
    if n == 'phi_I2': post_std[i] = 2.0

# Forsøk laste forrige posterior for sigma_C, h_c osv.
prev_path = ROT / "data" / "results" / "chain_v3_v2_posterior.json"
if prev_path.exists():
    with open(prev_path) as f:
        prev = json.load(f)
    summ = prev['summary']
    for i, n in enumerate(PARAM_NAMES):
        if n in summ:
            theta_start[i] = summ[n]['mean']
            post_std[i]    = max(summ[n].get('std', 0.05), 1e-4)
    print(f"  Lastet startverdier fra {prev_path.name}")

lp = log_posterior(theta_start, H, Sv, pre, post)
print(f"  Start lp = {lp:.2f}")
if not np.isfinite(lp):
    print("  ADVARSEL: Startverdi ugyldig — bruker K&M.")
    theta_start = np.array([KM.get(n, 0.5) for n in PARAM_NAMES])
    post_std    = np.array([0.05]*N_PARAMS)
    for i, n in enumerate(PARAM_NAMES):
        if n == 'phi_I1': post_std[i] = 1.0
        if n == 'phi_I2': post_std[i] = 2.0

t0 = time.time()
chain, lp_vec, meta = adaptive_mcmc_with_monitoring(
    Y_pre=pre, Y_post=post, H=H, Sv=Sv,
    theta_init=theta_start, post_std_init=post_std,
    n_production=100000, burnin=10000, adapt_every=500,
    check_every=10000, max_recalib=2,
    psrf_thr=1.10, ess_pct_thr=0.02,
    scale_init=0.5, seed=42, verbose=True,
    save_prefix="data/results/chain_fase2_prod",
)
t_total = time.time() - t0
print(f"\nFerdig på {t_total/60:.1f} min")

# Lagre summary
psrf = compute_psrf(chain)
ess  = [float(compute_ess(chain[:, i])) for i in range(N_PARAMS)]
summary = {
    n: {'mean': float(chain[:, i].mean()),
        'std':  float(chain[:, i].std()),
        'p05':  float(np.percentile(chain[:, i], 5)),
        'p95':  float(np.percentile(chain[:, i], 95)),
        'ess':  ess[i],
        'psrf': float(psrf[i])}
    for i, n in enumerate(PARAM_NAMES)
}
out = {
    'summary': summary,
    'meta': {'n_samples': int(len(chain)),
             'time_min': float(t_total/60),
             'param_names': PARAM_NAMES,
             'psrf_max': float(np.nanmax(psrf)),
             'ess_min': float(min(ess))},
}
out_path = ROT / "data" / "results" / "chain_fase2_prod_posterior.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"Lagret: {out_path}")
