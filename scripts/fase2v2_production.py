"""
[STAT] Fase 2v2 — re-estimering med Alt. A (NZ=49).

20 parametre inkl. nye phi_u (kapitalutnyttelse).
100k produksjonstrekk, ~25-35 min.
"""

import json
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

print(f"Fase 2v2 — {N_PARAMS} parametre (NZ=49 m/ kapitalutnyttelse)")

data_sti = ROT / "data" / "processed" / "nemo_data_faktisk_v2.csv"
df = pd.read_csv(data_sti, index_col=0)
df.index = pd.to_datetime(df.index)
pre  = df[df.index <= "2019-12-31"].values
post = df[df.index >= "2022-01-01"].values

H, Sv = build_H(), build_Sv()

# Startverdier: Fase 2 posterior + phi_u fra K&M
prev_path = ROT / "data" / "results" / "chain_fase2_prod_posterior.json"
with open(prev_path) as f:
    prev = json.load(f)['summary']

theta_start = np.zeros(N_PARAMS)
post_std    = np.zeros(N_PARAMS)
for i, n in enumerate(PARAM_NAMES):
    if n in prev:
        theta_start[i] = prev[n]['mean']
        post_std[i]    = max(prev[n].get('std', 0.05), 1e-4)
    else:
        theta_start[i] = KM.get(n, 0.5)
        if n == 'phi_u': post_std[i] = 0.05
        elif n == 'phi_I1': post_std[i] = 0.5
        elif n == 'phi_I2': post_std[i] = 2.0
        else: post_std[i] = 0.05

lp = log_posterior(theta_start, H, Sv, pre, post)
print(f"Start lp = {lp:.2f}")
assert np.isfinite(lp), "Start ugyldig"

t0 = time.time()
chain, lp_vec, meta = adaptive_mcmc_with_monitoring(
    Y_pre=pre, Y_post=post, H=H, Sv=Sv,
    theta_init=theta_start, post_std_init=post_std,
    n_production=100000, burnin=10000, adapt_every=500,
    check_every=10000, max_recalib=2,
    psrf_thr=1.10, ess_pct_thr=0.02,
    scale_init=0.5, seed=42, verbose=True,
    save_prefix="data/results/chain_fase2v2_prod",
)
t_total = time.time() - t0
print(f"\nFerdig på {t_total/60:.1f} min")

psrf = compute_psrf(chain)
ess  = [float(compute_ess(chain[:, i])) for i in range(N_PARAMS)]
summary = {n: {'mean': float(chain[:, i].mean()),
               'std':  float(chain[:, i].std()),
               'p05':  float(np.percentile(chain[:, i], 5)),
               'p95':  float(np.percentile(chain[:, i], 95)),
               'ess':  ess[i], 'psrf': float(psrf[i])}
           for i, n in enumerate(PARAM_NAMES)}
out = {'summary': summary,
       'meta': {'n_samples': int(len(chain)),
                'time_min': float(t_total/60),
                'param_names': PARAM_NAMES,
                'psrf_max': float(np.nanmax(psrf)),
                'ess_min': float(min(ess))}}
with open(ROT / "data" / "results" / "chain_fase2v2_prod_posterior.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"Lagret: data/results/chain_fase2v2_prod_posterior.json")
