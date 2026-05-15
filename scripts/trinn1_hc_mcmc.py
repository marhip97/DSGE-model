"""
[STAT] Trinn 1 — Diagnose: h_c-prior 0.9995 → 0.90.

Mål: Avgjøre om h_c-grensen driver TFP→BNP-bug.
  - Hvis ny posterior h_c < 0.85: B (prior-relaksering) løser problemet
  - Hvis posterior treffer ny grense 0.90: strukturelt problem, eskalere Alt. A

30k trekk: 5k burnin + 25k produksjon. ~10-12 min.
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

print(f"Trinn 1 — {N_PARAMS} parametre, h_c-prior øvre = 0.90")

data_sti = ROT / "data" / "processed" / "nemo_data_faktisk_v2.csv"
df = pd.read_csv(data_sti, index_col=0)
df.index = pd.to_datetime(df.index)
pre  = df[df.index <= "2019-12-31"].values
post = df[df.index >= "2022-01-01"].values

H  = build_H()
Sv = build_Sv()

# Startverdier: Fase 2 posterior, MEN h_c justert ned siden 0.988 > 0.90
prev_path = ROT / "data" / "results" / "chain_fase2_prod_posterior.json"
with open(prev_path) as f:
    prev = json.load(f)
summ = prev['summary']

theta_start = np.zeros(N_PARAMS)
post_std    = np.zeros(N_PARAMS)
for i, n in enumerate(PARAM_NAMES):
    if n in summ:
        theta_start[i] = summ[n]['mean']
        post_std[i]    = max(summ[n].get('std', 0.05), 1e-4)
    else:
        theta_start[i] = KM.get(n, 0.5)
        post_std[i]    = 0.05
    if n == 'h_c':
        theta_start[i] = 0.85   # innenfor nytt prior [0.30, 0.90]
        post_std[i]    = 0.03   # romsligere startstd

# phi_I-startverdier
for i, n in enumerate(PARAM_NAMES):
    if n == 'phi_I1' and post_std[i] < 0.1: post_std[i] = 0.5
    if n == 'phi_I2' and post_std[i] < 0.5: post_std[i] = 2.0

lp0 = log_posterior(theta_start, H, Sv, pre, post)
print(f"  Startverdi h_c={theta_start[PARAM_NAMES.index('h_c')]:.3f}, lp={lp0:.2f}")
assert np.isfinite(lp0), "Startverdi ugyldig"

t0 = time.time()
chain, lp_vec, meta = adaptive_mcmc_with_monitoring(
    Y_pre=pre, Y_post=post, H=H, Sv=Sv,
    theta_init=theta_start, post_std_init=post_std,
    n_production=25000, burnin=5000, adapt_every=300,
    check_every=5000, max_recalib=2,
    psrf_thr=1.15, ess_pct_thr=0.01,
    scale_init=0.4, seed=42, verbose=True,
    save_prefix="data/results/chain_trinn1_hc",
)
t_total = time.time() - t0
print(f"\nFerdig på {t_total/60:.1f} min")

# Resultat
psrf = compute_psrf(chain)
ess  = [float(compute_ess(chain[:, i])) for i in range(N_PARAMS)]
summary = {n: {'mean': float(chain[:, i].mean()),
               'std':  float(chain[:, i].std()),
               'p05':  float(np.percentile(chain[:, i], 5)),
               'p95':  float(np.percentile(chain[:, i], 95)),
               'ess':  ess[i],
               'psrf': float(psrf[i])}
           for i, n in enumerate(PARAM_NAMES)}
hc_mean = summary['h_c']['mean']
hc_p95  = summary['h_c']['p95']
print(f"\n  ▶ h_c posterior: mean={hc_mean:.4f}, p95={hc_p95:.4f}")
print(f"    Ny prior øvre grense: 0.90")

if hc_p95 > 0.895:
    print(f"\n  ⚠ KONKLUSJON: h_c treffer ny grense → STRUKTURELT problem, eskalere Alt. A")
elif hc_mean > 0.85:
    print(f"\n  ⚠ KONKLUSJON: h_c fortsatt høy — Alt. B løser delvis, anbefale Alt. A i tillegg")
else:
    print(f"\n  ✓ KONKLUSJON: h_c flyttet seg vesentlig — Alt. B fungerer")

out = {
    'summary': summary,
    'meta': {'n_samples': int(len(chain)),
             'time_min': float(t_total/60),
             'param_names': PARAM_NAMES,
             'psrf_max': float(np.nanmax(psrf)),
             'ess_min': float(min(ess)),
             'hc_prior_upper': 0.90}}
with open(ROT / "data" / "results" / "chain_trinn1_hc_posterior.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"  Lagret: data/results/chain_trinn1_hc_posterior.json")
