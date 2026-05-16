"""
[STAT] Fase 2 produksjonskjøring med logit-reparametrisering (C5 §2+§4).

200k produksjonstrekk, use_reparam=True.
h_c og psi_R sampler i ubegrenset logit-rom; chain lagres i naturlig rom.
Startpunkt: Fase 2v2 posterior.

Krav (C5 §4): PSRF < 1.05, ESS/n > 0.02 for ALLE 20 parametre.
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

print(f"Fase 2 reparam — {N_PARAMS} parametre, use_reparam=True")
print(f"Krav: PSRF < 1.05, ESS/n > 0.02 per parameter")

data_sti = ROT / "data" / "processed" / "nemo_data_faktisk_v2.csv"
df = pd.read_csv(data_sti, index_col=0)
df.index = pd.to_datetime(df.index)
pre  = df[df.index <= "2019-12-31"].values
post = df[df.index >= "2022-01-01"].values

H, Sv = build_H(), build_Sv()

# Startpunkt: Fase 2v2 posterior (100k trekk, siste kjøring)
prev_path = ROT / "data" / "results" / "chain_fase2v2_prod_posterior.json"
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
        post_std[i]    = 0.05

lp = log_posterior(theta_start, H, Sv, pre, post)
print(f"Start lp = {lp:.2f}")
assert np.isfinite(lp), "Startverdi ugyldig"

t0 = time.time()
chain, lp_vec, meta = adaptive_mcmc_with_monitoring(
    Y_pre=pre, Y_post=post, H=H, Sv=Sv,
    theta_init=theta_start, post_std_init=post_std,
    n_production=200000, burnin=20000,
    adapt_every=500, check_every=20000,
    max_recalib=3,
    psrf_thr=1.05, ess_pct_thr=0.02,
    scale_init=0.4, seed=42, verbose=True,
    save_prefix=str(ROT / "data" / "results" / "chain_fase2_reparam_prod"),
    use_reparam=True,
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
       'meta': {**meta,
                'n_samples': int(len(chain)),
                'time_min': float(t_total / 60),
                'param_names': PARAM_NAMES,
                'psrf_max': float(np.nanmax(psrf)),
                'ess_min': float(min(ess))}}

out_path = ROT / "data" / "results" / "chain_fase2_reparam_prod_posterior.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"Lagret: {out_path}")

# Rask konvergensoppsummering
n_psrf_ok = sum(p < 1.05 for p in psrf if np.isfinite(p))
n_ess_ok  = sum(e / 200000 >= 0.02 for e in ess)
print(f"\nKonvergens: PSRF<1.05 for {n_psrf_ok}/{N_PARAMS}  |  ESS/n>0.02 for {n_ess_ok}/{N_PARAMS}")
if n_psrf_ok == N_PARAMS and n_ess_ok == N_PARAMS:
    print("ALLE KRITERIER BESTÅTT — klar for analyse")
else:
    print("Noen kriterier ikke nådd — vurder lengre kjøring eller HMC")
