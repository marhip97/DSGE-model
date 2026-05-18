"""
[STAT] Fase 2 hc-fix — h_c kalibrert fast til K&M=0.938 (PE-godkjent 2026-05-18).

18 frie parametre (sigma_A, phi_I1, h_c alle faste).
sigma_rp estimeres fritt (C3-fix rullet tilbake — fiksering forverret IRF).
psi_R prior: Beta(2,2) på (0.01, 0.92) (PE-godkjent 2026-05-18).

Hypotese: h_c=0.938 gjenoppretter konsumkanalen (a3_W=0.032 vs 0.006 med h_c=0.988)
og gir BNP-ratio mot NB Memo 3/2024 ned fra 6-8× til ~1.5-2×.

Kjøring 5.
"""

import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROT / "src"))

from nemo.estimation.mcmc import (
    PARAM_NAMES, N_PARAMS, KM, PHI_I1_FIXED, H_C_FIXED,
    build_H, build_Sv, log_posterior,
    adaptive_mcmc_with_monitoring, compute_psrf, compute_ess,
)

print(f"Fase 2 hc-fix — {N_PARAMS} frie parametre")
print(f"phi_I1={PHI_I1_FIXED} (fast), h_c={H_C_FIXED} (fast, K&M)")
print(f"sigma_rp: fri (C3-fix rullet tilbake)")
print(f"psi_R prior: Beta(2,2) på (0.01, 0.92)")

data_sti = ROT / "data" / "processed" / "nemo_data_faktisk_v2.csv"
df = pd.read_csv(data_sti, index_col=0)
df.index = pd.to_datetime(df.index)
pre  = df[df.index <= "2019-12-31"].values
post = df[df.index >= "2022-01-01"].values
H, Sv = build_H(), build_Sv()

def idx(*names):
    return [PARAM_NAMES.index(n) for n in names]

blk1 = idx("psi_R", "psi_P1")
blk2 = idx("rho_A", "rho_C", "rho_O", "rho_rp")
blk3 = idx("rho_Ys", "rho_H", "psi_Y", "sigma_Ys")
blk4 = idx("sigma_C", "sigma_O", "sigma_rp", "sigma_i", "sigma_P", "sigma_H")
blk5 = idx("phi_I2", "phi_u")
blocks = [blk1, blk2, blk3, blk4, blk5]

# Startpunkt fra postfix-posterior (kjøring 3 — siste gode kjøring)
prev_path = ROT / "data" / "results" / "chain_fase2_postfix_prod_posterior.json"
with open(prev_path) as f:
    prev_summ = json.load(f)["summary"]

theta_start = np.zeros(N_PARAMS)
post_std    = np.zeros(N_PARAMS)
for i, n in enumerate(PARAM_NAMES):
    if n in prev_summ:
        theta_start[i] = prev_summ[n]["mean"]
        post_std[i]    = max(prev_summ[n].get("std", 0.05), 1e-4)
    else:
        theta_start[i] = KM.get(n, 0.5)
        post_std[i]    = 0.05

i_psi = PARAM_NAMES.index("psi_R")
print(f"psi_R startpunkt: {theta_start[i_psi]:.4f} (innenfor prior 0.92)")

lp = log_posterior(theta_start, H, Sv, pre, post)
print(f"\nStart lp = {lp:.2f}")
assert np.isfinite(lp), f"Startverdi ugyldig: lp={lp}"

t0 = time.time()
chain, lp_vec, meta = adaptive_mcmc_with_monitoring(
    Y_pre=pre, Y_post=post, H=H, Sv=Sv,
    theta_init=theta_start, post_std_init=post_std,
    n_production=200000, burnin=20000,
    adapt_every=500, check_every=20000,
    max_recalib=3,
    psrf_thr=1.05, ess_pct_thr=0.02,
    scale_init=0.5, seed=77, verbose=True,
    save_prefix=str(ROT / "data" / "results" / "chain_fase2_hcfix_prod"),
    use_reparam=True,
    block_indices=blocks,
)
t_total = time.time() - t0
print(f"\nFerdig på {t_total/60:.1f} min")

psrf = compute_psrf(chain)
ess  = [float(compute_ess(chain[:, i])) for i in range(N_PARAMS)]
summary = {n: {"mean": float(chain[:, i].mean()),
               "std":  float(chain[:, i].std()),
               "p05":  float(np.percentile(chain[:, i], 5)),
               "p95":  float(np.percentile(chain[:, i], 95)),
               "ess":  ess[i], "psrf": float(psrf[i])}
           for i, n in enumerate(PARAM_NAMES)}

out = {"summary": summary,
       "meta": {**meta,
                "n_samples": int(len(chain)),
                "time_min":  float(t_total / 60),
                "param_names": PARAM_NAMES,
                "phi_I1_fixed": PHI_I1_FIXED,
                "h_c_fixed":    H_C_FIXED,
                "modellfix": ["A4a","A4c","CEE","A5","LTV-fortegn-E3E4","h_c-kalibrering"],
                "prior_endringer": [
                    "psi_R: Beta(2,2)/(0.01,0.85) -> Beta(2,2)/(0.01,0.92)",
                    "h_c: fjernet fra estimering, fast=0.938 (K&M, PE-godkjent C2 Alt A)",
                    "sigma_rp: fri igjen (C3-fix rullet tilbake)"
                ],
                "psrf_max": float(np.nanmax(psrf)),
                "ess_min":  float(min(ess))}}

out_path = ROT / "data" / "results" / "chain_fase2_hcfix_prod_posterior.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"Lagret: {out_path}")

n_psrf_ok = sum(p < 1.05 for p in psrf if np.isfinite(p))
n_ess_ok  = sum(e / 200000 >= 0.02 for e in ess)
print(f"\nKonvergens: PSRF<1.05 for {n_psrf_ok}/{N_PARAMS}  |  ESS/n>0.02 for {n_ess_ok}/{N_PARAMS}")
if n_psrf_ok == N_PARAMS and n_ess_ok == N_PARAMS:
    print("ALLE KRITERIER BESTÅTT ✓")
else:
    print("Noen kriterier ikke nådd")
