"""
[STAT] Fase 2 produksjon — 5-blokks Metropolis-within-Gibbs (PE-godkjent 2026-05-16).

Forbedring fra blokk-v1 (2 blokker):
  - Blokk 2 (17 param) splittes i 4 under-blokker — reduserer IAT for rho_H, psi_Y
  - Max |r| innenfor ny blokk 2 er 0.39 — dimensjonalitet var årsaken til lav ESS

5 blokker:
  Blokk 1: {h_c, psi_R, psi_P1}           — r=0.91 (løst i forrige runde)
  Blokk 2: {rho_A, rho_C, rho_O, rho_rp}  — AR-parametre for TFP/konsum/olje/rp
  Blokk 3: {rho_Ys, rho_H, psi_Y, sigma_Ys} — rho_Ys-klynge (r=0.39)
  Blokk 4: {sigma_C, sigma_O, sigma_rp, sigma_i, sigma_P, sigma_H} — sjokk-std
  Blokk 5: {phi_I1, phi_I2, phi_u}         — strukturelle parametre

Start: chain_fase2_blokk_prod posterior (riktig modus, god kovarians).
Krav: PSRF < 1.05, ESS/n > 0.02 for ALLE 20 parametre.
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

def idx(*names):
    return [PARAM_NAMES.index(n) for n in names]

blk1 = idx("h_c", "psi_R", "psi_P1")
blk2 = idx("rho_A", "rho_C", "rho_O", "rho_rp")
blk3 = idx("rho_Ys", "rho_H", "psi_Y", "sigma_Ys")
blk4 = idx("sigma_C", "sigma_O", "sigma_rp", "sigma_i", "sigma_P", "sigma_H")
blk5 = idx("phi_I1", "phi_I2", "phi_u")
blocks = [blk1, blk2, blk3, blk4, blk5]

print(f"Fase 2 — 5-blokks sampler, {N_PARAMS} parametre")
for k, blk in enumerate(blocks, 1):
    print(f"  Blokk {k} ({len(blk)} param): {{{', '.join(PARAM_NAMES[i] for i in blk)}}}")
print(f"Krav: PSRF < 1.05, ESS/n > 0.02 per parameter")

data_sti = ROT / "data" / "processed" / "nemo_data_faktisk_v2.csv"
df = pd.read_csv(data_sti, index_col=0)
df.index = pd.to_datetime(df.index)
pre  = df[df.index <= "2019-12-31"].values
post = df[df.index >= "2022-01-01"].values
H, Sv = build_H(), build_Sv()

# Startpunkt: siste blokk-kjøring (allerede ved riktig modus og god empirisk kovarians)
prev_path = ROT / "data" / "results" / "chain_fase2_blokk_prod_posterior.json"
with open(prev_path) as f:
    prev = json.load(f)["summary"]

theta_start = np.array([prev[n]["mean"] for n in PARAM_NAMES])
post_std    = np.array([max(prev[n].get("std", 0.05), 1e-4) for n in PARAM_NAMES])

lp = log_posterior(theta_start, H, Sv, pre, post)
print(f"\nStart lp = {lp:.2f}")
assert np.isfinite(lp), "Startverdi ugyldig"

t0 = time.time()
chain, lp_vec, meta = adaptive_mcmc_with_monitoring(
    Y_pre=pre, Y_post=post, H=H, Sv=Sv,
    theta_init=theta_start, post_std_init=post_std,
    n_production=200000, burnin=20000,
    adapt_every=500, check_every=20000,
    max_recalib=3,
    psrf_thr=1.05, ess_pct_thr=0.02,
    scale_init=0.5, seed=7, verbose=True,
    save_prefix=str(ROT / "data" / "results" / "chain_fase2_5blokk_prod"),
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
                "time_min": float(t_total / 60),
                "param_names": PARAM_NAMES,
                "psrf_max": float(np.nanmax(psrf)),
                "ess_min": float(min(ess)),
                "n_blocks": len(blocks)}}

out_path = ROT / "data" / "results" / "chain_fase2_5blokk_prod_posterior.json"
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
