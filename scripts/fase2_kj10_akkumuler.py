"""
[STAT] Kjøring 10 — akkumulerende MCMC med korrigert modell (PE-godkjent 2026-05-21).

Modellrettelser fra Spor A (PE-godkjent 2026-05-21):
  A4d: Q_K-koeffisient (y−k̂) = 1.0 (ikke alpha_K) → TFP-kanal fikset, 15/15 IRF-krav
  A_phi_L: phi_L = 1.50 (K&M Tabell 8, ned fra 3.00) → sigma_tilde: 5.03 → 3.02

19 frie parametre (identisk med kjøring 9):
  phi_I1 fri, h_c=0.938 fast, sigma_A=0.006 fast
  phi_B=0.0016, phi_O=0.15 i UIP

Akkumuleringsstrategi (identisk med kjøring 9):
  Varm start hvis partial-fil finnes: burnin=500, max_recalib=0 (~3 min overhead)
  Kald start ellers: burnin=2000, max_recalib=1 (~10 min overhead)
  Partial lagres hvert 1000 trekk i mcmc.py.

Forventet: rho_A → ~0.95 (var 0.086 i kj9 pga. feil Q_K-koeff).
"""

import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROT / "src"))

from nemo.estimation.mcmc import (
    PARAM_NAMES, N_PARAMS, KM, H_C_FIXED,
    build_H, build_Sv, log_posterior,
    adaptive_mcmc_with_monitoring, compute_psrf, compute_ess,
)
from nemo.model.parameters import Parameters

phi_O = Parameters().phi_O
phi_B = Parameters().phi_B

PARTIAL_PATH = ROT / "data" / "results" / "chain_kj10_prod_partial.npy"
TEMP_PREFIX  = str(ROT / "data" / "results" / "chain_kj10_temp")
OUT_PATH     = ROT / "data" / "results" / "chain_kj10_prod_posterior.json"

print(f"Kjøring 10 akkumulerende — {N_PARAMS} frie parametre")
print(f"Modellrettelser: A4d (Q_K yk-koeff=1.0), phi_L=1.50 (K&M Tabell 8)")
print(f"phi_I1 fri, h_c={H_C_FIXED} (fast), phi_B={phi_B}, phi_O={phi_O}")

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
blk5 = idx("phi_I1", "phi_I2", "phi_u")
blocks = [blk1, blk2, blk3, blk4, blk5]

# ── Bestem startpunkt ─────────────────────────────────────────────────────────
if PARTIAL_PATH.exists():
    existing_chain = np.load(PARTIAL_PATH)
    n_existing = len(existing_chain)
    print(f"\nVARM START: {n_existing} eksisterende trekk funnet")

    theta_start = existing_chain[-1].copy()
    i_psi = PARAM_NAMES.index("psi_R")
    theta_start[i_psi] = 0.01 + 0.91 / (1 + np.exp(-theta_start[i_psi]))

    n_cov = min(n_existing, 5000)
    post_std = np.maximum(existing_chain[-n_cov:].std(axis=0), 1e-4)
    psi_R_bounded = 0.01 + 0.91 / (1 + np.exp(-existing_chain[-n_cov:, i_psi]))
    post_std[i_psi] = max(float(psi_R_bounded.std()), 1e-4)

    burnin_n    = 500
    max_recalib = 0
    seed        = int(time.time()) % 10000
    print(f"  burnin={burnin_n}, max_recalib={max_recalib}, seed={seed}")
else:
    existing_chain = None
    n_existing     = 0
    print(f"\nKALD START: ingen partial-fil — start fra kj9-posterior")

    # Startpunkt fra kjøring 9 for overlappende parametre
    prev_path = ROT / "data" / "results" / "chain_fase2_phio_phi1_prod_posterior.json"
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
            post_std[i]    = 0.5 if n == "phi_I1" else 0.05

    # rho_A: modellen er korrigert — start fra K&M-verdi (~0.95), ikke kj9 (0.086)
    i_rhoA = PARAM_NAMES.index("rho_A")
    theta_start[i_rhoA] = 0.80   # mellom K&M (0.95) og kj9 (0.086) — lar MCMC finne posterior
    post_std[i_rhoA]    = 0.15   # bred proposal siden ny modell endrer identifikasjonen

    burnin_n    = 2000
    max_recalib = 1
    seed        = 10
    print(f"  burnin={burnin_n}, max_recalib={max_recalib}, seed={seed}")
    print(f"  rho_A startpunkt: {theta_start[i_rhoA]:.2f} (kj9=0.086, K&M=0.95)")

lp = log_posterior(theta_start, H, Sv, pre, post)
print(f"Start lp = {lp:.2f}")
assert np.isfinite(lp), f"Startverdi ugyldig: lp={lp}"

# ── Kjør MCMC ────────────────────────────────────────────────────────────────
t0 = time.time()
chain_new, lp_vec, meta = adaptive_mcmc_with_monitoring(
    Y_pre=pre, Y_post=post, H=H, Sv=Sv,
    theta_init=theta_start, post_std_init=post_std,
    n_production=200000, burnin=burnin_n,
    adapt_every=500, check_every=5000,
    max_recalib=max_recalib,
    psrf_thr=1.10, ess_pct_thr=0.01,
    scale_init=1.0 if existing_chain is not None else 0.5,
    seed=seed, verbose=True,
    save_prefix=TEMP_PREFIX,
    use_reparam=True,
    block_indices=blocks,
)
t_run = time.time() - t0
print(f"\nKjøring ferdig på {t_run/60:.1f} min — {len(chain_new)} nye trekk")

# ── Concatener og lagre ───────────────────────────────────────────────────────
if existing_chain is not None:
    chain_total = np.concatenate([existing_chain, chain_new], axis=0)
    print(f"Akkumulert: {n_existing} + {len(chain_new)} = {len(chain_total)} trekk")
else:
    chain_total = chain_new

np.save(PARTIAL_PATH, chain_total)
print(f"Lagret akkumulert partial: {PARTIAL_PATH} ({len(chain_total)} trekk)")

# ── Posterior-oppsummering ────────────────────────────────────────────────────
i_psi = PARAM_NAMES.index("psi_R")
chain_out = chain_total.copy()
chain_out[:, i_psi] = 0.01 + 0.91 / (1 + np.exp(-chain_total[:, i_psi]))

psrf = compute_psrf(chain_out)
ess  = [float(compute_ess(chain_out[:, i])) for i in range(N_PARAMS)]
n_total = len(chain_total)

summary = {n: {"mean": float(chain_out[:, i].mean()),
               "std":  float(chain_out[:, i].std()),
               "p05":  float(np.percentile(chain_out[:, i], 5)),
               "p95":  float(np.percentile(chain_out[:, i], 95)),
               "ess":  ess[i], "psrf": float(psrf[i])}
           for i, n in enumerate(PARAM_NAMES)}

out = {"summary": summary,
       "meta": {"n_samples": n_total,
                "n_this_run": len(chain_new),
                "param_names": PARAM_NAMES,
                "phi_I1_free": True,
                "h_c_fixed":   H_C_FIXED,
                "phi_O": phi_O, "phi_B": phi_B,
                "psrf_max": float(np.nanmax(psrf)),
                "ess_min":  float(min(ess)),
                "modellfix": ["A4a","A4c","CEE","A5","LTV-fortegn-E3E4",
                              "h_c-kalibrering","phi_B-UIP","phi_O-UIP","phi_I1-fri",
                              "A4d-QK-yk-koeff-1.0","A_phi_L-1.50"],
                "begrunnelse": "PE-godkjent (2026-05-21): A4d Q_K-koeff + phi_L=1.50"}}

with open(OUT_PATH, "w") as f:
    json.dump(out, f, indent=2)
print(f"Posterior lagret: {OUT_PATH}")

# ── Statusrapport ─────────────────────────────────────────────────────────────
n_psrf_ok = sum(p < 1.10 for p in psrf if np.isfinite(p))
n_ess_ok  = sum(e / n_total >= 0.01 for e in ess)
print(f"\nKonvergens ({n_total} trekk totalt):")
print(f"  PSRF<1.10: {n_psrf_ok}/{N_PARAMS}  |  ESS/n>0.01: {n_ess_ok}/{N_PARAMS}")
print(f"  PSRF_max={np.nanmax(psrf):.3f}  ESS_min={min(ess):.0f}")

print(f"\nNøkkelparametre (inkl. A4d-effekt på rho_A):")
for nm in ["rho_A", "phi_I1", "sigma_rp", "psi_R", "rho_C", "rho_rp", "phi_u"]:
    sv = summary[nm]
    print(f"  {nm:<12} mean={sv['mean']:.4f}  std={sv['std']:.4f}  "
          f"[{sv['p05']:.4f}, {sv['p95']:.4f}]  ESS={sv['ess']:.0f}")

if n_psrf_ok == N_PARAMS and n_ess_ok == N_PARAMS:
    print(f"\nALLE KRITERIER BESTÅTT — kjeden er tilstrekkelig ✓")
else:
    print(f"\nKjør skriptet på nytt for å akkumulere flere trekk.")
    print(f"  Mål: {N_PARAMS}/{N_PARAMS} PSRF<1.10 og ESS/n>0.01")
