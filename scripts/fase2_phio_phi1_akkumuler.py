"""
[STAT] Kjøring 9 — akkumulerende MCMC for container-miljø (PE-godkjent 2026-05-20).

Strategi: container-grensen (~30-60 min) er kortere enn overhead (80 min burn-in+rekalib).
Løsning: hopp over burn-in når partial-fil finnes, akkumuler kjeden på tvers av restarter.

Logikk per kjøring:
  1. Hvis partial-fil finnes: varm start (theta fra siste trekk, cov fra eksisterende kjede,
     burnin=500, max_recalib=0) → ~3 min overhead, ~57 min produksjon ≈ 27k trekk
  2. Hvis ingen partial: kald start (burnin=2000, max_recalib=1) → ~10 min overhead

Etter kjøring: concatener ny kjede med eksisterende partial og lagre.
Over 3-4 restarter: 70-90k trekk totalt.

Kjøring 9. Skript kjøres gjentatte ganger til tilstrekkelig ESS er nådd.
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

PARTIAL_PATH = ROT / "data" / "results" / "chain_fase2_phio_phi1_prod_partial.npy"
TEMP_PREFIX  = str(ROT / "data" / "results" / "chain_fase2_phio_phi1_temp")
OUT_PATH     = ROT / "data" / "results" / "chain_fase2_phio_phi1_prod_posterior.json"

print(f"Kjøring 9 akkumulerende — {N_PARAMS} frie parametre")
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
    print(f"\nVARM START: {n_existing} eksisterende trekk funnet i partial-fil")

    # Startverdi = siste trekk i kjeden.
    # Kjeden lagrer psi_R i logit-rom — konverter til begrenset rom for log_posterior.
    theta_start = existing_chain[-1].copy()
    i_psi = PARAM_NAMES.index("psi_R")
    theta_start[i_psi] = 0.01 + 0.91 / (1 + np.exp(-theta_start[i_psi]))

    # Proposal-std fra siste 5000 trekk
    n_cov = min(n_existing, 5000)
    post_std = np.maximum(existing_chain[-n_cov:].std(axis=0), 1e-4)
    # psi_R std i begrenset rom
    psi_R_bounded = 0.01 + 0.91 / (1 + np.exp(-existing_chain[-n_cov:, i_psi]))
    post_std[i_psi] = max(float(psi_R_bounded.std()), 1e-4)

    burnin_n    = 500
    max_recalib = 0
    seed        = int(time.time()) % 10000  # ulik seed per kjøring
    print(f"  burnin={burnin_n}, max_recalib={max_recalib}, seed={seed}")
else:
    existing_chain = None
    n_existing     = 0
    print(f"\nKALD START: ingen partial-fil funnet")

    # Last fra kjøring 8-posterior
    prev_path = ROT / "data" / "results" / "chain_fase2_phio_prod_posterior.json"
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

    # phi_I1: start fra fase2v2-estimat
    theta_start[PARAM_NAMES.index("phi_I1")] = 0.5
    post_std[PARAM_NAMES.index("phi_I1")]    = 0.5

    burnin_n    = 2000
    max_recalib = 1
    seed        = 42
    print(f"  burnin={burnin_n}, max_recalib={max_recalib}, seed={seed}")

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
    print(f"Akkumulert kjede: {n_existing} + {len(chain_new)} = {len(chain_total)} trekk")
else:
    chain_total = chain_new

np.save(PARTIAL_PATH, chain_total)
print(f"Lagret akkumulert kjede: {PARTIAL_PATH} ({len(chain_total)} trekk)")

# ── Posterior-oppsummering ────────────────────────────────────────────────────
# psi_R tilbake til faktisk skala
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
                "n_restarts": meta.get("n_restarts", 0),
                "param_names": PARAM_NAMES,
                "phi_I1_free": True,
                "h_c_fixed":   H_C_FIXED,
                "phi_O": phi_O, "phi_B": phi_B,
                "psrf_max": float(np.nanmax(psrf)),
                "ess_min":  float(min(ess)),
                "modellfix": ["A4a","A4c","CEE","A5","LTV-fortegn-E3E4",
                              "h_c-kalibrering","phi_B-UIP","phi_O-UIP","phi_I1-fri"],
                "begrunnelse": "PE-godkjent (2026-05-20): akkumulerende kjøring"}}

with open(OUT_PATH, "w") as f:
    json.dump(out, f, indent=2)
print(f"Posterior lagret: {OUT_PATH}")

# ── Statusrapport ─────────────────────────────────────────────────────────────
n_psrf_ok = sum(p < 1.10 for p in psrf if np.isfinite(p))
n_ess_ok  = sum(e / n_total >= 0.01 for e in ess)
print(f"\nKonvergens ({n_total} trekk totalt):")
print(f"  PSRF<1.10: {n_psrf_ok}/{N_PARAMS}  |  ESS/n>0.01: {n_ess_ok}/{N_PARAMS}")
print(f"  PSRF_max={np.nanmax(psrf):.3f}  ESS_min={min(ess):.0f}")

print(f"\nNøkkelparametre:")
for nm in ["phi_I1","sigma_rp","psi_R","rho_A","rho_C","rho_rp"]:
    s = summary[nm]
    print(f"  {nm:<12} mean={s['mean']:.4f}  std={s['std']:.4f}  "
          f"[{s['p05']:.4f}, {s['p95']:.4f}]")

if n_psrf_ok == N_PARAMS and n_ess_ok == N_PARAMS:
    print(f"\nALLE KRITERIER BESTÅTT — kjeden er tilstrekkelig ✓")
else:
    print(f"\nKjør skriptet på nytt for å akkumulere flere trekk.")
    print(f"  Mål: {N_PARAMS}/{N_PARAMS} PSRF<1.10 og ESS/n>0.01")
