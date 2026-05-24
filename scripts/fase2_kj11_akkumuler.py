"""
[STAT] Kjøring 11 — akkumulerende MCMC fra kj10-posterior.

Endringer fra kjøring 10:
  rho_rp i egen blokk (blk2b) — adresserer strukturelt lav ESS/n=0.0078 (ACL≈140).
  Blokkstruktur kj11:
    blk1 : {psi_R, psi_P1}
    blk2a: {rho_A, rho_C, rho_O}        ← rho_rp tatt ut
    blk2b: {rho_rp}                      ← dedikert blokk
    blk3 : {rho_Ys, rho_H, psi_Y, sigma_Ys}
    blk4 : {sigma_C, sigma_O, sigma_rp, sigma_i, sigma_P, sigma_H}
    blk5 : {phi_I1, phi_I2, phi_u}

19 frie parametre (identisk med kjøring 10):
  phi_I1 fri, h_c=0.938 fast, sigma_A=0.006 fast
  phi_B=0.0016, phi_O=0.15 i UIP

Startpunkt: kj10-posterior (chain_kj10_prod_posterior.json).
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

PARTIAL_PATH = ROT / "data" / "results" / "chain_kj11_prod_partial.npy"
TEMP_PREFIX  = str(ROT / "data" / "results" / "chain_kj11_temp")
OUT_PATH     = ROT / "data" / "results" / "chain_kj11_prod_posterior.json"

print(f"Kjøring 11 akkumulerende — {N_PARAMS} frie parametre")
print(f"Endring: rho_rp i dedikert blokk (kj10 ESS/n=0.0078, ACL≈140)")
print(f"phi_I1 fri, h_c={H_C_FIXED} (fast), phi_B={phi_B}, phi_O={phi_O}")

data_sti = ROT / "data" / "processed" / "nemo_data_faktisk_v2.csv"
df = pd.read_csv(data_sti, index_col=0)
df.index = pd.to_datetime(df.index)
pre  = df[df.index <= "2019-12-31"].values
post = df[df.index >= "2022-01-01"].values
H, Sv = build_H(), build_Sv()

def idx(*names):
    return [PARAM_NAMES.index(n) for n in names]

blk1  = idx("psi_R", "psi_P1")
blk2a = idx("rho_A", "rho_C", "rho_O")          # rho_rp tatt ut
blk2b = idx("rho_rp",)                            # dedikert blokk for rho_rp
blk3  = idx("rho_Ys", "rho_H", "psi_Y", "sigma_Ys")
blk4  = idx("sigma_C", "sigma_O", "sigma_rp", "sigma_i", "sigma_P", "sigma_H")
blk5  = idx("phi_I1", "phi_I2", "phi_u")
blocks = [blk1, blk2a, blk2b, blk3, blk4, blk5]

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

    max_recalib = 1 if n_existing < 50000 else 0
    burnin_n    = 1000 if n_existing < 50000 else 500
    seed        = int(time.time()) % 10000
    print(f"  burnin={burnin_n}, max_recalib={max_recalib}, seed={seed}")
else:
    existing_chain = None
    n_existing     = 0
    print(f"\nKALD START: ingen partial-fil — start fra kj10-posterior")

    prev_path = ROT / "data" / "results" / "chain_kj10_prod_posterior.json"
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

    # rho_rp: bred proposal for den dedikerte blokken
    i_rp = PARAM_NAMES.index("rho_rp")
    post_std[i_rp] = max(post_std[i_rp] * 2.0, 0.15)

    burnin_n    = 2000
    max_recalib = 2
    seed        = 11
    print(f"  burnin={burnin_n}, max_recalib={max_recalib}, seed={seed}")
    print(f"  rho_rp startpunkt: {theta_start[i_rp]:.3f} std={post_std[i_rp]:.3f}")

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
       "meta": {"n_samples": int(n_total),
                "n_this_run": int(len(chain_new)),
                "param_names": list(PARAM_NAMES),
                "phi_I1_free": True,
                "h_c_fixed":   float(H_C_FIXED),
                "phi_O": float(phi_O), "phi_B": float(phi_B),
                "psrf_max": float(np.nanmax(psrf)),
                "ess_min":  float(min(ess)),
                "modellfix": ["A4a","A4c","CEE","A5","LTV-fortegn-E3E4",
                              "h_c-kalibrering","phi_B-UIP","phi_O-UIP","phi_I1-fri",
                              "A4d-QK-yk-koeff-1.0","A_phi_L-1.50"],
                "blokk_endring": "rho_rp dedikert blokk (blk2b) — kj11-forbedring",
                "begrunnelse": "kj11: rho_rp dedikert blokk for bedre ESS"}}

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
for nm in ["rho_rp", "rho_A", "phi_I1", "sigma_rp", "psi_R", "rho_C", "phi_u"]:
    sv = summary[nm]
    print(f"  {nm:<12} mean={sv['mean']:.4f}  std={sv['std']:.4f}  "
          f"[{sv['p05']:.4f}, {sv['p95']:.4f}]  ESS={sv['ess']:.0f}")

if n_psrf_ok == N_PARAMS and n_ess_ok == N_PARAMS:
    print(f"\nALLE KRITERIER BESTÅTT ✓")
else:
    print(f"\nKjør skriptet på nytt for å akkumulere flere trekk.")
    print(f"  Mål: {N_PARAMS}/{N_PARAMS} PSRF<1.10 og ESS/n>0.01")
