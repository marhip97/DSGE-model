"""
[STAT] Fase 2 kjøring 9 — phi_I1 fri + phi_B + phi_O (PE-godkjent 2026-05-20).

Bakgrunn: B5-benchmark avdekket at phi_I1=4.0 (fast, K&M) gir BNP-respons 0.4×
NB Memo 3/2024 Figur 1 — altfor liten. Fase2v2 med phi_I1 estimert fritt landet
på ~0.5 og traff NB eksakt (-0.447 vs -0.450). K&M-kalibreringen passer ikke
norske data for investeringsjusteringskostnader.

19 frie parametre: phi_I1 fri igjen, sigma_A og h_c faste.
Øvrige forbedringer beholdes: phi_B=0.0016, phi_O=0.15 i UIP, alle modellfix.

Prior phi_I1: Normal(2.0, 2.0) på (0.1, 15.0).
Startpunkt: kjøring 8 posterior for overlappende parametre, KM for phi_I1.

Kjøring 9.
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
print(f"Fase 2 kjøring 9 — {N_PARAMS} frie parametre")
print(f"phi_I1 fri igjen (PE-godkjent 2026-05-20), h_c={H_C_FIXED} (fast)")
print(f"phi_B={phi_B}, phi_O={phi_O} i UIP")
print(f"Prior phi_I1: Normal(2.0, 2.0) på (0.1, 15.0)")
print(f"psi_R prior: Beta(2,2) på (0.01, 0.92)")
assert "phi_I1" in PARAM_NAMES, f"phi_I1 mangler fra PARAM_NAMES — sjekk mcmc.py"

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

# Startpunkt fra kjøring 8 for overlappende parametre
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

# phi_I1 startpunkt: fase2v2-estimat ~0.5 (bedre enn K&M=4.0 for norske data)
i_phi1 = PARAM_NAMES.index("phi_I1")
theta_start[i_phi1] = 0.5
post_std[i_phi1]    = 0.5
print(f"phi_I1 startpunkt: {theta_start[i_phi1]:.2f} (fase2v2-estimat)")

i_psi = PARAM_NAMES.index("psi_R")
print(f"psi_R startpunkt: {theta_start[i_psi]:.4f}")

from nemo.estimation.mcmc import log_posterior
lp = log_posterior(theta_start, H, Sv, pre, post)
print(f"\nStart lp = {lp:.2f}")
assert np.isfinite(lp), f"Startverdi ugyldig: lp={lp}"

t0 = time.time()
chain, lp_vec, meta = adaptive_mcmc_with_monitoring(
    Y_pre=pre, Y_post=post, H=H, Sv=Sv,
    theta_init=theta_start, post_std_init=post_std,
    n_production=200000, burnin=20000,
    adapt_every=500, check_every=5000,
    max_recalib=3,
    psrf_thr=1.05, ess_pct_thr=0.02,
    scale_init=0.5, seed=99, verbose=True,
    save_prefix=str(ROT / "data" / "results" / "chain_fase2_phio_phi1_prod"),
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
                "phi_I1_free": True,
                "h_c_fixed":   H_C_FIXED,
                "phi_O":       phi_O,
                "phi_B":       phi_B,
                "modellfix": ["A4a","A4c","CEE","A5","LTV-fortegn-E3E4",
                              "h_c-kalibrering","phi_B-UIP","phi_O-UIP","phi_I1-fri"],
                "begrunnelse": "PE-godkjent (2026-05-20): phi_I1 fri — B5 viste 0.4× BNP med phi_I1=4.0",
                "psrf_max": float(np.nanmax(psrf)),
                "ess_min":  float(min(ess))}}

out_path = ROT / "data" / "results" / "chain_fase2_phio_phi1_prod_posterior.json"
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

phi_I1_mean = summary["phi_I1"]["mean"]
print(f"\nphi_I1 posterior: {phi_I1_mean:.3f}  (K&M=4.0, fase2v2≈0.5)")
sigma_rp_mean = summary["sigma_rp"]["mean"]
print(f"sigma_rp posterior: {sigma_rp_mean:.4f}  (K&M=0.006, kj8=0.014)")

# B5-indikator
from nemo.model.equations import E_i, I_R, Y as Y_idx, build_matrices_v3
from nemo.model.parameters import Parameters as BaseP
from nemo.solver.blanchard_kahn import compute_irf, solve as bk_solve
import warnings

class Pt(BaseP): pass
for i, n in enumerate(PARAM_NAMES):
    setattr(Pt, n, float(summary[n]["mean"]))
setattr(Pt, "sigma_A", 0.006)
setattr(Pt, "h_c", H_C_FIXED)

try:
    G0, G1, Psi, Pi = build_matrices_v3(Pt)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        T, R, diag = bk_solve(G0, G1, Psi, Pi, verbose=False)
    if diag["stable"]:
        irf = compute_irf(T, R, E_i, 0.0025, T_periods=20)
        topp = float(np.max(irf[:, I_R]))
        if topp > 0:
            irf_n = irf / topp
            bnp_q4 = irf_n[3, Y_idx]
            print(f"\nB5 indikator — BNP q4: {bnp_q4:+.3f}  (NB: -0.450, mål: ~1×)")
            print(f"  ratio: {abs(bnp_q4/0.45):.2f}× NB")
except Exception as e:
    print(f"\nB5 beregning feilet: {e}")
