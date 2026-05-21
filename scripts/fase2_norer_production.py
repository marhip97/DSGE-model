"""
[STAT] Fase 2 no-rer — RER utelatt fra observasjonssettet (PE-godkjent Alt. 4, 2026-05-19).

Hypotese: sigma_rp=0.017 skyldes UIP-feiltilpasning — modellen absorberer
uforklart valutadynamikk i sigma_rp. Uten RER i observasjonssettet bør
sigma_rp falle mot K&M-verdi (~0.006–0.010).

18 frie parametre (sigma_A, phi_I1, h_c alle faste).
h_c=0.938 (K&M, PE-godkjent 2026-05-18), psi_R prior (0.01, 0.92).

Kjøring 6.
"""

import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROT / "src"))

from nemo.estimation.mcmc import (
    PARAM_NAMES, N_PARAMS, KM, PHI_I1_FIXED, H_C_FIXED,
    OBS_NAMES_NO_RER, N_OBS_NO_RER,
    build_H_no_rer, build_Sv_no_rer, log_posterior,
    adaptive_mcmc_with_monitoring, compute_psrf, compute_ess,
)

print(f"Fase 2 no-rer — {N_PARAMS} frie parametre")
print(f"phi_I1={PHI_I1_FIXED} (fast), h_c={H_C_FIXED} (fast, K&M)")
print(f"Observasjoner: {N_OBS_NO_RER} (RER utelatt — PE-godkjent Alt. 4)")
print(f"psi_R prior: Beta(2,2) på (0.01, 0.92)")

data_sti = ROT / "data" / "processed" / "nemo_data_faktisk_v2.csv"
df = pd.read_csv(data_sti, index_col=0)
df.index = pd.to_datetime(df.index)

# Fjern ds_obs-kolonnen fra data
df_no_rer = df.drop(columns=['ds_obs'])
assert list(df_no_rer.columns) == OBS_NAMES_NO_RER, "Kolonneordre mismatch!"

pre  = df_no_rer[df_no_rer.index <= "2019-12-31"].values
post = df_no_rer[df_no_rer.index >= "2022-01-01"].values
H, Sv = build_H_no_rer(), build_Sv_no_rer()

print(f"Data: pre={pre.shape}, post={post.shape}, H={H.shape}")

def idx(*names):
    return [PARAM_NAMES.index(n) for n in names]

blk1 = idx("psi_R", "psi_P1")
blk2 = idx("rho_A", "rho_C", "rho_O", "rho_rp")
blk3 = idx("rho_Ys", "rho_H", "psi_Y", "sigma_Ys")
blk4 = idx("sigma_C", "sigma_O", "sigma_rp", "sigma_i", "sigma_P", "sigma_H")
blk5 = idx("phi_I2", "phi_u")
blocks = [blk1, blk2, blk3, blk4, blk5]

# Startpunkt fra postfix-posterior (kjøring 3 — siste fullstendige kjøring)
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
print(f"psi_R startpunkt: {theta_start[i_psi]:.4f}")

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
    scale_init=0.5, seed=99, verbose=True,
    save_prefix=str(ROT / "data" / "results" / "chain_fase2_norer_prod"),
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
                "obs_names":    OBS_NAMES_NO_RER,
                "rer_utelatt":  True,
                "modellfix": ["A4a","A4c","CEE","A5","LTV-fortegn-E3E4","h_c-kalibrering","no-RER"],
                "begrunnelse": "PE-godkjent Alt. 4 (2026-05-19): test om sigma_rp faller uten RER",
                "psrf_max": float(np.nanmax(psrf)),
                "ess_min":  float(min(ess))}}

out_path = ROT / "data" / "results" / "chain_fase2_norer_prod_posterior.json"
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

# Hurtig sigma_rp-sjekk
sigma_rp_mean = summary["sigma_rp"]["mean"]
print(f"\nsigma_rp posterior: {sigma_rp_mean:.4f}")
if sigma_rp_mean < 0.010:
    print("  → sigma_rp < 0.010 — Alt. 4 BEKREFTET: UIP-feiltilpasning var årsaken")
elif sigma_rp_mean < 0.014:
    print("  → sigma_rp redusert men ikke K&M — delvis bekreftelse")
else:
    print("  → sigma_rp fortsatt > 0.014 — strukturell endring (Alt. 2) nødvendig")
