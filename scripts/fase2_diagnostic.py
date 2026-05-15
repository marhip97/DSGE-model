"""
[STAT] Fase 2 — diagnostisk MCMC-kjøring.

Kort kjøring (2k burnin + 5k prod) for å verifisere:
  1. Empirisk-kovarians-proposal aktiveres etter burn-in
  2. phi_I1/phi_I2 estimeres uten å bryte BK-stabilitet
  3. Sterke korrelasjoner (sigma_C ↔ h_c) fanges av C_prop
  4. Acceptance rate og ESS er rimelige

Tar ~5-10 min på laptop. Full produksjon (200k) kjøres separat.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROT / "src"))

from nemo.estimation.mcmc import (
    PARAM_NAMES, N_PARAMS, KM, SIGMA_A_FIXED,
    build_H, build_Sv, log_posterior,
    adaptive_mcmc_with_monitoring, print_diagnostics,
    compute_psrf, compute_ess,
)

# ── Last data ─────────────────────────────────────────────────────────────────
print(f"Estimerte parametre (Fase 2): {N_PARAMS} stk")
print(f"  → {PARAM_NAMES}")

data_sti = ROT / "data" / "processed" / "nemo_data_faktisk_v2.csv"
df = pd.read_csv(data_sti, index_col=0)
df.index = pd.to_datetime(df.index)
print(f"\nLastet data: {df.shape[0]} kv ({df.index[0].date()} → {df.index[-1].date()})")

# Pre-COVID: ≤ 2019Q4, Post-COVID: ≥ 2022Q1
pre  = df[df.index <= "2019-12-31"].values
post = df[df.index >= "2022-01-01"].values
print(f"  Pre-COVID:  {len(pre)} kv")
print(f"  Post-COVID: {len(post)} kv")

H  = build_H()
Sv = build_Sv()

# ── Startverdier: K&M + nye phi_I-verdier ────────────────────────────────────
theta_start = np.array([KM.get(n, 0.5) for n in PARAM_NAMES])
post_std    = np.array([0.05]*N_PARAMS)
for i, n in enumerate(PARAM_NAMES):
    if n == 'phi_I1': post_std[i] = 1.0
    if n == 'phi_I2': post_std[i] = 2.0

# Sjekk startverdi
lp = log_posterior(theta_start, H, Sv, pre, post)
print(f"\nStartverdi log-posterior: {lp:.2f}")
if not np.isfinite(lp):
    print("ADVARSEL: Startverdi ugyldig. Justerer.")
    sys.exit(1)

# ── Kort diagnostisk kjøring ─────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  KORT DIAGNOSTISK MCMC: 2k burnin + 5k produksjon")
print(f"{'='*65}\n")

t0 = time.time()
chain, lp_vec, meta = adaptive_mcmc_with_monitoring(
    Y_pre=pre, Y_post=post, H=H, Sv=Sv,
    theta_init=theta_start, post_std_init=post_std,
    n_production=5000, burnin=2000, adapt_every=200,
    check_every=2500, max_recalib=1,
    psrf_thr=1.20, ess_pct_thr=0.005,  # Mer permissive for kort kjøring
    scale_init=0.5, seed=42, verbose=True,
    save_prefix="data/results/chain_fase2_diag",
)
t_total = time.time() - t0
print(f"\nDiagnostisk kjøring ferdig på {t_total/60:.1f} min")

# ── Diagnostikk ──────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  FASE 2 DIAGNOSTIKK")
print(f"{'='*65}")

psrf = compute_psrf(chain)
ess  = [compute_ess(chain[:, i]) for i in range(N_PARAMS)]
print(f"\n{'Parameter':<12} {'Mean':>9} {'Std':>8} {'ESS':>6} {'ESS/n':>7} {'PSRF':>6} {'K&M':>8}")
print("─" * 70)
for i, n in enumerate(PARAM_NAMES):
    s = chain[:, i]
    km = KM.get(n, float('nan'))
    flag = "!" if psrf[i] > 1.20 else " "
    print(f"{n:<12} {s.mean():>9.4f} {s.std():>8.4f} {ess[i]:>6.0f} "
          f"{ess[i]/len(chain):>7.4f} {psrf[i]:>5.3f}{flag} {km:>8.4f}")

# Korrelasjonsmatrise — vis sterkeste korrelasjoner
C = np.corrcoef(chain.T)
print(f"\nSterkeste posteriorkorrelasjoner:")
idx = np.argsort(np.abs(C).flatten())[::-1]
seen = set(); count = 0
for k in idx:
    i, j = k // N_PARAMS, k % N_PARAMS
    if i == j or (j, i) in seen: continue
    seen.add((i, j))
    if abs(C[i, j]) < 0.3 or count >= 8: break
    print(f"  {PARAM_NAMES[i]:10s} ↔ {PARAM_NAMES[j]:10s}: r={C[i,j]:+.3f}")
    count += 1

# Lagre
out = {
    'param_names': PARAM_NAMES,
    'mean': chain.mean(axis=0).tolist(),
    'std': chain.std(axis=0).tolist(),
    'ess': [float(e) for e in ess],
    'psrf': [float(p) for p in psrf],
    'correlation': C.tolist(),
    'n_samples': int(len(chain)),
    'time_min': float(t_total/60),
}
out_path = ROT / "data" / "results" / "fase2_diagnostic.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nLagret: {out_path}")
