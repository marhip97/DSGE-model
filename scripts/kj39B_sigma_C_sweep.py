"""
[STAT] kj39B analytisk sweep — sigma_C vs FEVD(I_R|E_C) og RMSE

Tester hypotesen: høy sigma_C (=0.107) er rotkårsaken til psi_R→prior-tak.
Med lavere sigma_C → konsumsjokk forklarer mindre av rentevariasjonen
→ psi_R kan identifiseres fra pengepolitikk-sjokk alene.

Analysen er rent analytisk (ingen MCMC). Holder alle andre parametere fast
ved kj38 posterior mean og varierer sigma_C ∈ [0.02, 0.15].

Output:
  - FEVD(I_R|E_C) per sigma_C-verdi
  - RMSE(16pt NB) per sigma_C
  - Anbefaling for kj39B-prior

Bruk:
  python scripts/kj39B_sigma_C_sweep.py
"""

from __future__ import annotations
import sys, json, warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nemo.estimation.mcmc import PARAM_NAMES
from nemo.model.equations import (
    build_matrices_v3_forward,
    Y, PI, I_R, RER, NZ, NE,
    E_A, E_C, E_H, E_O, E_Ys, E_rp, E_i, E_P,
)
from nemo.model.parameters import Parameters
from nemo.solver.blanchard_kahn import solve, compute_irf

Parameters.phi_PQ = 200.0
RESULTS = ROOT / "data" / "results"
SHOCK = 0.0025

NB_KORR = {
    "Y":   {"q1": -0.12, "q4": -0.47, "q8": -0.40, "q12": -0.25},
    "PI":  {"q1": -0.03, "q4": -0.14, "q8": -0.22, "q12": -0.22},
    "I_R": {"q1": +1.00, "q4": +0.55, "q8": +0.10, "q12": -0.15},
    "RER": {"q1": -1.50, "q4": -1.00, "q8": -0.50, "q12": -0.20},
}
VAR_IDX  = {"Y": Y, "PI": PI, "I_R": I_R, "RER": RER}
HORIZONS = {"q1": 0, "q4": 3, "q8": 7, "q12": 11}


def nb_rmse(irf: np.ndarray) -> float:
    errs = [(irf[HORIZONS[q], VAR_IDX[v]] - NB_KORR[v][q]) ** 2
            for v in NB_KORR for q in NB_KORR[v]]
    return float(np.sqrt(np.mean(errs)))


def compute_fevd_ir_konsum(T: np.ndarray, R: np.ndarray,
                            sigma_vec: np.ndarray) -> tuple[float, float]:
    """Returner (andel_konsum, andel_pengepol) i I_R FEVD ved q20."""
    contrib = np.zeros(NE)
    for e in range(NE):
        if sigma_vec[e] == 0:
            continue
        irf_e = compute_irf(T, R, e, sigma_vec[e], T_periods=20)
        contrib[e] = np.sum(irf_e[:, I_R] ** 2)
    total = contrib.sum()
    if total == 0:
        return 0.0, 0.0
    return contrib[E_C] / total * 100, contrib[E_i] / total * 100


def main() -> None:
    pf = RESULTS / "chain_kj38_prod_posterior.json"
    with open(pf) as f:
        post = json.load(f)["summary"]

    theta = np.array([post[n]["mean"] if n in post else 0.5 for n in PARAM_NAMES])
    p_base = Parameters()
    for i, n in enumerate(PARAM_NAMES):
        if hasattr(p_base, n):
            setattr(p_base, n, float(theta[i]))
    p_base.lambda_pi4 = 0.0

    # Basis sigma-vektor (kj38 posterior)
    sigma_base = np.zeros(NE)
    sigma_base[E_A]  = 0.006
    sigma_base[E_C]  = float(post["sigma_C"]["mean"])
    sigma_base[E_H]  = float(post["sigma_H"]["mean"])
    sigma_base[E_O]  = float(post["sigma_O"]["mean"])
    sigma_base[E_Ys] = float(post["sigma_Ys"]["mean"])
    sigma_base[E_rp] = 0.006
    sigma_base[E_i]  = float(post["sigma_i"]["mean"])
    sigma_base[E_P]  = float(post["sigma_P"]["mean"])

    print(f"\nkj38 posterior mean: sigma_C={sigma_base[E_C]:.4f}")
    print(f"{'='*72}")
    print(f"{'sigma_C':>9s} | {'FEVD I_R(E_C)':>14s} | {'FEVD I_R(E_i)':>14s} | "
          f"{'I_R.q12':>8s} | {'RMSE':>8s}")
    print(f"{'-'*72}")

    sigma_C_vals = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.107, 0.12, 0.15]

    for sigma_C in sigma_C_vals:
        p = Parameters()
        for i, n in enumerate(PARAM_NAMES):
            if hasattr(p, n):
                setattr(p, n, float(theta[i]))
        p.lambda_pi4 = 0.0

        try:
            G0, G1, Psi, Pi = build_matrices_v3_forward(p, lambda_pi4=0.0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                T, R, diag = solve(G0, G1, Psi, Pi, verbose=False)
            if not diag.get("stable", False):
                print(f"  {sigma_C:>9.3f} | BK ustabil")
                continue
        except Exception as e:
            print(f"  {sigma_C:>9.3f} | Feil: {e}")
            continue

        sigma_vec = sigma_base.copy()
        sigma_vec[E_C] = sigma_C

        fC, fI = compute_fevd_ir_konsum(T, R, sigma_vec)

        # IRF for NB-score
        irf_raw = compute_irf(T, R, E_i, SHOCK, T_periods=20)
        peak = float(np.max(irf_raw[:, I_R]))
        if peak <= 0:
            print(f"  {sigma_C:>9.3f} | IRF-peak ≤ 0")
            continue
        irf_norm = irf_raw / peak
        rmse = nb_rmse(irf_norm)
        ir_q12 = irf_norm[11, I_R]

        flag = "← basis" if abs(sigma_C - sigma_base[E_C]) < 0.001 else ""
        warn = " ⚠️" if fC > 70 else (" ✅" if fC < 30 else "")
        print(f"  {sigma_C:>9.3f} | {fC:>13.1f}%{warn} | {fI:>13.1f}% | "
              f"{ir_q12:>+8.3f} | {rmse:>8.4f} {flag}")

    print(f"\n  Konklusjon:")
    print(f"  - Hvis FEVD(I_R|E_C) < 50% ved sigma_C ≤ 0.05: kj39B er gjennomførbar")
    print(f"  - Merk: I_R.q12 endres ikke med sigma_C — det er en IRF-egenskap, ikke FEVD")
    print(f"  - psi_R må fortsatt dogmatiseres for å endre I_R-dynamikk")


if __name__ == "__main__":
    main()
