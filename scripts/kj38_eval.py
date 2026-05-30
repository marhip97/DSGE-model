"""
[STAT] kj38 evaluering — build_matrices_v3_forward + phi_PQ=200

Beregner IRF ved posterior mean fra kj38, sammenligner mot NB-benchmark
(16 referansepunkter), og rapporterer RMSE, MAD og B5-kriterier.

Bruk:
  python scripts/kj38_eval.py [posterior_json]

Eksempel:
  python scripts/kj38_eval.py data/results/chain_kj38_prod_posterior.json
"""

from __future__ import annotations
import sys
import json
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nemo.estimation.mcmc import PARAM_NAMES
from nemo.model.equations import build_matrices_v3_forward, Y, PI, I_R, RER, E_i
from nemo.model.parameters import Parameters
from nemo.solver.blanchard_kahn import solve, compute_irf

Parameters.phi_PQ = 200.0
LAMBDA_PI4 = 0.0

NB_KORR = {
    "Y":   {"q1": -0.12, "q4": -0.47, "q8": -0.40, "q12": -0.25},
    "PI":  {"q1": -0.03, "q4": -0.14, "q8": -0.22, "q12": -0.22},
    "I_R": {"q1": +1.00, "q4": +0.55, "q8": +0.10, "q12": -0.15},
    "RER": {"q1": -1.50, "q4": -1.00, "q8": -0.50, "q12": -0.20},
}
VAR_IDX  = {"Y": Y, "PI": PI, "I_R": I_R, "RER": RER}
HORIZONS = {"q1": 0, "q4": 3, "q8": 7, "q12": 11}
SHOCK    = 0.0025


def _build(p):
    p.lambda_pi4 = LAMBDA_PI4
    return build_matrices_v3_forward(p, theta_H=0.05, lambda_pi4=LAMBDA_PI4)


def lag_irf_normalisert(theta: np.ndarray) -> np.ndarray | None:
    p = Parameters()
    for i, n in enumerate(PARAM_NAMES):
        if hasattr(p, n):
            setattr(p, n, float(theta[i]))
    try:
        G0, G1, Psi, Pi = _build(p)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            T, R, diag = solve(G0, G1, Psi, Pi, verbose=False)
        if not diag.get("stable", False):
            return None
        irf_raw = compute_irf(T, R, E_i, SHOCK, T_periods=20)
        peak = float(np.max(irf_raw[:, I_R]))
        return None if peak <= 0 else irf_raw / peak
    except Exception:
        return None


def nb_score(irf: np.ndarray) -> dict:
    errs = {}
    for vname, vidx in VAR_IDX.items():
        errs[vname] = {}
        for qname, qidx in HORIZONS.items():
            modell = float(irf[qidx, vidx])
            nb     = NB_KORR[vname][qname]
            errs[vname][qname] = {"modell": modell, "nb": nb, "err": modell - nb}
    alle_err = [errs[v][q]["err"] for v in errs for q in errs[v]]
    return {
        "per_var": errs,
        "rmse":    float(np.sqrt(np.mean(np.array(alle_err) ** 2))),
        "mad":     float(np.mean(np.abs(alle_err))),
    }


def print_rapport(score: dict, label: str, post: dict) -> None:
    print(f"\n=== kj38 NB-benchmark ({label}) — phi_PQ=200, forward Taylor (λ=0) ===")
    print(f"{'Var':5s} {'H':5s} {'Modell':>9s} {'NB':>7s} {'Err':>8s}")
    print("-" * 42)
    for vname in ["Y", "PI", "I_R", "RER"]:
        for qname in ["q1", "q4", "q8", "q12"]:
            d = score["per_var"][vname][qname]
            flag = "✅" if abs(d["err"]) < 0.15 else ("⚠️ " if abs(d["err"]) < 0.30 else "❌")
            print(f"{vname:5s} {qname:5s} {d['modell']:>+9.4f} {d['nb']:>+7.3f} {d['err']:>+8.4f} {flag}")
    print("-" * 42)
    print(f"RMSE(16 pt) = {score['rmse']:.4f}   MAD = {score['mad']:.4f}")

    irf_vals = score["per_var"]
    by4  = abs(irf_vals["Y"]["q4"]["modell"])   / 0.47
    bpi4 = abs(irf_vals["PI"]["q4"]["modell"])  / 0.14
    b5ok = 0.80 <= by4 <= 1.50 and bpi4 >= 0.35
    print(f"\nB5: by4={by4:.3f} {'✅' if 0.80<=by4<=1.50 else '❌'}  "
          f"bpi4={bpi4:.3f} {'✅' if bpi4>=0.35 else '❌'}  "
          f"BESTÅTT={'✅' if b5ok else '❌'}")

    print(f"\nPosterior mean (nøkkelparametere):")
    for n in ["psi_R", "psi_P1", "rho_s", "gamma_p", "rho_A", "rho_C", "rho_O", "rho_Ys", "rho_rp", "rho_H"]:
        if n in post:
            print(f"  {n:12s} = {post[n]['mean']:.4f}  (std={post[n]['std']:.4f})")


def main() -> None:
    if len(sys.argv) >= 2:
        pf = Path(sys.argv[1])
    else:
        pf = ROOT / "data/results/chain_kj38_prod_posterior.json"

    if not pf.exists():
        print(f"Finner ikke {pf}", file=sys.stderr)
        sys.exit(1)

    with open(pf) as f:
        data = json.load(f)
    post = data["summary"]
    label = sys.argv[2] if len(sys.argv) >= 3 else "kj38"

    theta = np.array([post[n]["mean"] if n in post else 0.5 for n in PARAM_NAMES])

    print(f"Beregner IRF med build_matrices_v3_forward (phi_PQ=200, λ_pi4=0.0)...")
    irf = lag_irf_normalisert(theta)
    if irf is None:
        print("BK-ustabilt ved posterior mean.", file=sys.stderr)
        sys.exit(1)

    score = nb_score(irf)
    print_rapport(score, label, post)


if __name__ == "__main__":
    main()
