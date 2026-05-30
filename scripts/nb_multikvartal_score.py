"""
[STAT] Multi-kvartal NB-benchmark score — kj31, kj32, ...

Beregner avvik mellom NEMO v3 IRF og NB Memo 3/2024 Figur 1 referanse
ved q1, q4, q8, q12 for Y, PI, I_R og RER (16 referansepunkter).

Output:
  - Tabell: modell vs. NB per variabel/horisont
  - RMSE og MAD over alle 16 punkter
  - psi_R-sweep: RMSE som funksjon av psi_R [0.60 → 0.999]

Bruk:
  python scripts/nb_multikvartal_score.py [posterior_json] [run_label]

Eksempel:
  python scripts/nb_multikvartal_score.py \
      data/results/chain_kj31_prod_posterior.json kj31
"""

from __future__ import annotations
import sys
import json
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nemo.estimation.mcmc import PARAM_NAMES
from nemo.model.equations import build_matrices_v3, Y, PI, I_R, RER, E_i
from nemo.model.parameters import Parameters
from nemo.solver.blanchard_kahn import solve, compute_irf

# NB Memo 3/2024 Figur 1 — korrigerte avlesninger (2026-05-30)
# Normalisering: styringsrente peak = +1 ppt annualisert (consistent med NB-tekst).
# Korreksjon vs. originale verdier: RER var ~3× for liten; I_R q12 endret fortegn
# (NB-modellen underskyter); PI q8/q12 mer persistent enn antatt.
# Y q1 justert ned (NB-figur viser liten impact-respons ved x=1).
NB_FIGUR1 = {
    "Y":   {"q1": -0.12, "q4": -0.47, "q8": -0.40, "q12": -0.25},
    "PI":  {"q1": -0.03, "q4": -0.14, "q8": -0.22, "q12": -0.22},
    "I_R": {"q1": +1.00, "q4": +0.55, "q8": +0.10, "q12": -0.15},
    "RER": {"q1": -1.50, "q4": -1.00, "q8": -0.50, "q12": -0.20},
}

VAR_IDX  = {"Y": Y,  "PI": PI,  "I_R": I_R,  "RER": RER}
HORIZONS = {"q1": 0, "q4": 3,   "q8": 7,     "q12": 11}
SHOCK    = 0.0025   # 25 bp → normalisert mot peak I_R=1


def lag_irf_normalisert(theta: np.ndarray) -> np.ndarray | None:
    """Returnerer (20×NZ) normalisert IRF, eller None ved BK-feil."""
    p = Parameters()
    for i, n in enumerate(PARAM_NAMES):
        if hasattr(p, n):
            setattr(p, n, float(theta[i]))
    try:
        G0, G1, Psi, Pi = build_matrices_v3(p)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            T, R, diag = solve(G0, G1, Psi, Pi, verbose=False)
        if not diag.get("stable", False):
            return None
        irf_raw = compute_irf(T, R, E_i, SHOCK, T_periods=20)
        peak = float(np.max(irf_raw[:, I_R]))
        if peak <= 0:
            return None
        return irf_raw / peak
    except Exception:
        return None


def nb_score(irf: np.ndarray) -> dict:
    """Beregner multi-kvartal NB-score for en normalisert IRF."""
    errs = {}
    for vname, vidx in VAR_IDX.items():
        errs[vname] = {}
        for qname, qidx in HORIZONS.items():
            modell = float(irf[qidx, vidx])
            nb     = NB_FIGUR1[vname][qname]
            errs[vname][qname] = {"modell": modell, "nb": nb, "err": modell - nb}

    alle_err = [errs[v][q]["err"] for v in errs for q in errs[v]]
    return {
        "per_var": errs,
        "rmse":    float(np.sqrt(np.mean(np.array(alle_err) ** 2))),
        "mad":     float(np.mean(np.abs(alle_err))),
    }


def print_rapport(score: dict, label: str = "") -> None:
    print(f"\n=== NB multi-kvartal benchmark{': ' + label if label else ''} ===")
    print(f"{'Var':5s} {'H':5s} {'Modell':>9s} {'NB':>7s} {'Err':>8s}")
    print("-" * 40)
    for vname in ["Y", "PI", "I_R", "RER"]:
        for qname in ["q1", "q4", "q8", "q12"]:
            d = score["per_var"][vname][qname]
            flag = "✅" if abs(d["err"]) < 0.15 else ("⚠️ " if abs(d["err"]) < 0.30 else "❌")
            print(f"{vname:5s} {qname:5s} {d['modell']:>+9.4f} {d['nb']:>+7.3f} {d['err']:>+8.4f} {flag}")
    print("-" * 40)
    print(f"RMSE(16 pt) = {score['rmse']:.4f}   MAD = {score['mad']:.4f}")


def psi_R_sweep(theta_base: np.ndarray, psi_R_vals: list[float]) -> None:
    psi_R_idx = PARAM_NAMES.index("psi_R")
    print(f"\n=== psi_R sweep (andre param: posterior mean) ===")
    print(f"{'psi_R':>8} | {'I_R(q4)':>8} | {'I_R(q8)':>8} | {'RER(q1)':>8} | {'RMSE':>8}")
    print("-" * 55)
    for psi_R in psi_R_vals:
        theta = theta_base.copy()
        theta[psi_R_idx] = psi_R
        irf = lag_irf_normalisert(theta)
        if irf is None:
            print(f"{psi_R:>8.3f} | BK-ustabil")
            continue
        sc = nb_score(irf)
        print(
            f"{psi_R:>8.3f} | {irf[3,I_R]:>+8.4f} | {irf[7,I_R]:>+8.4f} | "
            f"{irf[0,RER]:>+8.4f} | {sc['rmse']:>8.4f}"
        )


def main() -> None:
    rot = Path(__file__).parent.parent

    # Velg posterior
    if len(sys.argv) >= 2:
        posterior_sti = Path(sys.argv[1])
    else:
        # Siste kj*-posterior
        for kand in ["kj32", "kj31", "kj30"]:
            p = rot / f"data/results/chain_{kand}_prod_posterior.json"
            if p.exists():
                posterior_sti = p
                break
        else:
            print("Ingen posterior funnet.", file=sys.stderr)
            sys.exit(1)

    label = sys.argv[2] if len(sys.argv) >= 3 else posterior_sti.stem.replace("chain_", "").replace("_prod_posterior", "")

    with open(posterior_sti) as f:
        post = json.load(f)["summary"]

    theta = np.array([post[n]["mean"] if n in post else 0.5 for n in PARAM_NAMES])

    irf = lag_irf_normalisert(theta)
    if irf is None:
        print("BK-ustabilt ved posterior mean.", file=sys.stderr)
        sys.exit(1)

    score = nb_score(irf)
    print_rapport(score, label)

    # B5
    by4  = irf[3, Y]   / (-0.450)
    bpi4 = irf[3, PI]  / (-0.150)
    print(f"\nB5: by4={by4:.4f}  bpi4={bpi4:.4f}  BESTÅTT={0.8 <= by4 <= 1.5 and bpi4 >= 0.35}")

    # psi_R sweep
    psi_R_sweep(
        theta,
        [0.666, 0.75, 0.80, 0.84, 0.86, 0.88, 0.90, 0.92, 0.95, 0.989, 0.999],
    )

    # Returner score for bruk i andre scripts
    return score


if __name__ == "__main__":
    main()
