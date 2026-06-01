"""
[STAT] kj39 FEVD-diagnostikk — kj38 vs kj35 sjokkbidrag

Beregner Forecast Error Variance Decomposition (FEVD) ved posterior mean
for kj38 og kj35. Avdekker hvilke sjokk som identifiserer psi_R.

Nøkkelfunn (bekreftet 2026-05-30):
  - I_R-variasjon forklares 87.5% av konsumsjokk (E_C), bare 11.3% av pengepolitikk
  - PI-variasjon forklares 0% av pengepolitikk — markupsjokk dominerer (44%)
  - psi_R er identifisert via konsumsjokk-respons, ikke pengepolitikk-sjokket
  - Dette er rotkårsaken til psi_R→prior-tak i alle kj35–kj38

Bruk:
  python scripts/kj39_fevd_diag.py
"""

from __future__ import annotations
import sys, json, warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nemo.estimation.mcmc import PARAM_NAMES
from nemo.model.equations import (
    build_matrices_v3, build_matrices_v3_forward,
    Y, PI, I_R, RER, NZ, NE,
    E_A, E_C, E_H, E_O, E_Ys, E_rp, E_i, E_P,
)
from nemo.model.parameters import Parameters
from nemo.solver.blanchard_kahn import solve, compute_irf

RESULTS = ROOT / "data" / "results"

SHOCK_LABELS = {
    E_A: "TFP", E_C: "Konsum", E_H: "Bolig", 3: "Off.forbruk",
    E_O: "Oljepris", E_Ys: "Utenl.ettersp", E_rp: "Risikopremie",
    E_i: "Pengepolitikk", E_P: "Prismarkup",
    9: "LTV hush.", 10: "Pengepremie", 11: "Inv.just.kost", 12: "Utenl.inflasjon",
}
VARS = {"Y": Y, "PI": PI, "I_R": I_R, "RER": RER}
H = 20


def build_sigma_vec(post: dict) -> np.ndarray:
    vec = np.zeros(NE)
    vec[E_A]   = 0.006
    vec[E_C]   = float(post.get("sigma_C", {}).get("mean", 0.107))
    vec[E_H]   = float(post.get("sigma_H", {}).get("mean", 0.341))
    vec[E_O]   = float(post.get("sigma_O", {}).get("mean", 0.151))
    vec[E_Ys]  = float(post.get("sigma_Ys", {}).get("mean", 0.017))
    vec[E_rp]  = 0.006
    vec[E_i]   = float(post.get("sigma_i", {}).get("mean", 0.0006))
    vec[E_P]   = float(post.get("sigma_P", {}).get("mean", 0.007))
    return vec


def compute_fevd_q20(posterior_path: Path, phi_pq: float = 669.0,
                     use_forward: bool = False) -> dict | None:
    """Beregn FEVD ved q20 for en posterior-fil. Returner andeler per sjokk per variabel."""
    with open(posterior_path) as f:
        post = json.load(f)["summary"]

    Parameters.phi_PQ = phi_pq
    theta = np.array([post[n]["mean"] if n in post else 0.5 for n in PARAM_NAMES])
    p = Parameters()
    for i, n in enumerate(PARAM_NAMES):
        if hasattr(p, n):
            setattr(p, n, float(theta[i]))

    try:
        if use_forward:
            p.lambda_pi4 = 0.0
            G0, G1, Psi, Pi = build_matrices_v3_forward(p, lambda_pi4=0.0)
        else:
            G0, G1, Psi, Pi = build_matrices_v3(p)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            T, R, diag = solve(G0, G1, Psi, Pi, verbose=False)
        if not diag.get("stable", False):
            print(f"  BK ustabil for {posterior_path.name}")
            return None
    except Exception as e:
        print(f"  Feil ved bygging: {e}")
        return None

    sigma_vec = build_sigma_vec(post)
    contrib = np.zeros((len(VARS), NE))

    for e in range(NE):
        if sigma_vec[e] == 0:
            continue
        irf_e = compute_irf(T, R, e, sigma_vec[e], T_periods=H)
        for vi, (vname, vidx) in enumerate(VARS.items()):
            contrib[vi, e] = np.sum(irf_e[:, vidx] ** 2)

    fevd = {}
    for vi, vname in enumerate(VARS.keys()):
        total = contrib[vi].sum()
        if total == 0:
            continue
        fevd[vname] = {
            SHOCK_LABELS.get(e, f"sjokk{e}"): contrib[vi, e] / total * 100
            for e in range(NE)
            if contrib[vi, e] / total * 100 > 0.5
        }
    return fevd, {n: float(post[n]["mean"]) for n in ["psi_R", "sigma_C", "rho_H"] if n in post}


def print_fevd_tabell(fevd: dict, params: dict, label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  FEVD ved q20 — {label}")
    print(f"  psi_R={params.get('psi_R', '?'):.4f}  sigma_C={params.get('sigma_C', '?'):.4f}  rho_H={params.get('rho_H', '?'):.4f}")
    print(f"{'='*60}")
    for vname in ["Y", "PI", "I_R", "RER"]:
        if vname not in fevd:
            continue
        print(f"\n  {vname}:")
        sorted_shocks = sorted(fevd[vname].items(), key=lambda x: -x[1])
        for sname, pct in sorted_shocks:
            bar = "█" * int(pct / 5)
            flag = " ⚠️" if vname == "I_R" and sname == "Konsum" and pct > 50 else ""
            flag = " ✅" if vname == "I_R" and sname == "Pengepolitikk" and pct > 30 else flag
            print(f"    {sname:22s}: {pct:5.1f}% {bar}{flag}")


def main() -> None:
    runs = [
        ("kj35", RESULTS / "chain_kj35_prod_posterior.json", 669.0, False),
        ("kj38", RESULTS / "chain_kj38_prod_posterior.json", 200.0, True),
    ]

    resultater = {}
    for label, path, phi_pq, forward in runs:
        if not path.exists():
            print(f"Finner ikke {path.name} — hopper over {label}")
            continue
        print(f"\nBeregner FEVD for {label} (phi_PQ={phi_pq}, forward={forward})...")
        res = compute_fevd_q20(path, phi_pq=phi_pq, use_forward=forward)
        if res is None:
            continue
        fevd, params = res
        print_fevd_tabell(fevd, params, label)
        resultater[label] = fevd

    if len(resultater) == 2:
        print(f"\n{'='*60}")
        print("  SAMMENLIGNING: I_R-sjokkandeler kj35 vs kj38")
        print(f"{'='*60}")
        for sname in ["Konsum", "Pengepolitikk", "Bolig", "Oljepris", "Risikopremie"]:
            v35 = resultater["kj35"]["I_R"].get(sname, 0.0)
            v38 = resultater["kj38"]["I_R"].get(sname, 0.0)
            delta = v38 - v35
            arrow = "↑" if delta > 0 else "↓"
            print(f"  {sname:22s}: kj35={v35:5.1f}%  kj38={v38:5.1f}%  {arrow}{abs(delta):.1f}pp")

        print(f"\n  Diagnose:")
        ir_konsum_38 = resultater["kj38"]["I_R"].get("Konsum", 0)
        ir_peng_38   = resultater["kj38"]["I_R"].get("Pengepolitikk", 0)
        if ir_konsum_38 > 70:
            print(f"  ❌ Konsumsjokk dominerer I_R ({ir_konsum_38:.1f}%) — psi_R identifisert feil")
            print(f"     Løsning: dogmatisk prior på psi_R (kj39A) eller sigma_C shrinkage (kj39B)")
        if ir_peng_38 < 20:
            print(f"  ❌ Pengepolitikksjokk forklarer kun {ir_peng_38:.1f}% av I_R-variasjon")


if __name__ == "__main__":
    main()
