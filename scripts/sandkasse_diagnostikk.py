"""
[STAT] Sandkasse — Full modellprestasjon mot NB (Fase 0.75)

Diagnostisk verktøy for systematisk å identifisere hvilke parametre og
modellmekanismer som forklarer modellens persistensunderskudd i q8–q12
for Y, PI, I_R og RER vs NB Memo 3/2024 Figur 1.

Bruk:
  python scripts/sandkasse_diagnostikk.py [posterior_json]

Eksempel:
  python scripts/sandkasse_diagnostikk.py data/results/chain_kj33_tail_posterior.json
  python scripts/sandkasse_diagnostikk.py data/results/chain_kj34_prod_posterior.json
"""

from __future__ import annotations
import sys
import json
import warnings
from pathlib import Path
from copy import deepcopy

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nemo.estimation.mcmc import PARAM_NAMES
from nemo.model.equations import build_matrices_v3, Y, PI, I_R, RER, E_i, SHOCK_NAMES
from nemo.model.parameters import Parameters
from nemo.solver.blanchard_kahn import solve, compute_irf

# NB Memo 3/2024 Figur 1
NB_FIGUR1 = {
    "Y":   {"q1": -0.20, "q4": -0.45, "q8": -0.35, "q12": -0.15},
    "PI":  {"q1": -0.05, "q4": -0.15, "q8": -0.20, "q12": -0.10},
    "I_R": {"q1": +1.00, "q4": +0.60, "q8": +0.20, "q12": +0.05},
    "RER": {"q1": -0.50, "q4": -0.40, "q8": -0.20, "q12": -0.05},
}
VAR_IDX  = {"Y": Y,  "PI": PI,  "I_R": I_R,  "RER": RER}
HORIZONS = {"q1": 0, "q4": 3,   "q8": 7,     "q12": 11}
SHOCK    = 0.0025


def _irf(p: Parameters) -> np.ndarray | None:
    """Normalisert IRF (20×NZ), eller None ved feil."""
    try:
        G0, G1, Psi, Pi = build_matrices_v3(p)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            T, R, diag = solve(G0, G1, Psi, Pi, verbose=False)
        if not diag.get("stable", False):
            return None
        raw = compute_irf(T, R, E_i, SHOCK, T_periods=20)
        peak = float(np.max(raw[:, I_R]))
        return raw / peak if peak > 0 else None
    except Exception:
        return None


def rmse_16pt(irf: np.ndarray) -> float:
    sq = []
    for vname, vidx in VAR_IDX.items():
        for qname, h in HORIZONS.items():
            sq.append((irf[h, vidx] - NB_FIGUR1[vname][qname]) ** 2)
    return float(np.sqrt(np.mean(sq)))


def rmse_per_var(irf: np.ndarray) -> dict[str, float]:
    out = {}
    for vname, vidx in VAR_IDX.items():
        sq = [(irf[h, vidx] - NB_FIGUR1[vname][qname]) ** 2
              for qname, h in HORIZONS.items()]
        out[vname] = float(np.sqrt(np.mean(sq)))
    return out


def theta_til_params(theta: np.ndarray) -> Parameters:
    p = Parameters()
    for i, n in enumerate(PARAM_NAMES):
        if hasattr(p, n):
            setattr(p, n, float(theta[i]))
    return p


def last_posterior(path: str) -> tuple[np.ndarray, str]:
    """Laster posterior fra JSON-fil eller .npy chain (tail [55k:74k])."""
    fp = Path(path)
    if fp.suffix == '.npy':
        chain = np.load(fp)
        tail = chain[max(0, len(chain) - 19000):]
        return tail.mean(0), fp.name
    with open(fp) as f:
        d = json.load(f)
    key = 'posterior_mean' if 'posterior_mean' in d else 'summary'
    data = d[key]
    theta = np.zeros(len(PARAM_NAMES))
    for i, n in enumerate(PARAM_NAMES):
        if n in data:
            v = data[n]
            theta[i] = float(v['mean']) if isinstance(v, dict) else float(v)
        else:
            p0 = Parameters()
            theta[i] = getattr(p0, n, 0.5)
    return theta, fp.name


# ─────────────────────────────────────────────────────────────────────────────
# Spor A — Datagrunnlag
# ─────────────────────────────────────────────────────────────────────────────

def spor_a_data(theta: np.ndarray) -> None:
    """A1: KPI total vs KPI-JAE — IRF er modellens egenskap, uavhengig av data.
    Sammenligner IRF ved posterior fra de to datasettene hvis begge finnes."""
    print("\n" + "=" * 65)
    print("SPOR A — DATAGRUNNLAG")
    print("=" * 65)

    rot = Path(__file__).parent.parent
    posteriors = {
        "kj33-tail (KPI-JAE)": rot / "data/results/chain_kj33_tail_posterior.json",
        "kj31 (KPI-JAE)":      rot / "data/results/chain_kj31_prod_posterior.json",
        "kj34 (KPI-JAE)":      rot / "data/results/chain_kj34_prod_posterior.json",
    }

    print(f"\n{'Posterior':<30} {'RMSE(16pt)':>10} {'Y.q4':>7} {'PI.q4':>7} {'I_R.q4':>8} {'RER.q4':>8}")
    print("-" * 75)
    for label, fp in posteriors.items():
        if not fp.exists():
            print(f"  {label:<28} — mangler")
            continue
        t, _ = last_posterior(str(fp))
        p = theta_til_params(t)
        irf = _irf(p)
        if irf is None:
            print(f"  {label:<28} BK-ustabil")
            continue
        rm = rmse_16pt(irf)
        print(f"  {label:<28} {rm:>10.4f} {irf[3,Y]:>7.3f} {irf[3,PI]:>7.3f} {irf[3,I_R]:>8.3f} {irf[3,RER]:>8.3f}")
    print()
    print("NB-referanse (normalisert):     —       -0.450  -0.150   +0.600  -0.400")


# ─────────────────────────────────────────────────────────────────────────────
# Spor B — Modellstruktur: parametersweeps
# ─────────────────────────────────────────────────────────────────────────────

def _sweep(theta_base: np.ndarray, param_name: str, values: list[float],
           param_is_pyattr: bool = True) -> None:
    """Generell parametersweep. param_is_pyattr=True: sett p.<param_name> direkte."""
    psi_R_idx = PARAM_NAMES.index("psi_R") if "psi_R" in PARAM_NAMES else -1

    print(f"\n  {param_name} sweep:")
    print(f"  {'Verdi':>8} {'RMSE(16pt)':>10} {'PI.q4':>7} {'PI.q8':>7} "
          f"{'Y.q4':>7} {'I_R.q12':>8} {'RER.q12':>8} {'stabil':>7}")
    print("  " + "-" * 68)
    for v in values:
        theta = theta_base.copy()
        if param_name in PARAM_NAMES:
            theta[PARAM_NAMES.index(param_name)] = v
        p = theta_til_params(theta)
        if param_is_pyattr and param_name not in PARAM_NAMES:
            setattr(p, param_name, v)
        irf = _irf(p)
        if irf is None:
            print(f"  {v:>8.3f} {'BK-ustabil':>10}")
            continue
        rm = rmse_16pt(irf)
        print(f"  {v:>8.3f} {rm:>10.4f} {irf[3,PI]:>7.3f} {irf[7,PI]:>7.3f} "
              f"{irf[3,Y]:>7.3f} {irf[11,I_R]:>8.3f} {irf[11,RER]:>8.3f}    OK")


def spor_b_modellstruktur(theta: np.ndarray) -> None:
    print("\n" + "=" * 65)
    print("SPOR B — MODELLSTRUKTUR (parametersweeps)")
    print("=" * 65)
    print(f"\nBasispunkt — kj33/kj34 posterior:")
    p0 = theta_til_params(theta)
    irf0 = _irf(p0)
    if irf0 is not None:
        rm0 = rmse_16pt(irf0)
        rv0 = rmse_per_var(irf0)
        print(f"  RMSE(16pt)={rm0:.4f}  Y={rv0['Y']:.3f}  PI={rv0['PI']:.3f}  "
              f"I_R={rv0['I_R']:.3f}  RER={rv0['RER']:.3f}")
        print(f"  Parametre: gamma_p={getattr(p0,'gamma_p',0.0):.3f}  "
              f"kappa_M={p0.kappa_M:.3f}  h_c={p0.h_c:.3f}  "
              f"phi_PQ={p0.phi_PQ:.0f}")

    # B1a: gamma_p (inflasjonspersistens)
    _sweep(theta, "gamma_p",
           [0.00, 0.13, 0.30, 0.50, 0.65, 0.80],
           param_is_pyattr=True)

    # B1b: kappa_M (importprispass-through)
    _sweep(theta, "kappa_M",
           [0.03, 0.06, 0.10, 0.15, 0.20, 0.25],
           param_is_pyattr=True)

    # B1c: h_c (habit persistence konsum)
    _sweep(theta, "h_c",
           [0.70, 0.80, 0.85, 0.90, 0.93, 0.96],
           param_is_pyattr=True)

    # B1d: rho_s (AR-glatting av RER i UIP)
    _sweep(theta, "rho_s", [0.0, 0.05, 0.15, 0.30, 0.50, 0.70])

    # B1e: phi_PQ (Rotemberg-kostnad → kappa_P)
    print(f"\n  phi_PQ sweep (påvirker kappa_P = 6×5/phi_PQ):")
    print(f"  {'phi_PQ':>8} {'kappa_P':>8} {'RMSE(16pt)':>10} {'PI.q4':>7} {'PI.q8':>7} "
          f"{'Y.q4':>7} {'stabil':>7}")
    print("  " + "-" * 60)
    for phi_pq in [200, 350, 500, 669, 900]:
        t2 = theta.copy()
        p2 = theta_til_params(t2)
        p2.phi_PQ = phi_pq
        irf2 = _irf(p2)
        kp = 6 * 5 / phi_pq
        if irf2 is None:
            print(f"  {phi_pq:>8.0f} {kp:>8.4f} {'BK-ustabil':>10}")
            continue
        rm2 = rmse_16pt(irf2)
        print(f"  {phi_pq:>8.0f} {kp:>8.4f} {rm2:>10.4f} {irf2[3,PI]:>7.3f} "
              f"{irf2[7,PI]:>7.3f} {irf2[3,Y]:>7.3f}    OK")

    # B1f: sigma_rp diagnose — risikopremie-sjokk påvirker RER kanal
    _sweep(theta, "rho_rp", [0.30, 0.50, 0.62, 0.74, 0.85, 0.95])

    # B1g: psi_R (bekrefter opt) — NB: psi_R er i PARAM_NAMES
    _sweep(theta, "psi_R", [0.70, 0.80, 0.88, 0.90, 0.95, 0.99])


# ─────────────────────────────────────────────────────────────────────────────
# Spor C — Estimering: FEVD + Taylor-sweep
# ─────────────────────────────────────────────────────────────────────────────

def spor_c_estimering(theta: np.ndarray) -> None:
    print("\n" + "=" * 65)
    print("SPOR C — ESTIMERING")
    print("=" * 65)

    p0 = theta_til_params(theta)
    try:
        G0, G1, Psi, Pi = build_matrices_v3(p0)
        T, R, diag = solve(G0, G1, Psi, Pi, verbose=False)
        if not diag.get("stable", False):
            print("  BK-ustabil ved basisparametere")
            return
    except Exception as e:
        print(f"  Feil: {e}")
        return

    # C1: FEVD — varians-dekomposisjon for Y, PI, I_R, RER
    print(f"\nC1 — FEVD (20 kvartaler, andelsbidrag per sjokk):")
    NZ_T, NE = T.shape[0], R.shape[1]
    IRFs = np.zeros((20, NZ_T, NE))
    for e in range(NE):
        ei = np.zeros(NE); ei[e] = SHOCK
        IRFs[0, :, e] = R @ ei
        for t in range(1, 20):
            IRFs[t, :, e] = T @ IRFs[t-1, :, e]

    var_total = {v: 0.0 for v in VAR_IDX}
    var_by_shock = {v: np.zeros(NE) for v in VAR_IDX}
    for vname, vidx in VAR_IDX.items():
        for e in range(NE):
            contrib = float(np.sum(IRFs[:, vidx, e] ** 2))
            var_by_shock[vname][e] = contrib
            var_total[vname] += contrib

    snames = SHOCK_NAMES[:NE]
    header = f"  {'Sjokk':<22}" + "".join(f" {v:>7}" for v in VAR_IDX)
    print(header)
    print("  " + "-" * (22 + 7 * len(VAR_IDX)))
    for e, sn in enumerate(snames):
        row = f"  {sn:<22}"
        for vname in VAR_IDX:
            pct = 100 * var_by_shock[vname][e] / max(var_total[vname], 1e-12)
            row += f" {pct:>6.1f}%"
        print(row)

    # C2: sigma_rp sweep (risikopremie-sjokk volatilitet)
    print(f"\nC2 — sigma_rp sweep:")
    _sweep(theta, "rho_rp", [0.40, 0.55, 0.62, 0.74, 0.85])

    # C4: Taylor-regel koeffisienter
    print(f"\nC4a — psi_P1 sweep (Taylor inflasjonssensitivitet):")
    _sweep(theta, "psi_P1", [0.10, 0.20, 0.29, 0.40, 0.60, 0.80])

    print(f"\nC4b — psi_Y sweep (Taylor outputgap-sensitivitet):")
    _sweep(theta, "psi_Y", [0.10, 0.20, 0.24, 0.35, 0.50, 0.70])


# ─────────────────────────────────────────────────────────────────────────────
# Kombinert analyse: beste kombinasjon
# ─────────────────────────────────────────────────────────────────────────────

def beste_kombinasjon(theta: np.ndarray) -> None:
    """Tester kombinasjoner av beste enkeltparametre fra sweep."""
    print("\n" + "=" * 65)
    print("KOMBINASJONSTEST — topp enkeltparametre kombinert")
    print("=" * 65)

    p0 = theta_til_params(theta)
    irf0 = _irf(p0)
    rm0 = rmse_16pt(irf0) if irf0 is not None else 999.0

    kombinasjoner = [
        ("Basis (kj33/kj34)",
         {}),
        ("gamma_p=0.50",
         {"gamma_p": 0.50}),
        ("gamma_p=0.65",
         {"gamma_p": 0.65}),
        ("kappa_M=0.10",
         {"kappa_M": 0.10}),
        ("gamma_p=0.50 + kappa_M=0.10",
         {"gamma_p": 0.50, "kappa_M": 0.10}),
        ("gamma_p=0.65 + kappa_M=0.10",
         {"gamma_p": 0.65, "kappa_M": 0.10}),
        ("gamma_p=0.50 + kappa_M=0.15 + h_c=0.90",
         {"gamma_p": 0.50, "kappa_M": 0.15, "h_c": 0.90}),
        ("gamma_p=0.65 + kappa_M=0.15 + phi_PQ=400",
         {"gamma_p": 0.65, "kappa_M": 0.15, "phi_PQ": 400}),
        ("gamma_p=0.65 + kappa_M=0.20 + phi_PQ=350",
         {"gamma_p": 0.65, "kappa_M": 0.20, "phi_PQ": 350}),
    ]

    print(f"\n  {'Kombinasjon':<45} {'RMSE':>8} {'PI.q4':>7} {'PI.q8':>7} "
          f"{'Y.q4':>7} {'I_R.q12':>8} {'RER.q12':>8}")
    print("  " + "-" * 95)
    for label, overrides in kombinasjoner:
        p = theta_til_params(theta)
        for attr, val in overrides.items():
            setattr(p, attr, val)
        irf = _irf(p)
        if irf is None:
            print(f"  {label:<45} {'BK-ustabil':>8}")
            continue
        rm = rmse_16pt(irf)
        by4 = irf[3, Y] / (-0.45)
        bpi4 = irf[3, PI] / (-0.15)
        marker = " ✅" if rm < rm0 else ""
        print(f"  {label:<45} {rm:>8.4f} {irf[3,PI]:>7.3f} {irf[7,PI]:>7.3f} "
              f"{irf[3,Y]:>7.3f} {irf[11,I_R]:>8.3f} {irf[11,RER]:>8.3f}{marker}")

    print(f"\n  NB-referanse:                                  {'—':>8} "
          f"  -0.150  -0.200  -0.450   +0.050  -0.050")


# ─────────────────────────────────────────────────────────────────────────────
# Hovedprogram
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    rot = Path(__file__).parent.parent

    # Last posterior — prioritér kj34, fallback til kj33 tail
    if len(sys.argv) > 1:
        posterior_path = sys.argv[1]
    else:
        # Prøv i rekkefølge
        for candidate in [
            rot / "data/results/chain_kj34_prod_posterior.json",
            rot / "data/results/chain_kj33_tail_posterior.json",
            rot / "data/results/chain_kj33_prod_partial.npy",
            rot / "data/results/chain_kj31_prod_posterior.json",
        ]:
            if candidate.exists():
                posterior_path = str(candidate)
                break
        else:
            print("Ingen posterior funnet. Bruk: python scripts/sandkasse_diagnostikk.py <posterior_json>")
            sys.exit(1)

    theta, label = last_posterior(posterior_path)
    print(f"\n{'='*65}")
    print(f"SANDKASSE DIAGNOSTIKK — Fase 0.75")
    print(f"Posterior: {label}")
    print(f"psi_R={theta[PARAM_NAMES.index('psi_R')]:.4f}  "
          f"gamma_p={theta[PARAM_NAMES.index('gamma_p')]:.4f}  "
          f"rho_A={theta[PARAM_NAMES.index('rho_A')]:.4f}")
    print(f"{'='*65}")

    spor_a_data(theta)
    spor_b_modellstruktur(theta)
    spor_c_estimering(theta)
    beste_kombinasjon(theta)

    print(f"\n{'='*65}")
    print("Kjøring fullført. Dokumenter topp-3 kandidater i mcmc_log.md")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
