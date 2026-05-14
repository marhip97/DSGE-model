"""
================================================================================
NEMO — HOVEDSKRIPT
Kjør hele pipelinen: data -> løsning -> estimering -> analyse
================================================================================
Bruk:
    python -m nemo --step all          # kjør alle steg
    python -m nemo --step data         # kun datainnhenting
    python -m nemo --step solve        # kun modellløsning (kalibrert)
    python -m nemo --step estimate     # kun bayesiansk estimering (~2 timer)
    python -m nemo --step analyse      # kun IRF, FEVD og fremskrivning
    python -m nemo --step test         # rask sanity-sjekk (kalibrert modell)
================================================================================
"""

import argparse
import os
import subprocess
import sys
import warnings

import numpy as np


# ── STEG 1: DATAINNHENTING ────────────────────────────────────────────────────
def step_data():
    print("\n" + "=" * 60)
    print("STEG 1: Datainnhenting")
    print("=" * 60)
    ret = subprocess.call([sys.executable, "-m", "nemo.data.innhenting"])
    if ret != 0:
        print("  ADVARSEL: datainnhenting returnerte ikke-null exit-kode.")


# ── STEG 2: MODELLLØSNING (KALIBRERT) ─────────────────────────────────────────
def step_solve():
    print("\n" + "=" * 60)
    print("STEG 2: Modellløsning (kalibrert, K&M 2019-parametere)")
    print("=" * 60)

    from nemo.model.equations import build_matrices_v3, NZ
    from nemo.model.parameters import Parameters as P
    from nemo.solver.blanchard_kahn import solve as bk_solve

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        G0, G1, Psi, Pi = build_matrices_v3(P, theta_H=0.05)
        T_cal, R_cal, d = bk_solve(G0, G1, Psi, Pi, verbose=True)

    eig_max = float(np.abs(np.linalg.eigvals(T_cal)).max())
    print(f"\n  Kalibrert modell: stabil={d['stable']}, max|eig|={eig_max:.6f}")
    print(f"  Tilstandsrom: {NZ} variable")

    os.makedirs("data/results", exist_ok=True)
    np.save("data/results/T_v3_calibrated.npy", T_cal)
    np.save("data/results/R_v3_calibrated.npy", R_cal)
    print("  Lagret: data/results/T_v3_calibrated.npy, data/results/R_v3_calibrated.npy")

    return T_cal, R_cal


# ── STEG 3: ESTIMERING ────────────────────────────────────────────────────────
def step_estimate():
    print("\n" + "=" * 60)
    print("STEG 3: Bayesiansk estimering (MH-MCMC, ~2 timer)")
    print("=" * 60)
    ret = subprocess.call([sys.executable, "-m", "nemo.estimation.mcmc"])
    if ret != 0:
        print("  FEIL: Estimering returnerte ikke-null exit-kode.")
    else:
        print("  Estimering fullført.")


# ── STEG 4: ANALYSE ───────────────────────────────────────────────────────────
def step_analyse(posterior_file: str = "data/results/chain_v3_v2_posterior.json",
                 data_file: str = "data/processed/nemo_data_faktisk_v2.csv"):
    print("\n" + "=" * 60)
    print("STEG 4: Analyse (IRF, FEVD, fremskrivning)")
    print("=" * 60)

    if not os.path.exists(posterior_file):
        print(f"  FEIL: {posterior_file} ikke funnet. Kjør estimering først.")
        return

    subprocess.call([
        sys.executable, "-m", "nemo.analysis.analyse",
        "--posterior", posterior_file,
        "--data", data_file,
        "--output", "data/results/analyse_resultater.json",
    ])


# ── STEG 5: HURTIGTEST AV MODELLEN ────────────────────────────────────────────
def step_test():
    """Rask sanity-sjekk: løs kalibrert modell og verifiser BK og 3 IRF-tegn."""
    print("\n" + "=" * 60)
    print("TEST: Modellverifisering (kalibrert)")
    print("=" * 60)

    from nemo.model.equations import build_matrices_v3, NZ, Y, PI, E_i, E_P
    from nemo.model.parameters import Parameters as P
    from nemo.solver.blanchard_kahn import solve as bk_solve

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        G0, G1, Psi, Pi = build_matrices_v3(P, theta_H=0.05)
        T, R, d = bk_solve(G0, G1, Psi, Pi, verbose=False)

    ok = True
    bk_ok = d["stable"]
    print(f"  BK-betingelse:           {'OK' if bk_ok else 'FEIL'}")
    if not bk_ok:
        ok = False

    def irf_cum(sidx, vidx, sz=0.001, periods=8):
        state = R[:, sidx] * sz
        vals = []
        for _ in range(periods):
            vals.append(float(state[vidx]))
            state = T @ state
        return sum(vals)

    checks = [
        ("Pengepol.(+) -> BNP(-)", irf_cum(E_i, Y) < 0),
        ("Pengepol.(+) -> pi(-) ", irf_cum(E_i, PI) < 0),
        ("Kostnadssjokk -> pi(+)", irf_cum(E_P, PI) > 0),
    ]
    for desc, result in checks:
        print(f"  {desc}: {'OK' if result else 'FEIL'}")
        if not result:
            ok = False

    print(f"\n  {'ALLE TESTER BESTAATT' if ok else 'NOEN TESTER FEILET'}")
    return ok


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="NEMO hovedskript")
    parser.add_argument(
        "--step",
        choices=["all", "data", "solve", "estimate", "analyse", "test"],
        default="test",
        help="Hvilket steg skal kjøres",
    )
    parser.add_argument("--posterior", default="data/results/chain_v3_v2_posterior.json")
    parser.add_argument("--data", default="data/processed/nemo_data_faktisk_v2.csv")
    args = parser.parse_args()

    print("NEMO - Makromodell for norsk økonomi")
    print("Kravik & Mimir (2019) - Bayesiansk estimert")

    if args.step in ("all", "test"):
        step_test()
    if args.step in ("all", "data"):
        step_data()
    if args.step in ("all", "solve"):
        step_solve()
    if args.step in ("all", "estimate"):
        step_estimate()
    if args.step in ("all", "analyse"):
        step_analyse(args.posterior, args.data)

    if args.step == "all":
        print("\n" + "=" * 60)
        print("Pipeline fullført.")
        print("=" * 60)


if __name__ == "__main__":
    main()
