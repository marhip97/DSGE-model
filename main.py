"""
================================================================================
NEMO FASE II v3 — HOVEDSKRIPT
Kjør hele pipelinen: data → løsning → estimering → analyse
================================================================================
Bruk:
    python main.py --step all          # kjør alle steg
    python main.py --step data         # kun datainnhenting
    python main.py --step solve        # kun modellløsning (kalibrert)
    python main.py --step estimate     # kun bayesiansk estimering (~2 timer)
    python main.py --step analyse      # kun IRF, FEVD og fremskrivning

Avhengigheter:
    pip install numpy scipy pandas requests matplotlib

Filstruktur:
    nemo_kodebase/
    ├── main.py                          (dette skriptet)
    ├── model/
    │   ├── equations.py                 (likningssystem, 48 variable)
    │   └── parameters.py               (kalibrerte parametere, K&M 2019)
    ├── solver/
    │   └── blanchard_kahn.py           (BK-løser, QZ-dekomposisjon)
    ├── estimering/
    │   └── nemo_estimering_v3.py       (MH-MCMC, 200k trekk)
    ├── analyse/
    │   └── nemo_analyse.py             (IRF, FEVD, fremskrivning)
    └── data/
        └── nemo_data_innhenting.py     (SSB, Norges Bank, FRED)
================================================================================
"""

import argparse
import os
import sys
import warnings
import numpy as np
import json

# Legg til undermapper i path
BASE = os.path.dirname(os.path.abspath(__file__))
for sub in ['model', 'solver', 'estimering', 'analyse', 'data']:
    sys.path.insert(0, os.path.join(BASE, sub))


# ── STEG 1: DATAINNHENTING ────────────────────────────────────────────────────
def step_data():
    print("\n" + "="*60)
    print("STEG 1: Datainnhenting")
    print("="*60)
    try:
        import nemo_data_innhenting
        print("  Kjører nemo_data_innhenting...")
        # Data-modulen kjøres som skript
        os.system(f"python {os.path.join(BASE,'data','nemo_data_innhenting.py')}")
    except ImportError:
        print("  ADVARSEL: nemo_data_innhenting.py ikke funnet.")
        print("  Last ned data manuelt og legg nemo_data_faktisk_v2.csv i arbeidsmappen.")


# ── STEG 2: MODELLLØSNING (KALIBRERT) ─────────────────────────────────────────
def step_solve():
    print("\n" + "="*60)
    print("STEG 2: Modellløsning (kalibrert, K&M 2019-parametere)")
    print("="*60)

    from equations import build_matrices_v3, NZ
    from blanchard_kahn import solve as bk_solve
    from parameters import Parameters as P

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        G0, G1, Psi, Pi = build_matrices_v3(P, theta_H=0.05)
        T_cal, R_cal, d = bk_solve(G0, G1, Psi, Pi, verbose=True)

    eig_max = float(np.abs(np.linalg.eigvals(T_cal)).max())
    print(f"\n  Kalibrert modell: stabil={d['stable']}, max|eig|={eig_max:.6f}")
    print(f"  Tilstandsrom: {NZ} variable")

    np.save('T_v3_calibrated.npy', T_cal)
    np.save('R_v3_calibrated.npy', R_cal)
    print("  Lagret: T_v3_calibrated.npy, R_v3_calibrated.npy")

    return T_cal, R_cal


# ── STEG 3: ESTIMERING ────────────────────────────────────────────────────────
def step_estimate():
    print("\n" + "="*60)
    print("STEG 3: Bayesiansk estimering (MH-MCMC, ~2 timer)")
    print("="*60)
    print("  Kjører nemo_estimering_v3.py som separat prosess...")
    ret = os.system(f"python {os.path.join(BASE,'estimering','nemo_estimering_v3.py')}")
    if ret != 0:
        print("  FEIL: Estimering returnerte ikke-null exit-kode.")
    else:
        print("  Estimering fullført.")


# ── STEG 4: ANALYSE ───────────────────────────────────────────────────────────
def step_analyse(posterior_file: str = 'chain_v3_v2_posterior.json',
                 data_file: str = 'nemo_data_faktisk_v2.csv'):
    print("\n" + "="*60)
    print("STEG 4: Analyse (IRF, FEVD, fremskrivning)")
    print("="*60)

    if not os.path.exists(posterior_file):
        print(f"  FEIL: {posterior_file} ikke funnet. Kjør estimering først.")
        return

    os.system(
        f"python {os.path.join(BASE,'analyse','nemo_analyse.py')} "
        f"--posterior {posterior_file} "
        f"--data {data_file} "
        f"--output analyse_resultater.json"
    )


# ── STEG 5: HURTIGTEST AV MODELLEN ────────────────────────────────────────────
def step_test():
    """Rask sanity-sjekk: løs kalibrert modell og verifiser BK og 3 IRF-tegn."""
    print("\n" + "="*60)
    print("TEST: Modellverifisering (kalibrert)")
    print("="*60)

    from equations import (build_matrices_v3, NZ, Y, PI, I_R, E_i, E_P, E_O)
    from blanchard_kahn import solve as bk_solve
    from parameters import Parameters as P

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        G0, G1, Psi, Pi = build_matrices_v3(P, theta_H=0.05)
        T, R, d = bk_solve(G0, G1, Psi, Pi, verbose=False)

    ok = True
    # BK
    bk_ok = d['stable']
    print(f"  BK-betingelse:           {'OK' if bk_ok else 'FEIL'}")
    if not bk_ok: ok = False

    # IRF-fortegnssjekker (3 enkle krav)
    def irf_cum(sidx, vidx, sz=0.001, periods=8):
        state = R[:, sidx] * sz
        vals = []
        for _ in range(periods):
            vals.append(float(state[vidx]))
            state = T @ state
        return sum(vals)

    checks = [
        ("Pengepol.(+) → BNP(-)", irf_cum(E_i, Y)  < 0),
        ("Pengepol.(+) → π(-)  ", irf_cum(E_i, PI) < 0),
        ("Kostnadssjokk → π(+) ", irf_cum(E_P, PI) > 0),
    ]
    for desc, result in checks:
        print(f"  {desc}: {'OK' if result else 'FEIL'}")
        if not result: ok = False

    print(f"\n  {'ALLE TESTER BESTÅTT' if ok else 'NOEN TESTER FEILET'}")
    return ok


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NEMO v3 hovedskript')
    parser.add_argument('--step', choices=['all','data','solve','estimate','analyse','test'],
                        default='test', help='Hvilket steg skal kjøres')
    parser.add_argument('--posterior', default='chain_v3_v2_posterior.json')
    parser.add_argument('--data',      default='nemo_data_faktisk_v2.csv')
    args = parser.parse_args()

    print("NEMO FASE II v3 — Makromodell for norsk økonomi")
    print("Kravik & Mimir (2019) — Bayesiansk estimert")
    print("Python-implementering")

    if args.step in ('all', 'test'):
        step_test()
    if args.step in ('all', 'data'):
        step_data()
    if args.step in ('all', 'solve'):
        step_solve()
    if args.step in ('all', 'estimate'):
        step_estimate()
    if args.step in ('all', 'analyse'):
        step_analyse(args.posterior, args.data)

    if args.step == 'all':
        print("\n" + "="*60)
        print("Pipeline fullført.")
        print("="*60)
