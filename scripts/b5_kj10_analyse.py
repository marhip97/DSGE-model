"""
[NUM] B5-benchmark kj10 vs kj9 vs NB + TFP-IRF diagnose (rho_A).

Oppgave 1: Pengepolitikkssjokk IRF — kj10 vs kj9 vs NB Memo 3/2024 Figur 1
Oppgave 2: TFP-sjokk IRF — kj10 rho_A=0.390 vs kj10 rho_A=0.950 (K&M-kalibrert)

Output:
  data/results/B5_irf_kj10_vs_kj9_vs_nb.png  — 8-panels figur
  stdout — ratio-tabell og tekstoppsummering
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

# Legg til src i path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nemo.estimation.mcmc import PARAM_NAMES
from nemo.model.equations import (
    E_A, E_i,
    I_R, INV, PI, RER, Y, C,
    build_matrices_v3,
)
from nemo.model.parameters import Parameters
from nemo.solver.blanchard_kahn import compute_irf, solve as bk_solve

# ── Referanseverdier NB Memo 3/2024 Figur 1 ──────────────────────────────────
NB_REFERANSE = {
    "Y":   {"q4": -0.450},
    "PI":  {"q4": -0.150},
    "RER": {"q4": -0.400},
    "I_R": {"q4": +0.600},
}

T_PERIODER = 20
SHOCK_SIZE_I  = 0.0025   # 25 bp pengepolitikksjokk (normaliseres)
SHOCK_SIZE_A  = 0.007    # TFP-sjokk per K&M


def bygg_params_fra_posterior(summary: dict) -> type:
    """Lager en Parameters-subklasse med posterior mean-verdier."""
    class Pt(Parameters):
        pass
    for navn in PARAM_NAMES:
        if navn in summary:
            setattr(Pt, navn, float(summary[navn]["mean"]))
    # sigma_A kalibrert fast (ikke estimert)
    Pt.sigma_A = 0.006
    return Pt


def les_posterior(sti: Path) -> dict:
    with sti.open() as fh:
        return json.load(fh)["summary"]


def bygg_og_loess(p_klasse: type, verbose: bool = False) -> tuple:
    """Bygger modellmatriser og løser BK. Returnerer (T, R, diag) eller None."""
    try:
        G0, G1, Psi, Pi = build_matrices_v3(p_klasse)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            T, R, diag = bk_solve(G0, G1, Psi, Pi, verbose=verbose)
        if not diag["stable"]:
            return None
        return T, R, diag
    except Exception as e:
        print(f"  FEIL: {e}", file=sys.stderr)
        return None


def normaliser_til_1pp(irf: np.ndarray) -> np.ndarray:
    """Skaler IRF slik at toppen av I_R = +1.0 pp."""
    topp = float(np.max(irf[:, I_R]))
    if topp <= 0:
        print("  ADVARSEL: Renten stiger ikke ved pengepolitikksjokk!", file=sys.stderr)
        return irf
    return irf / topp


def ratio_tabell(label: str, irf_norm: np.ndarray) -> str:
    """Formater ratio-tabell for q1, q4, q8 mot NB-referanse."""
    horisonter = [(0, "q1"), (3, "q4"), (7, "q8"), (11, "q12")]
    var_map = [("BNP",   Y,   "Y"),   ("KPI",   PI,  "PI"),
               ("RER",   RER, "RER"), ("Rente", I_R, "I_R")]
    linjer = [f"\n=== {label} — Pengepolitikkssjokk vs. NB Memo 3/2024 ==="]
    linjer.append(f"{'Var':<8} {'q1':>8} {'q4':>8} {'q8':>8} {'q12':>8}  |  {'NB q4':>8}  {'ratio q4':>10}")
    linjer.append("-" * 70)
    for navn, idx, nokkel in var_map:
        verdier = [f"{irf_norm[h, idx]*100:+.3f}" for h, _ in horisonter]
        nb_q4 = NB_REFERANSE.get(nokkel, {}).get("q4", None)
        nb_str = f"{nb_q4*100:+.3f}" if nb_q4 is not None else "   N/A"
        q4_val = irf_norm[3, idx] * 100
        ratio_str = f"{q4_val/nb_q4:.2f}x" if nb_q4 is not None and nb_q4 != 0 else "   N/A"
        linjer.append(f"{navn:<8} {verdier[0]:>8} {verdier[1]:>8} {verdier[2]:>8} {verdier[3]:>8}  |  {nb_str:>8}  {ratio_str:>10}")
    return "\n".join(linjer)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res_dir = ROOT / "data" / "results"

    # ── Last posterior-data ───────────────────────────────────────────────────
    print("Laster posterior-data ...")
    kj10_summary = les_posterior(res_dir / "chain_kj10_prod_posterior.json")
    kj9_summary  = les_posterior(res_dir / "chain_fase2_phio_phi1_prod_posterior.json")

    p_kj10 = bygg_params_fra_posterior(kj10_summary)
    p_kj9  = bygg_params_fra_posterior(kj9_summary)

    print(f"  kj10 rho_A={p_kj10.rho_A:.4f}, psi_R={p_kj10.psi_R:.4f}")
    print(f"  kj9  rho_A={p_kj9.rho_A:.4f},  psi_R={p_kj9.psi_R:.4f}")

    # ════════════════════════════════════════════════════════════════════════
    # OPPGAVE 1: Pengepolitikkssjokk IRF
    # ════════════════════════════════════════════════════════════════════════
    print("\nOppgave 1: Pengepolitikkssjokk IRF ...")

    res_kj10 = bygg_og_loess(p_kj10, verbose=True)
    res_kj9  = bygg_og_loess(p_kj9,  verbose=False)

    if res_kj10 is None:
        print("FEIL: BK-løsning feilet for kj10", file=sys.stderr)
        sys.exit(1)
    if res_kj9 is None:
        print("ADVARSEL: BK-løsning feilet for kj9 — hopper over", file=sys.stderr)

    T10, R10, diag10 = res_kj10
    irf10_raw = compute_irf(T10, R10, E_i, SHOCK_SIZE_I, T_periods=T_PERIODER)
    irf10 = normaliser_til_1pp(irf10_raw)

    if res_kj9 is not None:
        T9, R9, _ = res_kj9
        irf9_raw = compute_irf(T9, R9, E_i, SHOCK_SIZE_I, T_periods=T_PERIODER)
        irf9 = normaliser_til_1pp(irf9_raw)
    else:
        irf9 = None

    print(ratio_tabell("kj10", irf10))
    if irf9 is not None:
        print(ratio_tabell("kj9", irf9))

    # ════════════════════════════════════════════════════════════════════════
    # OPPGAVE 2: TFP-sjokk IRF — rho_A diagnose
    # ════════════════════════════════════════════════════════════════════════
    print("\nOppgave 2: TFP-sjokk IRF (rho_A diagnose) ...")

    # kj10 rho_A = 0.390 (posterior mean)
    class P_kj10_rhoA_low(p_kj10):
        pass
    # Allerede satt fra posterior (rho_A=0.390)

    # kj10 rho_A = 0.950 (K&M-kalibrert)
    class P_kj10_rhoA_high(p_kj10):
        rho_A = 0.950

    print(f"  TFP rho_A lav (kj10 posterior): {p_kj10.rho_A:.4f}")
    print(f"  TFP rho_A høy (K&M-kalibrert):  {P_kj10_rhoA_high.rho_A:.4f}")

    res_tfp_low  = bygg_og_loess(p_kj10,         verbose=False)
    res_tfp_high = bygg_og_loess(P_kj10_rhoA_high, verbose=False)

    if res_tfp_low is None or res_tfp_high is None:
        print("FEIL: BK-løsning feilet for TFP-analyse", file=sys.stderr)
        sys.exit(1)

    T_low,  R_low,  _ = res_tfp_low
    T_high, R_high, _ = res_tfp_high

    irf_tfp_low  = compute_irf(T_low,  R_low,  E_A, SHOCK_SIZE_A, T_periods=T_PERIODER)
    irf_tfp_high = compute_irf(T_high, R_high, E_A, SHOCK_SIZE_A, T_periods=T_PERIODER)

    # TFP-diagnose tekstrapport
    kvartaler = np.arange(1, T_PERIODER + 1)
    print("\n=== TFP-sjokk IRF (sigma_A=0.007) — rho_A diagnose ===")
    print(f"{'Var':<12} {'rhoA=0.39 q1':>14} {'rhoA=0.39 q4':>14} {'rhoA=0.95 q1':>14} {'rhoA=0.95 q4':>14}")
    print("-" * 70)
    for navn, idx in [("BNP (Y)", Y), ("KPI (PI)", PI), ("Inv (INV)", INV), ("Konsum (C)", C)]:
        l1 = irf_tfp_low[0, idx]*100;  l4 = irf_tfp_low[3, idx]*100
        h1 = irf_tfp_high[0, idx]*100; h4 = irf_tfp_high[3, idx]*100
        print(f"{navn:<12} {l1:>+14.3f} {l4:>+14.3f} {h1:>+14.3f} {h4:>+14.3f}")

    # Tid til halvparten av peak
    def halvtid(irf_col):
        peak_idx = np.argmax(np.abs(irf_col))
        peak_val = irf_col[peak_idx]
        if abs(peak_val) < 1e-10:
            return float("nan")
        for t in range(peak_idx, len(irf_col)):
            if abs(irf_col[t]) <= abs(peak_val) * 0.5:
                return t + 1  # 1-indeksert
        return float("nan")

    print("\n  Halvtid (kvartaler til halv peak) for BNP:")
    print(f"    rho_A=0.390: {halvtid(irf_tfp_low[:, Y]):.0f} kvartaler")
    print(f"    rho_A=0.950: {halvtid(irf_tfp_high[:, Y]):.0f} kvartaler")

    # ════════════════════════════════════════════════════════════════════════
    # PLOT — 8-panels figur
    # ════════════════════════════════════════════════════════════════════════
    print("\nLager 8-panels figur ...")

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    kv = np.arange(1, T_PERIODER + 1)

    # ── Panel 1-4: Pengepolitikkssjokk ────────────────────────────────────
    panels_mp = [
        (Y,   "Panel 1: BNP-gap (%)",        "Y"),
        (PI,  "Panel 2: KPI-inflasjon (%)",  "PI"),
        (RER, "Panel 3: RER-gap (%)",         "RER"),
        (I_R, "Panel 4: Styringsrente (pp)",  "I_R"),
    ]

    for ax, (idx, tittel, nokkel) in zip(axes[0], panels_mp):
        ax.axhline(0, color="0.6", linewidth=0.6)

        # kj9 — grå stiplet
        if irf9 is not None:
            ax.plot(kv, irf9[:, idx] * 100, color="0.5", linewidth=1.4,
                    linestyle="--", label="kj9 (grå stiplet)")

        # kj10 — blå
        ax.plot(kv, irf10[:, idx] * 100, color="C0", linewidth=2.0,
                label="kj10 (blå)")

        # NB referansepunkter — oransje prikk
        nb_q = [1, 4, 8, 12]
        nb_vals_map = {
            "Y":   [-0.20, -0.45, -0.35, -0.15],
            "PI":  [-0.05, -0.15, -0.20, -0.10],
            "RER": [-0.50, -0.40, -0.20, -0.05],
            "I_R": [+1.00, +0.60, +0.20, +0.05],
        }
        if nokkel in nb_vals_map:
            ax.scatter(nb_q, nb_vals_map[nokkel], color="darkorange",
                       marker="o", s=50, zorder=5, label="NB Memo 3/2024")

        ax.set_title(tittel, fontsize=9)
        ax.set_xlabel("Kvartal", fontsize=8)
        ax.set_ylabel("%", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    # ── Panel 5-8: TFP-sjokk ─────────────────────────────────────────────
    panels_tfp = [
        (Y,   "Panel 5: BNP (TFP-sjokk)"),
        (PI,  "Panel 6: KPI (TFP-sjokk)"),
        (INV, "Panel 7: Inv (TFP-sjokk)"),
        (C,   "Panel 8: Konsum (TFP-sjokk)"),
    ]

    for ax, (idx, tittel) in zip(axes[1], panels_tfp):
        ax.axhline(0, color="0.6", linewidth=0.6)

        # kj10 rho_A=0.390 — blå
        ax.plot(kv, irf_tfp_low[:, idx] * 100, color="C0", linewidth=2.0,
                label=r"kj10 $\rho_A$=0.390 (blå)")

        # kj10 rho_A=0.950 — rød stiplet
        ax.plot(kv, irf_tfp_high[:, idx] * 100, color="C3", linewidth=1.6,
                linestyle="--", label=r"K&M $\rho_A$=0.950 (rød stiplet)")

        ax.set_title(tittel, fontsize=9)
        ax.set_xlabel("Kvartal", fontsize=8)
        ax.set_ylabel("%", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle(
        "NEMO B5: kj10 vs kj9 vs NB Memo 3/2024  |  TFP rho_A diagnose\n"
        f"Pengepolitikk: normalisert +1 pp rente-topp  |  TFP: sjokk=0.007  |  kj10 max|eig(T)|={diag10['max_eig_T']:.5f}",
        fontsize=10,
    )
    fig.tight_layout()

    ut_sti = res_dir / "B5_irf_kj10_vs_kj9_vs_nb.png"
    fig.savefig(ut_sti, dpi=130)
    plt.close(fig)
    print(f"\nFigur lagret: {ut_sti}")

    # ── Diagnostikk-oppsummering ──────────────────────────────────────────
    print("\n" + "="*70)
    print("OPPSUMMERING")
    print("="*70)
    print(f"\nModell-stabilitet (kj10): max|eig(T)| = {diag10['max_eig_T']:.6f}  {'OK' if diag10['stable'] else 'USTABIL!'}")
    print(f"Løsningsmetode: {diag10['method']}")

    print("\n--- PENGEPOLITIKKSSJOKK: kj10 vs NB q4 (%-avvik) ---")
    for navn, idx, nokkel in [("BNP", Y, "Y"), ("KPI", PI, "PI"), ("RER", RER, "RER"), ("Rente", I_R, "I_R")]:
        kj10_q4 = irf10[3, idx] * 100
        nb_q4   = NB_REFERANSE[nokkel]["q4"] * 100
        avvik   = kj10_q4 - nb_q4
        print(f"  {navn:<8}  kj10={kj10_q4:+.3f}%  NB={nb_q4:+.3f}%  avvik={avvik:+.3f}%")

    if irf9 is not None:
        print("\n--- PENGEPOLITIKKSSJOKK: kj9 vs NB q4 (%-avvik) ---")
        for navn, idx, nokkel in [("BNP", Y, "Y"), ("KPI", PI, "PI"), ("RER", RER, "RER"), ("Rente", I_R, "I_R")]:
            kj9_q4  = irf9[3, idx] * 100
            nb_q4   = NB_REFERANSE[nokkel]["q4"] * 100
            avvik   = kj9_q4 - nb_q4
            print(f"  {navn:<8}  kj9={kj9_q4:+.3f}%   NB={nb_q4:+.3f}%  avvik={avvik:+.3f}%")

    print("\n--- TFP rho_A DIAGNOSE ---")
    print(f"  rho_A=0.390 (kj10 posterior): BNP peak q{np.argmax(irf_tfp_low[:,Y])+1}={np.max(irf_tfp_low[:,Y])*100:+.3f}%")
    print(f"  rho_A=0.950 (K&M-kalibrert): BNP peak q{np.argmax(irf_tfp_high[:,Y])+1}={np.max(irf_tfp_high[:,Y])*100:+.3f}%")
    halvtid_low  = halvtid(irf_tfp_low[:, Y])
    halvtid_high = halvtid(irf_tfp_high[:, Y])
    print(f"  Halvtid BNP:  rho_A=0.390 -> {halvtid_low:.0f} kvartaler,  rho_A=0.950 -> {halvtid_high:.0f} kvartaler")

    print("\n  VURDERING (rho_A=0.390):")
    if halvtid_low is not None and not np.isnan(halvtid_low) and halvtid_low <= 4:
        print("  -> rho_A=0.390 gir SVÆRT rask tilbakevending (< 4 kvartaler halvtid).")
        print("     Dette er et identifikasjonsproblem: posteriorens rho_A er for lavt")
        print("     estimert, trolig fordi andre sjokk absorberer TFP-persistensen.")
        print("     Sammenlign med K&M (2019) rho_A=0.95 — klar 'for-rask-tilbakevending'-feil.")
    elif halvtid_low is not None and not np.isnan(halvtid_low) and halvtid_low <= 8:
        print("  -> rho_A=0.390 gir moderat rask tilbakevending (4-8 kvartaler halvtid).")
        print("     Potensiell identifikasjonssvakhet — TFP er svakt identifisert (jf. C4).")
    else:
        print(f"  -> rho_A=0.390 gir halvtid {halvtid_low:.0f} kvartaler — rimelig?")

    print("\nAlle output lagret i data/results/")


if __name__ == "__main__":
    main()
