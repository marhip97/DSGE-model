"""
[NUM] B5-benchmark kj12 vs kj10 vs NB Memo 3/2024 Figur 1.

Kjøring 12 implementerer:
  - A6: sigma_A frigjort (var fast=0.006). Posterior ~0.013.
  - A7: psi_W-ledd lagt til i mimicking rule (G0[20, PIW] = -(1-psi_R)*psi_W).

Modellen har nå 20 frie parametre (var 19 i kj10).

Output:
  data/results/B5_irf_kj12_vs_kj10_vs_nb.png  — 4-panels figur
  stdout — sammenligningstabell og konklusjon
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nemo.estimation.mcmc import PARAM_NAMES
from nemo.model.equations import (
    E_i,
    I_R, PI, RER, Y,
    build_matrices_v3,
)
from nemo.model.parameters import Parameters
from nemo.solver.blanchard_kahn import compute_irf, solve as bk_solve

# ── NB Memo 3/2024 Figur 1 referanseverdier (normalisert til +1 pp rentejokk) ──
NB_Q4 = {"Y": -0.450, "PI": -0.150, "RER": -0.400, "I_R": +0.600}

T_PERIODER   = 20
SHOCK_SIZE_I = 0.0025   # normaliseres til +1 pp på rentetopp

# psi_R lagres i logit-rom i kj12-kjeden. Konverter til naturlig rom:
#   psi_R_natur = 0.01 + 0.91/(1 + exp(-logit))
PSI_R_LO, PSI_R_HI = 0.01, 0.92   # prior-grenser
def logit_to_psi_R(z: float) -> float:
    return PSI_R_LO + (PSI_R_HI - PSI_R_LO) / (1.0 + np.exp(-z))


def posterior_mean_kj12(chain: np.ndarray) -> dict:
    """Beregner posterior mean fra siste halvdel av kj12-kjeden.

    psi_R konverteres fra logit- til naturlig rom før gjennomsnitt.
    """
    n = chain.shape[0]
    half = chain[n // 2 :, :]            # siste halvdel
    psi_R_idx = PARAM_NAMES.index("psi_R")
    half_nat = half.copy()
    half_nat[:, psi_R_idx] = logit_to_psi_R(half[:, psi_R_idx])
    means = half_nat.mean(axis=0)
    return {name: float(means[i]) for i, name in enumerate(PARAM_NAMES)}


def bygg_params(navnverdier: dict, h_c: float = 0.938) -> type:
    """Lager Parameters-subklasse med oppgitte verdier."""
    class Pt(Parameters):
        pass
    for navn, val in navnverdier.items():
        setattr(Pt, navn, float(val))
    Pt.h_c = h_c
    return Pt


def les_kj10_mean() -> dict:
    with (ROOT / "data" / "results" / "chain_kj10_prod_posterior.json").open() as fh:
        summary = json.load(fh)["summary"]
    return {n: float(summary[n]["mean"]) for n in summary}


def bygg_og_loes(p_klasse: type, verbose: bool = False):
    """Bygger modell og løser med BK. Returnerer (T, R, diag) eller None."""
    try:
        G0, G1, Psi, Pi = build_matrices_v3(p_klasse)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            T, R, diag = bk_solve(G0, G1, Psi, Pi, verbose=verbose)
        if not diag["stable"]:
            print(f"  ADVARSEL: ustabil løsning, max|eig|={diag['max_eig_T']}",
                  file=sys.stderr)
            return None
        return T, R, diag
    except Exception as e:
        print(f"  FEIL i bygg_og_loes: {e}", file=sys.stderr)
        return None


def normaliser(irf: np.ndarray) -> np.ndarray:
    topp = float(np.max(irf[:, I_R]))
    if topp <= 0:
        print("  ADVARSEL: rentetopp <= 0", file=sys.stderr)
        return irf
    return irf / topp


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res_dir = ROOT / "data" / "results"

    # ── Last kj12-kjeden ─────────────────────────────────────────────────────
    print("Laster kj12-kjeden ...")
    chain = np.load(res_dir / "chain_kj12_temp_partial.npy")
    print(f"  shape: {chain.shape}")

    kj12_mean = posterior_mean_kj12(chain)
    print("\nkj12 posterior mean (siste halvdel av kjeden):")
    for n in PARAM_NAMES:
        print(f"  {n:10s} = {kj12_mean[n]:+.5f}")

    # ── kj10 posterior mean (json-summary) ───────────────────────────────────
    kj10_mean = les_kj10_mean()
    print(f"\nkj10 posterior mean (fra json):  rho_A={kj10_mean['rho_A']:.4f}, "
          f"psi_R={kj10_mean['psi_R']:.4f}")

    # ── Bygg Parameters-klasser ──────────────────────────────────────────────
    P12 = bygg_params(kj12_mean, h_c=0.938)         # kj12: alt fra kjede + h_c fast
    P10_vals = dict(kj10_mean)
    P10_vals.setdefault("sigma_A", 0.006)            # kj10 hadde sigma_A=0.006 fast
    P10 = bygg_params(P10_vals, h_c=0.938)

    print(f"\nP12: sigma_A={P12.sigma_A:.5f}, psi_R={P12.psi_R:.4f}, psi_W={P12.psi_W:.4f}")
    print(f"P10: sigma_A={P10.sigma_A:.5f}, psi_R={P10.psi_R:.4f}, psi_W={P10.psi_W:.4f}")

    # ── Løs BK ───────────────────────────────────────────────────────────────
    print("\nLøser BK for kj12 ...")
    res12 = bygg_og_loes(P12, verbose=True)
    if res12 is None:
        print("FEIL: kj12 BK feilet", file=sys.stderr)
        sys.exit(1)
    T12, R12, d12 = res12

    print("\nLøser BK for kj10 ...")
    res10 = bygg_og_loes(P10, verbose=False)
    if res10 is None:
        print("FEIL: kj10 BK feilet", file=sys.stderr)
        sys.exit(1)
    T10, R10, d10 = res10

    # ── IRF (normalisert til +1 pp rentetopp) ────────────────────────────────
    irf12 = normaliser(compute_irf(T12, R12, E_i, SHOCK_SIZE_I, T_periods=T_PERIODER))
    irf10 = normaliser(compute_irf(T10, R10, E_i, SHOCK_SIZE_I, T_periods=T_PERIODER))

    # ── Sammenligningstabell q4 ──────────────────────────────────────────────
    vars_ = [("BNP",   Y,   "Y"),
             ("KPI",   PI,  "PI"),
             ("RER",   RER, "RER"),
             ("Rente", I_R, "I_R")]
    print("\n" + "=" * 70)
    print("Q4-respons (% / pp etter normalisering til +1 pp rentetopp)")
    print("=" * 70)
    print(f"| {'Variabel':<10}| {'kj12':>8} | {'kj10':>8} | {'NB':>8} | {'kj12/NB':>8} |")
    print(f"|{'-'*11}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*10}|")
    for navn, idx, key in vars_:
        v12 = irf12[3, idx] * 100
        v10 = irf10[3, idx] * 100
        nb  = NB_Q4[key] * 100
        ratio = v12 / nb if nb != 0 else float("nan")
        print(f"| {navn+' q4':<10}| {v12:+7.2f}% | {v10:+7.2f}% | {nb:+7.2f}% | {ratio:7.2f}x |")

    # ── Plot ─────────────────────────────────────────────────────────────────
    print("\nLager 4-panels figur ...")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    kv = np.arange(1, T_PERIODER + 1)

    nb_horisonter = [1, 4, 8, 12]
    nb_vals = {
        "Y":   [-0.20, -0.45, -0.35, -0.15],
        "PI":  [-0.05, -0.15, -0.20, -0.10],
        "RER": [-0.50, -0.40, -0.20, -0.05],
        "I_R": [+1.00, +0.60, +0.20, +0.05],
    }
    panels = [
        (Y,   "BNP-gap (%)",        "Y"),
        (PI,  "KPI-inflasjon (%)",  "PI"),
        (RER, "RER-gap (%)",         "RER"),
        (I_R, "Styringsrente (pp)",  "I_R"),
    ]

    for ax, (idx, tittel, key) in zip(axes, panels):
        ax.axhline(0, color="0.6", linewidth=0.6)
        ax.plot(kv, irf10[:, idx] * 100, color="0.5", linewidth=1.4,
                linestyle="--", label="kj10")
        ax.plot(kv, irf12[:, idx] * 100, color="C0", linewidth=2.0,
                label="kj12")
        ax.scatter(nb_horisonter, nb_vals[key], color="darkorange",
                   marker="o", s=50, zorder=5, label="NB Memo 3/2024")
        ax.set_title(tittel, fontsize=10)
        ax.set_xlabel("Kvartal", fontsize=9)
        ax.set_ylabel("%", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(
        "NEMO B5: kj12 (A6+A7) vs kj10 vs NB Memo 3/2024 Figur 1\n"
        f"Pengepolitikkssjokk normalisert til +1 pp rentetopp  |  "
        f"kj12 max|eig(T)|={d12['max_eig_T']:.5f}",
        fontsize=11,
    )
    fig.tight_layout()

    ut = res_dir / "B5_irf_kj12_vs_kj10_vs_nb.png"
    fig.savefig(ut, dpi=130)
    plt.close(fig)
    print(f"Figur lagret: {ut}")

    # ── Konklusjon (kvantitativ) ─────────────────────────────────────────────
    kpi_kj12 = irf12[3, PI] * 100
    kpi_kj10 = irf10[3, PI] * 100
    kpi_nb   = NB_Q4["PI"] * 100
    ratio12  = kpi_kj12 / kpi_nb
    ratio10  = kpi_kj10 / kpi_nb
    forbedring = abs(kpi_kj10 - kpi_nb) - abs(kpi_kj12 - kpi_nb)

    print("\n" + "=" * 70)
    print("KONKLUSJON (KPI q4)")
    print("=" * 70)
    print(f"  kj10 KPI q4: {kpi_kj10:+.3f}%  (ratio NB: {ratio10:.2f}x)")
    print(f"  kj12 KPI q4: {kpi_kj12:+.3f}%  (ratio NB: {ratio12:.2f}x)")
    print(f"  NB   KPI q4: {kpi_nb:+.3f}%")
    print(f"  Forbedring i absolutt avvik: {forbedring:+.3f} pp")

    print("\nAlt lagret i data/results/")


if __name__ == "__main__":
    main()
