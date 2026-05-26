"""
[NUM] B5-benchmark kj15 + kj17 vs kj12 vs NB Memo 3/2024 Figur 1.

kj15 — Test A: fjernet i_3m_obs (13 obs), 100k trekk
kj17 — Test C: kun pre-COVID (75 kv), 100k trekk
Begge bruker use_reparam=False → psi_R i naturlig rom (ingen logit-konvertering).

Beslutningspunkter:
  A (kj15): KPI-ratio >= 0.35× → fjern i_3m_obs i kj18
  C (kj17): KPI-ratio >= 0.35× → pre-only i kj18

Output:
  data/results/B5_irf_kj15_kj17_vs_kj12_vs_nb.png
  stdout — sammenligningstabell og konklusjon
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nemo.estimation.mcmc import PARAM_NAMES, KM
from nemo.model.equations import E_i, I_R, PI, RER, Y, build_matrices_v3
from nemo.model.parameters import Parameters
from nemo.solver.blanchard_kahn import compute_irf, solve as bk_solve

NB_Q4 = {"Y": -0.450, "PI": -0.150, "RER": -0.400, "I_R": +0.600}
T_PERIODER   = 20
SHOCK_SIZE_I = 0.0025

KJ12_NAMES = [
    'rho_A','rho_C','rho_O','rho_Ys','rho_rp','rho_H',
    'sigma_A','sigma_C','sigma_O','sigma_Ys','sigma_i','sigma_P','sigma_H',
    'psi_R','psi_P1','psi_Y','gamma_p','phi_I1','phi_I2','phi_u',
]


def posterior_mean(chain: np.ndarray, param_names: list[str]) -> dict:
    """Posterior mean fra siste halvdel. Ingen reparametrisering."""
    n = chain.shape[0]
    half = chain[n // 2:, :]
    means = half.mean(axis=0)
    return {name: float(means[i]) for i, name in enumerate(param_names)}


def bygg_params(navnverdier: dict, h_c: float = 0.938) -> type:
    class Pt(Parameters):
        pass
    for navn, val in navnverdier.items():
        setattr(Pt, navn, float(val))
    Pt.h_c = h_c
    Pt.kappa_M = KM["kappa_M"]
    return Pt


def bygg_og_loes(p_klasse: type, label: str = ""):
    try:
        G0, G1, Psi, Pi = build_matrices_v3(p_klasse)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            T, R, diag = bk_solve(G0, G1, Psi, Pi, verbose=False)
        if not diag["stable"]:
            print(f"  ADVARSEL [{label}]: ustabil løsning", file=sys.stderr)
            return None
        return T, R, diag
    except Exception as e:
        print(f"  FEIL [{label}]: {e}", file=sys.stderr)
        return None


def normaliser(irf: np.ndarray) -> np.ndarray:
    topp = float(np.max(irf[:, I_R]))
    if topp <= 0:
        return irf
    return irf / topp


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res_dir = ROOT / "data" / "results"

    # ── Last kjeder ──────────────────────────────────────────────────────────────
    print("Laster kjeder ...")

    kj12_chain = np.load(res_dir / "chain_kj12_prod.npy")
    kj12_mean  = posterior_mean(kj12_chain, KJ12_NAMES)
    print(f"  kj12: {kj12_chain.shape[0]} trekk, psi_R={kj12_mean['psi_R']:.4f}, psi_P1={kj12_mean['psi_P1']:.4f}")

    kj15_chain = np.load(res_dir / "chain_kj15_prod.npy")
    kj15_mean  = posterior_mean(kj15_chain, PARAM_NAMES)
    print(f"  kj15: {kj15_chain.shape[0]} trekk, psi_R={kj15_mean['psi_R']:.4f}, psi_P1={kj15_mean['psi_P1']:.4f}")

    kj17_chain = np.load(res_dir / "chain_kj17_prod.npy")
    kj17_mean  = posterior_mean(kj17_chain, PARAM_NAMES)
    print(f"  kj17: {kj17_chain.shape[0]} trekk, psi_R={kj17_mean['psi_R']:.4f}, psi_P1={kj17_mean['psi_P1']:.4f}")

    # ── Bygg og løs ───────────────────────────────────────────────────────────────
    kjeder = [
        ("kj12", kj12_mean),
        ("kj15", kj15_mean),
        ("kj17", kj17_mean),
    ]
    resultater = {}
    for label, mean_dict in kjeder:
        params = {k: v for k, v in mean_dict.items() if k in PARAM_NAMES}
        # kj12 brukte sigma_A=0.006 fast; PARAM_NAMES inneholder sigma_A
        params.setdefault("sigma_A", 0.006)
        P = bygg_params(params, h_c=0.938)
        print(f"\nLøser BK for {label} ...")
        res = bygg_og_loes(P, label=label)
        if res is None:
            print(f"HOPPER OVER {label}")
            continue
        T, R, diag = res
        irf = normaliser(compute_irf(T, R, E_i, SHOCK_SIZE_I, T_periods=T_PERIODER))
        resultater[label] = (irf, diag)
        print(f"  max|eig(T)|={diag['max_eig_T']:.5f}  stabil={diag['stable']}")

    # ── Sammenligningstabell ───────────────────────────────────────────────────────
    vars_ = [("BNP", Y, "Y"), ("KPI", PI, "PI"), ("RER", RER, "RER"), ("Rente", I_R, "I_R")]
    print("\n" + "=" * 80)
    print("Q4-respons (% / pp, normalisert til +1 pp rentetopp)")
    print("=" * 80)
    header = f"| {'Var':<10}|"
    for lbl in resultater:
        header += f" {lbl:>8} |"
    header += f" {'NB':>8} |"
    for lbl in resultater:
        header += f" {lbl+'/NB':>8} |"
    print(header)
    print("|" + "-"*11 + "|" + ("|".join(["-"*10]*( len(resultater)*2+1 ))) + "|")

    kpi_ratioer = {}
    for navn, idx, key in vars_:
        nb = NB_Q4[key] * 100
        row = f"| {navn+' q4':<10}|"
        for lbl, (irf, _) in resultater.items():
            row += f" {irf[3, idx]*100:+7.2f}% |"
        row += f" {nb:+7.2f}% |"
        for lbl, (irf, _) in resultater.items():
            ratio = (irf[3, idx]*100) / nb if nb != 0 else float("nan")
            row += f" {ratio:7.2f}x |"
            if key == "PI":
                kpi_ratioer[lbl] = ratio
        print(row)

    # ── Konklusjon ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("KONKLUSJON — Beslutningspunkter")
    print("=" * 80)
    for lbl, ratio in kpi_ratioer.items():
        terskel = 0.35
        status = "HJELPER" if ratio >= terskel else "HJELPER IKKE"
        print(f"  {lbl}: KPI q4-ratio = {ratio:.3f}x NB  →  {status} (terskel {terskel}x)")

    # ── Plot ───────────────────────────────────────────────────────────────────────
    print("\nLager figur ...")
    farger = {"kj12": "C0", "kj15": "C1", "kj17": "C2"}
    panels = [(Y, "BNP-gap (%)","Y"), (PI,"KPI-inflasjon (%)","PI"),
              (RER,"RER-gap (%)","RER"), (I_R,"Styringsrente (pp)","I_R")]
    nb_horisonter = [1, 4, 8, 12]
    nb_vals = {
        "Y":   [-0.20, -0.45, -0.35, -0.15],
        "PI":  [-0.05, -0.15, -0.20, -0.10],
        "RER": [-0.50, -0.40, -0.20, -0.05],
        "I_R": [+1.00, +0.60, +0.20, +0.05],
    }
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    kv = np.arange(1, T_PERIODER + 1)
    for ax, (idx, tittel, key) in zip(axes, panels):
        ax.axhline(0, color="0.6", linewidth=0.6)
        for lbl, (irf, _) in resultater.items():
            ax.plot(kv, irf[:, idx]*100, color=farger.get(lbl,"grey"),
                    linewidth=2.0, label=lbl)
        ax.scatter(nb_horisonter, nb_vals[key], color="darkorange",
                   marker="o", s=50, zorder=5, label="NB Memo")
        ax.set_title(tittel, fontsize=10)
        ax.set_xlabel("Kvartal", fontsize=9)
        ax.set_ylabel("%", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    kpi_txt = "  ".join([f"{lbl}={kpi_ratioer.get(lbl,float('nan')):.2f}x" for lbl in resultater])
    fig.suptitle(
        f"NEMO B5: kj15 (u/i_3m) + kj17 (pre-COVID) vs kj12 vs NB Memo 3/2024\n"
        f"KPI q4-ratio: {kpi_txt}  |  Terskel: 0.35x",
        fontsize=11,
    )
    fig.tight_layout()
    ut = res_dir / "B5_irf_kj15_kj17_vs_kj12_vs_nb.png"
    fig.savefig(ut, dpi=130)
    plt.close(fig)
    print(f"Figur lagret: {ut}")
    print("\nFerdig.")


if __name__ == "__main__":
    main()
