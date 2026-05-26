"""
[NUM] B5-benchmark kj18 vs kj16 vs kj12 vs NB Memo 3/2024 Figur 1.

kj18 — KPI-JAE + psi_R prior Beta(2,3,[0.01,0.970]), 200k trekk (PE-godkjent 2026-05-26)

Beslutningspunkt kj18:
  OK:   KPI q4-ratio ≥ 0.35× OG BNP q4-ratio 0.8–1.5× → kj18 er produksjonskjøring
  DELVIS: KPI OK men BNP ustabil → strukturell UIP-utvidelse (Fase 1B)
  FEIL: KPI < 0.35× → prior-justering underdrev KPI-kanalen

Output:
  data/results/B5_irf_kj18_vs_kj16_vs_kj12_vs_nb.png
  stdout — sammenligningstabell
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
NB_BNP_TERSKEL_LO = 0.8
NB_BNP_TERSKEL_HI = 1.5
KPI_TERSKEL = 0.35
T_PERIODER   = 20
SHOCK_SIZE_I = 0.0025

KJ12_NAMES = [
    'rho_A','rho_C','rho_O','rho_Ys','rho_rp','rho_H',
    'sigma_A','sigma_C','sigma_O','sigma_Ys','sigma_i','sigma_P','sigma_H',
    'psi_R','psi_P1','psi_Y','gamma_p','phi_I1','phi_I2','phi_u',
]


def posterior_mean(chain: np.ndarray, param_names: list[str]) -> dict:
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

    # ── Last kjeder ──────────────────────────────────────────────────────────
    kjeder_spec = [
        ("kj12", res_dir / "chain_kj12_prod.npy",  KJ12_NAMES),
        ("kj16", res_dir / "chain_kj16_prod.npy",  PARAM_NAMES),
        ("kj18", res_dir / "chain_kj18_prod.npy",  PARAM_NAMES),
    ]

    print("Laster kjeder ...")
    resultater = {}
    for label, fil, names in kjeder_spec:
        if not fil.exists():
            print(f"  {label}: {fil.name} ikke funnet — hopper over")
            continue
        chain = np.load(fil)
        mean_dict = posterior_mean(chain, names)
        mean_dict.setdefault("sigma_A", 0.006)
        print(f"  {label}: {chain.shape[0]} trekk, psi_R={mean_dict['psi_R']:.4f}, psi_P1={mean_dict['psi_P1']:.4f}")

        P = bygg_params({k: v for k, v in mean_dict.items() if k in PARAM_NAMES})
        print(f"\nLøser BK for {label} ...")
        res = bygg_og_loes(P, label=label)
        if res is None:
            continue
        T, R, diag = res
        irf = normaliser(compute_irf(T, R, E_i, SHOCK_SIZE_I, T_periods=T_PERIODER))
        resultater[label] = (irf, diag, mean_dict)
        print(f"  max|eig(T)|={diag['max_eig_T']:.5f}  stabil={diag['stable']}")

    # ── Sammenligningstabell ─────────────────────────────────────────────────
    vars_ = [("BNP", Y, "Y"), ("KPI", PI, "PI"), ("RER", RER, "RER"), ("Rente", I_R, "I_R")]
    print("\n" + "=" * 90)
    print("Q4-respons (% / pp, normalisert til +1 pp rentetopp)")
    print("=" * 90)

    kpi_ratioer = {}
    bnp_ratioer = {}
    for navn, idx, key in vars_:
        nb = NB_Q4[key] * 100
        row = f"| {navn+' q4':<10}|"
        for lbl, (irf, _, _) in resultater.items():
            row += f" {irf[3, idx]*100:+7.2f}% |"
        row += f" {nb:+7.2f}% (NB) |"
        print(row)
        for lbl, (irf, _, _) in resultater.items():
            ratio = (irf[3, idx]*100) / nb if nb != 0 else float("nan")
            if key == "PI":
                kpi_ratioer[lbl] = ratio
            if key == "Y":
                bnp_ratioer[lbl] = ratio

    # ── Nøkkelparametere ────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("Nøkkelparametere (posterior mean)")
    print(f"{'':12} {'kj12':>8} {'kj16':>8} {'kj18':>8} {'K&M':>8}")
    for par in ['psi_R','psi_P1','psi_Y','sigma_H','sigma_C','rho_H']:
        row = f"  {par:12}"
        for lbl in ['kj12','kj16','kj18']:
            if lbl in resultater:
                row += f" {resultater[lbl][2].get(par, float('nan')):8.4f}"
            else:
                row += f" {'—':>8}"
        from nemo.model.parameters import Parameters as P0
        row += f" {getattr(P0, par, float('nan')):8.4f}"
        print(row)

    # ── Konklusjon ───────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("KONKLUSJON — Beslutningspunkt kj18")
    print("=" * 90)
    if "kj18" in kpi_ratioer:
        kr = kpi_ratioer["kj18"]
        br = bnp_ratioer["kj18"]
        kpi_ok = kr >= KPI_TERSKEL
        bnp_ok = NB_BNP_TERSKEL_LO <= br <= NB_BNP_TERSKEL_HI
        print(f"  KPI q4-ratio = {kr:.3f}× NB  ({'≥' if kpi_ok else '<'}{KPI_TERSKEL}×) → {'OK ✓' if kpi_ok else 'FEIL ✗'}")
        print(f"  BNP q4-ratio = {br:.3f}× NB  ([{NB_BNP_TERSKEL_LO},{NB_BNP_TERSKEL_HI}]×) → {'OK ✓' if bnp_ok else 'USTABIL ✗'}")
        if kpi_ok and bnp_ok:
            print("\n  → kj18 er produksjonskjøring. Fase 1 kan starte.")
        elif kpi_ok and not bnp_ok:
            print("\n  → KPI OK men BNP ustabil. Vurder strukturell UIP-utvidelse (Fase 1B).")
        else:
            print("\n  → KPI under terskel. Prior-justering underdrev KPI-kanalen. Rekalibrer.")

    # ── Plot ─────────────────────────────────────────────────────────────────
    farger = {"kj12": "C0", "kj16": "C1", "kj18": "C2"}
    panels = [(Y,"BNP-gap (%)","Y"), (PI,"KPI-inflasjon (%)","PI"),
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
        for lbl, (irf, _, _) in resultater.items():
            ax.plot(kv, irf[:, idx]*100, color=farger.get(lbl,"grey"),
                    linewidth=2.0, label=lbl)
        ax.scatter(nb_horisonter, nb_vals[key], color="darkorange",
                   marker="o", s=50, zorder=5, label="NB Memo")
        ax.set_title(tittel, fontsize=10)
        ax.set_xlabel("Kvartal", fontsize=9)
        ax.set_ylabel("%", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    kpi_txt = "  ".join([f"{lbl}={kpi_ratioer.get(lbl,float('nan')):.2f}×"
                          for lbl in resultater])
    fig.suptitle(
        f"NEMO B5: kj18 (KPI-JAE+psi_R prior) vs kj16 vs kj12 vs NB Memo 3/2024\n"
        f"KPI q4-ratio: {kpi_txt}  |  Terskel: {KPI_TERSKEL}×",
        fontsize=11,
    )
    fig.tight_layout()
    ut = res_dir / "B5_irf_kj18_vs_kj16_vs_kj12_vs_nb.png"
    fig.savefig(ut, dpi=130)
    plt.close(fig)
    print(f"\nFigur lagret: {ut}")
    print("Ferdig.")


if __name__ == "__main__":
    main()
