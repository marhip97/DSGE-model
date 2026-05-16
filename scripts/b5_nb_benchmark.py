"""
[NUM] Spor B5 — NB Memo 3/2024 Figur 1 benchmark.

Reproduserer impulssvar fra et pengepolitikkssjokk og sammenligner mot
NB Memo 3/2024 Figur 1.

Tre steg:
  1. Punkt-estimat: posterior mean, normalisert til +1 pp toppunkt på styringsrenten
  2. Usikkerhetsbånd: 500 trekk fra Gaussian-approksimering av posterior
  3. Avvik-rapport: numerisk sammenligning mot NB-figur

Output:
  - data/results/B5_nb_benchmark.json: numerisk IRF (mean, bånd, NB-referanse)
  - data/results/B5_nb_benchmark.png:  4-panels figur (BNP, KPI, Rente, RER)
  - data/results/B5_avvik_tabell.md:   markdown-tabell med avvik

Fase 2v2: Bruker reelle posteriortrekk fra chain_fase2v2_prod.npy (100k trekk, 20 param).
Usikkerhetsbånd basert på 500 tilfeldig utvalgte trekk fra kjeden.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import numpy as np

from nemo.estimation.mcmc import PARAM_NAMES
from nemo.model.equations import (
    E_i, I_R, PI, Q_H, RER, Y,
    build_matrices_v3,
)
from nemo.model.parameters import Parameters
from nemo.solver.blanchard_kahn import compute_irf, solve as bk_solve

logger = logging.getLogger(__name__)

# ── NB Memo 3/2024 Figur 1: ca. avlesning (kvartal 1, 4, 8, 12) ──────────────
# Verdier i prosent (BNP, KPI, RER) og prosentpoeng (rente).
# Disse er manuelle avlesninger fra figuren og brukes som indikative
# benchmarks, ikke autoritative tall.
NB_FIGUR1 = {
    "Y":   {"q1": -0.20, "q4": -0.45, "q8": -0.35, "q12": -0.15},
    "PI":  {"q1": -0.05, "q4": -0.15, "q8": -0.20, "q12": -0.10},
    "I_R": {"q1": +1.00, "q4": +0.60, "q8": +0.20, "q12": +0.05},
    "RER": {"q1": -0.50, "q4": -0.40, "q8": -0.20, "q12": -0.05},
}

VARIABLER = [("BNP-gap (Y)", Y), ("KPI-inflasjon (PI)", PI),
             ("Styringsrente (I_R)", I_R), ("RER-gap (RER)", RER),
             ("Boligpris (Q_H)", Q_H)]

T_PERIODER = 20
N_SAMPLES  = 500
SHOCK_SIZE = 0.0025  # 25 bp initialsjokk, skaleres mot +1 pp rente-topp


def les_posterior(sti: Path) -> dict[str, dict[str, float]]:
    with sti.open() as fh:
        return json.load(fh)["summary"]


def bygg_pt_fra_theta(theta: np.ndarray) -> type:
    """Lager en Parameters-subklasse med oppdaterte estimerte verdier."""
    class Pt(Parameters):
        pass
    for i, n in enumerate(PARAM_NAMES):
        setattr(Pt, n, float(theta[i]))
    Pt.sigma_A = 0.006
    return Pt


def lag_irf(theta: np.ndarray, shock_size: float = SHOCK_SIZE) -> np.ndarray | None:
    """Returnerer (T_PERIODER × NZ) IRF, eller None ved BK-feil."""
    Pt = bygg_pt_fra_theta(theta)
    try:
        G0, G1, Psi, Pi = build_matrices_v3(Pt)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            T, R, diag = bk_solve(G0, G1, Psi, Pi, verbose=False)
        if not diag["stable"]:
            return None
        return compute_irf(T, R, E_i, shock_size, T_periods=T_PERIODER)
    except Exception:
        return None


def normaliser_til_1pp(irf: np.ndarray) -> np.ndarray:
    """Skaler IRF slik at toppen av I_R blir +1.0 (prosentpoeng)."""
    topp = float(np.max(irf[:, I_R]))
    if topp <= 0:
        return irf
    return irf / topp


def baand_fra_samples(samples: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Beregner median, 5- og 95-prosentilbånd over (n_samples × T × NZ)."""
    arr = np.stack(samples, axis=0)  # (n, T, NZ)
    return {
        "median": np.median(arr, axis=0),
        "p5":     np.percentile(arr, 5,  axis=0),
        "p95":    np.percentile(arr, 95, axis=0),
    }


def lag_avvik_tabell(punkt: np.ndarray) -> str:
    """Markdown-tabell: vår modell vs. NB Memo 3/2024."""
    horisonter = [(0, "q1"), (3, "q4"), (7, "q8"), (11, "q12")]
    rader = []
    rader.append("| Variabel | Horisont | Vår modell | NB Memo 3/2024 | Avvik |")
    rader.append("|----------|----------|-----------:|---------------:|------:|")
    nokkel_map = {Y: "Y", PI: "PI", I_R: "I_R", RER: "RER"}
    navn_map   = {Y: "BNP-gap", PI: "KPI-infl.", I_R: "Rente", RER: "RER"}
    for var_idx, nokkel in nokkel_map.items():
        for h_idx, h_navn in horisonter:
            vaar = punkt[h_idx, var_idx]
            nb   = NB_FIGUR1[nokkel][h_navn]
            avvik = vaar - nb
            rader.append(
                f"| {navn_map[var_idx]:<8} | {h_navn} | {vaar:+.3f} | {nb:+.3f} | {avvik:+.3f} |"
            )
    return "\n".join(rader)


def plot_irf(punkt: np.ndarray, baand: dict[str, np.ndarray], ut_sti: Path) -> None:
    """Lager 4-panels figur (BNP, KPI, Rente, RER)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    paneler = [
        (Y,   "BNP-gap (%)",         "Y"),
        (PI,  "KPI-inflasjon (%)",   "PI"),
        (I_R, "Styringsrente (pp)",  "I_R"),
        (RER, "RER-gap (%)",         "RER"),
    ]
    kvartaler = np.arange(1, T_PERIODER + 1)

    for ax, (var_idx, tittel, nokkel) in zip(axes.flat, paneler):
        ax.axhline(0, color="0.6", linewidth=0.5)
        ax.fill_between(
            kvartaler, baand["p5"][:, var_idx], baand["p95"][:, var_idx],
            alpha=0.25, color="C0", label="5/95-bånd (500 trekk)",
        )
        ax.plot(kvartaler, baand["median"][:, var_idx],
                color="C0", linewidth=1.2, label="Posterior median")
        ax.plot(kvartaler, punkt[:, var_idx],
                color="C3", linewidth=1.8, label="Posterior mean (norm.)")

        # NB Memo 3/2024 referansepunkter
        nb_q = [1, 4, 8, 12]
        nb_v = [NB_FIGUR1[nokkel]["q1"], NB_FIGUR1[nokkel]["q4"],
                NB_FIGUR1[nokkel]["q8"], NB_FIGUR1[nokkel]["q12"]]
        ax.scatter(nb_q, nb_v, color="k", marker="x", s=40, zorder=5,
                   label="NB Memo 3/2024")

        ax.set_title(tittel)
        ax.set_xlabel("Kvartal")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        "Pengepolitikkssjokk: NEMO v3 vs. NB Memo 3/2024 Figur 1\n"
        "Normalisert: +1 pp toppunkt i styringsrenten",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(ut_sti, dpi=120)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rot = Path(__file__).resolve().parents[1]
    posterior_sti = rot / "data" / "results" / "chain_fase2v2_prod_posterior.json"
    kjede_sti     = rot / "data" / "results" / "chain_fase2v2_prod.npy"
    ut_dir = rot / "data" / "results"

    summary = les_posterior(posterior_sti)
    theta_mean = np.array([summary[n]["mean"] for n in PARAM_NAMES])

    logger.info("Steg 1/3 — Punkt-estimat (posterior mean)")
    irf_mean = lag_irf(theta_mean)
    if irf_mean is None:
        raise RuntimeError("BK-feil ved posterior mean — kan ikke fortsette")
    punkt = normaliser_til_1pp(irf_mean)

    logger.info("Steg 2/3 — Usikkerhetsbånd (%d trekk fra kjeden)", N_SAMPLES)
    kjede = np.load(kjede_sti)  # (100000, 20)
    rng = np.random.default_rng(2024)
    idx = rng.choice(len(kjede), size=N_SAMPLES * 4, replace=False)
    samples_norm: list[np.ndarray] = []
    avvist = 0
    for i in idx:
        if len(samples_norm) >= N_SAMPLES:
            break
        theta = kjede[i]
        irf = lag_irf(theta)
        if irf is None:
            avvist += 1
            continue
        samples_norm.append(normaliser_til_1pp(irf))
    logger.info("  %d trekk akseptert, %d avvist (BK-feil)", len(samples_norm), avvist)

    baand = baand_fra_samples(samples_norm)

    logger.info("Steg 3/3 — Avvik-rapport mot NB Memo 3/2024 Figur 1")
    tabell = lag_avvik_tabell(punkt)

    # ── Lagre resultater ─────────────────────────────────────────────────────
    json_sti = ut_dir / "B5_nb_benchmark.json"
    plot_sti = ut_dir / "B5_nb_benchmark.png"
    md_sti   = ut_dir / "B5_avvik_tabell.md"

    resultat = {
        "punkt_estimat": punkt.tolist(),
        "median":        baand["median"].tolist(),
        "p5":            baand["p5"].tolist(),
        "p95":           baand["p95"].tolist(),
        "nb_referanse":  NB_FIGUR1,
        "meta": {
            "n_samples":   len(samples_norm),
            "n_avvist":    avvist,
            "shock_size":  SHOCK_SIZE,
            "normalisert": "+1 pp toppunkt I_R",
            "posterior":   posterior_sti.name,
            "merknad":     "Reelle posteriortrekk fra chain_fase2v2_prod.npy (100k, 20 param)",
        },
    }
    json_sti.write_text(json.dumps(resultat, indent=2))
    md_sti.write_text(
        "# B5 — Avvik mot NB Memo 3/2024 Figur 1\n\n"
        "Pengepolitikkssjokk, normalisert til +1 pp toppunkt i styringsrenten.\n"
        f"Vår modell = posterior mean fra `{posterior_sti.name}` (Fase 2v2, NZ=49, 20 param).\n\n"
        + tabell + "\n"
    )

    try:
        plot_irf(punkt, baand, plot_sti)
        logger.info("Figur lagret: %s", plot_sti)
    except ImportError:
        logger.warning("matplotlib mangler — hopper over plot. Installer med: pip install -e '.[viz]'")

    logger.info("\n%s", tabell)
    logger.info("\nResultater lagret:")
    logger.info("  %s", json_sti)
    logger.info("  %s", md_sti)


if __name__ == "__main__":
    main()
