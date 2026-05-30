"""
[STAT] IRF-graf: kj35 vs kj34/kj31 vs NB Memo 3/2024 Figur 1

Sammenligner normalisert IRF (pengepolitisk sjokk +25 bp, peak I_R=1)
for Y, PI, I_R og RER over 20 kvartaler mot NB-referansepunkter ved
q1/q4/q8/q12.

Bruk:
  python scripts/irf_kj35_vs_nb.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import nb_multikvartal_score as nm  # lag_irf_normalisert, NB_FIGUR1, VAR_IDX, PARAM_NAMES

RESULTS = Path(__file__).parent.parent / "data" / "results"

# Horisont-indekser for NB-referansepunkter
NB_H = {"q1": 0, "q4": 3, "q8": 7, "q12": 11}

# Kjøringer å plotte: (label, posterior-fil, farge, stil)
RUNS = [
    ("kj31 (RMSE=0.353)", "chain_kj31_prod_posterior.json", "#bbbbbb", "--"),
    ("kj34 (RMSE=0.200)", "chain_kj34_prod_posterior.json", "#5b8def", "-."),
    ("kj35 (RMSE=0.154)", "chain_kj35_prod_posterior.json", "#d1495b", "-"),
]

VAR_TITLER = {
    "Y": "BNP (Y)",
    "PI": "KPI-inflasjon (PI)",
    "I_R": "Styringsrente (I_R)",
    "RER": "Realvalutakurs (RER)",
}


def _hent_theta(fil: Path) -> np.ndarray | None:
    """Leser posterior mean-vektor i PARAM_NAMES-rekkefølge fra JSON."""
    if not fil.exists():
        return None
    d = json.load(open(fil))
    # To formater: {'summary': {param: {'mean':..}}} eller {'mean': [..]}
    if "summary" in d and isinstance(d["summary"], dict):
        s = d["summary"]
        return np.array([s[n]["mean"] for n in nm.PARAM_NAMES])
    if "mean" in d:
        return np.array(d["mean"])
    return None


def main() -> None:
    n_per = 20
    t = np.arange(n_per)

    irfs = {}
    for label, filnavn, _, _ in RUNS:
        theta = _hent_theta(RESULTS / filnavn)
        if theta is None:
            print(f"  hopper over {label}: fant ikke {filnavn}")
            continue
        irf = nm.lag_irf_normalisert(theta)
        if irf is None:
            print(f"  hopper over {label}: IRF feilet (BK)")
            continue
        irfs[label] = irf

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        "NEMO IRF mot NB Memo 3/2024 Figur 1 — pengepolitisk sjokk (+25 bp, peak I_R=1)",
        fontsize=13,
        fontweight="bold",
    )

    for ax, vname in zip(axes.flat, ["Y", "PI", "I_R", "RER"]):
        vidx = nm.VAR_IDX[vname]

        # Modell-baner
        for label, filnavn, farge, stil in RUNS:
            if label not in irfs:
                continue
            ax.plot(
                t,
                irfs[label][:n_per, vidx],
                stil,
                color=farge,
                linewidth=2 if "kj35" in label else 1.5,
                label=label,
            )

        # NB-referansepunkter
        nb_x = [NB_H[q] for q in ["q1", "q4", "q8", "q12"]]
        nb_y = [nm.NB_FIGUR1[vname][q] for q in ["q1", "q4", "q8", "q12"]]
        ax.plot(
            nb_x,
            nb_y,
            "D",
            color="black",
            markersize=9,
            markerfacecolor="gold",
            markeredgewidth=1.5,
            label="NB Figur 1",
            zorder=5,
        )

        ax.axhline(0, color="#888888", linewidth=0.7)
        ax.set_title(VAR_TITLER[vname], fontsize=11)
        ax.set_xlabel("Kvartal etter sjokk")
        ax.set_ylabel("Normalisert respons")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    ut = RESULTS / "irf_kj35_vs_nb.png"
    fig.savefig(ut, dpi=130, bbox_inches="tight")
    print(f"Lagret: {ut}")


if __name__ == "__main__":
    main()
