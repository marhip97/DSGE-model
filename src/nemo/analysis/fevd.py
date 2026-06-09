"""
[ARK/DSGE] Varians-dekomposisjon (FEVD) for NEMO v3.

Referanse: Kravik & Mimir (2019) §5.
"""

import logging
from typing import Dict, List

import numpy as np

from nemo.model.equations import NZ, Y, PI, I_R, RER, Q_H
from nemo.analysis.irf import SHOCK_NAMES, compute_irf

logger = logging.getLogger(__name__)

_VAR_MAP: Dict[str, int] = {
    'bnp': Y, 'pi': PI, 'rente': I_R, 'rer': RER, 'bolig': Q_H,
}


def compute_fevd(
    T: np.ndarray,
    R: np.ndarray,
    sigma_vals: Dict[int, float],
    n_periods: int = 20,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Beregn FEVD med σ²-normalisering.

    Returns:
        {variabelnavn: {sjoknavn: [andel per horisont, 0-indeksert]}}
    """
    cumvars: Dict[str, Dict[str, np.ndarray]] = {vn: {} for vn in _VAR_MAP}

    for sidx, sname in SHOCK_NAMES.items():
        sigma = sigma_vals[sidx]
        irf = compute_irf(T, R, sidx, sigma, n_periods)
        for vn, vidx in _VAR_MAP.items():
            cumvars[vn][sname] = np.cumsum(irf[:, vidx] ** 2)

    fevd_pct: Dict[str, Dict[str, List[float]]] = {}
    for vn in _VAR_MAP:
        fevd_pct[vn] = {}
        for h in range(n_periods):
            tot = sum(cumvars[vn][s][h] for s in cumvars[vn])
            for s in cumvars[vn]:
                if s not in fevd_pct[vn]:
                    fevd_pct[vn][s] = []
                fevd_pct[vn][s].append(
                    round(float(cumvars[vn][s][h] / max(tot, 1e-15) * 100), 2)
                )
    logger.info("FEVD beregnet: %d variabler, %d horisonter", len(fevd_pct), n_periods)
    return fevd_pct


def print_fevd_table(fevd_pct: Dict[str, Dict[str, List[float]]], horizons: List[int] = None) -> None:
    """Skriv FEVD-tabell til stdout."""
    if horizons is None:
        horizons = [1, 4, 8, 20]
    for vn, vdata in fevd_pct.items():
        print(f"\nFEVD — {vn.upper()}")
        h_strs = [f"h={h:2d}" for h in horizons]
        print(f"  {'Sjokk':<16} " + "  ".join(f"{h:>6}" for h in h_strs))
        print("  " + "─" * 52)
        for sname, vals in vdata.items():
            row = "  ".join(f"{vals[h - 1]:>6.1f}%" for h in horizons)
            print(f"  {sname:<16} {row}")
