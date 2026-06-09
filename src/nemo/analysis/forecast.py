"""
[ARK/DSGE] Sjokk-betinget og ubetinget fremskrivning for NEMO v3.
"""

import logging
from typing import Dict, List

import numpy as np

from nemo.model.equations import NE, NZ, Y, PI, I_R, RER, Q_H
from nemo.analysis.irf import SHOCK_NAMES

logger = logging.getLogger(__name__)

_VAR_STD_MAP: Dict[str, int] = {
    'y': Y, 'pi': PI, 'i': I_R, 'rer': RER, 'bolig': Q_H,
}


def shock_conditional_forecast(
    T: np.ndarray,
    R: np.ndarray,
    sigma_vals: Dict[int, float],
    rho_vals: Dict[int, float],
    last_state: np.ndarray,
    last_innov: np.ndarray,
    n_fcst: int = 16,
) -> Dict:
    """
    Sjokk-betinget fremskrivning.

    Formel: z(T+h) = T^h·z_T + Σ_k ε_{k,T} · Σ_{s=0}^{h-1} ρ_k^(h-1-s) · T^s·r_k

    Returns:
        dict med nøkler:
            baseline:      (n_fcst × NZ) ubetinget bane
            conditional:   (n_fcst × NZ) betinget bane
            shock_contrib: {sjoknavn: (n_fcst × NZ)}
            eps_T:         {sjokk_idx: estimert sjokkstørrelse}
            std_P:         {variabelnøkkel: [prognosestandardavvik per horisont]}
    """
    # Identifiser aktive sjokk fra siste innovasjon
    eps_T: Dict[int, float] = {}
    for sidx in SHOCK_NAMES:
        r = R[:, sidx]
        eps = float(r @ last_innov) / (float(r @ r) + 1e-12)
        ratio = eps / sigma_vals[sidx]
        if abs(ratio) > 2.0:
            eps = np.sign(ratio) * 2.0 * sigma_vals[sidx]
        eps_T[sidx] = eps

    # Preberegn T^s matriser
    Ts = [np.eye(NZ)]
    for _ in range(1, n_fcst):
        Ts.append(T @ Ts[-1])

    # Baseline: T^h · z_T
    baseline = np.zeros((n_fcst, NZ))
    s = last_state.copy()
    for h in range(n_fcst):
        s = T @ s
        baseline[h] = s

    # Sjokk-bidrag
    shock_contrib: Dict[str, np.ndarray] = {}
    total_shock = np.zeros((n_fcst, NZ))
    for sidx, sname in SHOCK_NAMES.items():
        r = R[:, sidx]
        rho = rho_vals[sidx]
        eps = eps_T[sidx]
        contrib = np.zeros((n_fcst, NZ))
        for h in range(n_fcst):
            accum = np.zeros(NZ)
            for s_idx in range(h):
                accum += (rho ** (h - 1 - s_idx)) * (Ts[s_idx] @ r)
            contrib[h] = eps * accum
        shock_contrib[sname] = contrib
        total_shock += contrib

    conditional = baseline + total_shock

    # Prognosebånd P(h) fra P(0)=0
    Q_mat = np.zeros((NE, NE))
    for sidx, sigma in sigma_vals.items():
        Q_mat[sidx, sidx] = sigma ** 2
    RQR = R @ Q_mat @ R.T

    std_P: Dict[str, List[float]] = {vn: [] for vn in _VAR_STD_MAP}
    Ph = np.zeros((NZ, NZ))
    for h in range(n_fcst):
        Ph = T @ Ph @ T.T + RQR
        for vn, vidx in _VAR_STD_MAP.items():
            std_P[vn].append(float(np.sqrt(max(Ph[vidx, vidx], 0))))

    logger.info("Fremskrivning beregnet: %d perioder", n_fcst)
    return {
        'baseline': baseline,
        'conditional': conditional,
        'shock_contrib': shock_contrib,
        'eps_T': eps_T,
        'std_P': std_P,
    }
