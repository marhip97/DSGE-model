"""
[ARK/DSGE] Impulsrespons-funksjoner (IRF) for NEMO v3.

Referanse: Kravik & Mimir (2019) §4.
"""

import json
import logging
import warnings
from typing import Dict, List, Tuple

import numpy as np

from nemo.model.equations import (
    NE, NZ,
    Y, PI, I_R, RER, Q_H,
    E_A, E_C, E_P, E_O, E_Ys, E_rp, E_i, E_H,
    build_matrices_v3,
)
from nemo.model.parameters import Parameters as P
from nemo.solver.blanchard_kahn import solve as bk_solve

logger = logging.getLogger(__name__)

PARAM_NAMES: List[str] = [
    'rho_A', 'rho_C', 'rho_O', 'rho_Ys', 'rho_rp', 'rho_H',
    'sigma_C', 'sigma_O', 'sigma_Ys', 'sigma_rp', 'sigma_i',
    'sigma_P', 'sigma_H', 'psi_R', 'psi_P1', 'psi_Y', 'h_c',
]

SIGMA_A_FIXED: float = 0.006

SHOCK_NAMES: Dict[int, str] = {
    E_A: 'TFP', E_C: 'Konsum', E_P: 'Prismarkup', E_O: 'Oljepris',
    E_Ys: 'Ettersp.', E_rp: 'Risikopremie', E_i: 'Pengepol.', E_H: 'Bolig',
}

VAR_NAMES: Dict[int, str] = {
    Y: 'BNP-gap', PI: 'KPI-inflasjon', I_R: 'Styringsrente',
    RER: 'RER-gap', Q_H: 'Boligpris-gap',
}


def load_posterior(json_path: str) -> Tuple[np.ndarray, Dict[int, float], Dict[int, float]]:
    """
    Last posterior-oppsummering fra JSON.

    Returns:
        theta_post: parameter-vektor (posterior mean)
        sigma_vals: sjokk-standardavvik per sjokk-indeks
        rho_vals:   AR-koeffisienter per sjokk-indeks
    """
    with open(json_path) as f:
        post = json.load(f)
    summ = post['summary']
    theta_post = np.array([summ[n]['mean'] for n in PARAM_NAMES])

    idx = PARAM_NAMES.index

    sigma_vals: Dict[int, float] = {
        E_A:   SIGMA_A_FIXED,
        E_C:   float(theta_post[idx('sigma_C')]),
        E_P:   float(theta_post[idx('sigma_P')]),
        E_O:   float(theta_post[idx('sigma_O')]),
        E_Ys:  float(theta_post[idx('sigma_Ys')]),
        E_rp:  float(theta_post[idx('sigma_rp')]),
        E_i:   float(theta_post[idx('sigma_i')]),
        E_H:   float(theta_post[idx('sigma_H')]),
    }
    rho_vals: Dict[int, float] = {
        E_A:   float(theta_post[idx('rho_A')]),
        E_C:   float(theta_post[idx('rho_C')]),
        E_P:   0.0,
        E_O:   float(theta_post[idx('rho_O')]),
        E_Ys:  float(theta_post[idx('rho_Ys')]),
        E_rp:  float(theta_post[idx('rho_rp')]),
        E_i:   0.0,
        E_H:   float(theta_post[idx('rho_H')]),
    }
    return theta_post, sigma_vals, rho_vals


def build_estimated_model(theta_post: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Bygg og løs v3-modellen fra en parameter-vektor.

    Returns:
        T:      tilstandsovergangsmatrise (NZ × NZ)
        R:      sjokk-inngangsmatrise (NZ × NE)
        stable: True hvis max|eig(T)| < 1
    """
    class Pt(P):
        pass
    for i, name in enumerate(PARAM_NAMES):
        setattr(Pt, name, float(theta_post[i]))
    setattr(Pt, 'sigma_A', SIGMA_A_FIXED)
    G0, G1, Psi, Pi = build_matrices_v3(Pt, theta_H=0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        T, R, d = bk_solve(G0, G1, Psi, Pi, verbose=False)
    return T, R, d['stable']


def compute_irf(
    T: np.ndarray,
    R: np.ndarray,
    shock_idx: int,
    shock_size: float,
    n_periods: int = 20,
) -> np.ndarray:
    """
    Beregn impulsrespons for ett sjokk.

    Returns:
        irf: (n_periods × NZ) array — log-%-avvik fra stasjonær tilstand
    """
    irf = np.zeros((n_periods, NZ))
    state = R[:, shock_idx] * shock_size
    for t in range(n_periods):
        irf[t] = state
        state = T @ state
    return irf


def compute_all_irf(
    T: np.ndarray,
    R: np.ndarray,
    sigma_vals: Dict[int, float],
    n_periods: int = 20,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Beregn IRF for alle sjokk skalert med posterior standardavvik.

    Returns:
        {sjoknavn: {variabelnavn: [verdi per periode]}}
    """
    results: Dict[str, Dict[str, List[float]]] = {}
    for sidx, sname in SHOCK_NAMES.items():
        sigma = sigma_vals[sidx]
        irf = compute_irf(T, R, sidx, sigma, n_periods)
        results[sname] = {
            vname: [round(float(irf[t, vidx] * 100), 4) for t in range(n_periods)]
            for vidx, vname in VAR_NAMES.items()
        }
    logger.info("IRF beregnet: %d sjokk, %d perioder", len(results), n_periods)
    return results
