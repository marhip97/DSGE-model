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
    build_matrices_v3_forward,
)
from nemo.model.parameters import Parameters as P
from nemo.solver.blanchard_kahn import solve as bk_solve

logger = logging.getLogger(__name__)

# Legacy-liste beholdt for bakoverkompatibilitet (eksportert via nemo.analysis).
# Selve innlastingen drives nå av posteriorens egne nøkler, se load_posterior.
PARAM_NAMES: List[str] = [
    'rho_A', 'rho_C', 'rho_O', 'rho_Ys', 'rho_rp', 'rho_H',
    'sigma_C', 'sigma_O', 'sigma_Ys', 'sigma_rp', 'sigma_i',
    'sigma_P', 'sigma_H', 'psi_R', 'psi_P1', 'psi_Y', 'h_c',
]

# ── kj41-kalibrering (referanseestimat) ───────────────────────────────────────
# kj41 fikserer disse utenfor estimeringen (se mcmc_log.md "kj41", PE-godkjent):
#   sigma_A=0.006, sigma_rp=0.006 (K&M-verdi, 2026-05-24), phi_PQ=150, lambda_pi4=0.
# Modellen bygges med build_matrices_v3_forward (NZ=50). Øvrige faste parametere
# (h_c=0.938, phi_u, kappa_M, phi_O ...) arves fra Parameters-defaultene, identisk
# med scripts/kj41_fevd_hd.py. Reproduserer data/results/kj41_fevd.json eksakt.
SIGMA_A_FIXED: float = 0.006
SIGMA_RP_FIXED: float = 0.006
PHI_PQ_KJ41: float = 150.0

SHOCK_NAMES: Dict[int, str] = {
    E_A: 'TFP', E_C: 'Konsum', E_P: 'Prismarkup', E_O: 'Oljepris',
    E_Ys: 'Utenl. ettersp.', E_rp: 'Risikopremie', E_i: 'Pengepol.', E_H: 'Bolig',
}

VAR_NAMES: Dict[int, str] = {
    Y: 'BNP-gap', PI: 'KPI-inflasjon', I_R: 'Styringsrente',
    RER: 'RER-gap', Q_H: 'Boligpris-gap',
}


def load_posterior(
    json_path: str,
) -> Tuple[Dict[str, float], Dict[int, float], Dict[int, float]]:
    """
    Last posterior-oppsummering fra JSON.

    Drives av posteriorens egne nøkler slik at vilkårlige estimerte parametersett
    (f.eks. kj41 som estimerer gamma_p/phi_I1/phi_I2/rho_s/phi_H1, men fikserer
    sigma_rp/h_c) håndteres uten å anta en fast liste. sigma_A og sigma_rp settes
    til de faste kj41-verdiene uavhengig av hva posterioren inneholder.

    Returns:
        param_means: {parameternavn: posterior mean} for de estimerte parameterne
        sigma_vals:  sjokk-standardavvik per sjokk-indeks
        rho_vals:    AR-koeffisienter per sjokk-indeks
    """
    with open(json_path) as f:
        post = json.load(f)
    summ = post['summary']
    param_means: Dict[str, float] = {n: float(summ[n]['mean']) for n in summ}

    def _m(name: str, default: float = 0.0) -> float:
        return param_means.get(name, default)

    sigma_vals: Dict[int, float] = {
        E_A:   SIGMA_A_FIXED,
        E_C:   _m('sigma_C'),
        E_P:   _m('sigma_P'),
        E_O:   _m('sigma_O'),
        E_Ys:  _m('sigma_Ys'),
        E_rp:  SIGMA_RP_FIXED,
        E_i:   _m('sigma_i'),
        E_H:   _m('sigma_H'),
    }
    rho_vals: Dict[int, float] = {
        E_A:   _m('rho_A'),
        E_C:   _m('rho_C'),
        E_P:   0.0,
        E_O:   _m('rho_O'),
        E_Ys:  _m('rho_Ys'),
        E_rp:  _m('rho_rp'),
        E_i:   0.0,
        E_H:   _m('rho_H'),
    }
    return param_means, sigma_vals, rho_vals


def build_estimated_model(
    param_means: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Bygg og løs kj41-modellen fra posteriorens estimerte parametere.

    Setter de estimerte parameterne fra ``param_means`` på en Parameters-subklasse
    med kj41-kalibreringen (phi_PQ=150, lambda_pi4=0; sigma_A/sigma_rp fikseres i
    ``load_posterior``), og bygger med build_matrices_v3_forward (NZ=50). Identisk
    konstruksjon som scripts/kj41_fevd_hd.py — reproduserer kj41_fevd.json eksakt.

    Returns:
        T:      tilstandsovergangsmatrise (NZ × NZ)
        R:      sjokk-inngangsmatrise (NZ × NE)
        stable: True hvis max|eig(T)| < 1
    """
    class Pt(P):
        phi_PQ = PHI_PQ_KJ41

    p = Pt()
    for name, value in param_means.items():
        if hasattr(p, name):
            setattr(p, name, float(value))
    p.lambda_pi4 = 0.0
    G0, G1, Psi, Pi = build_matrices_v3_forward(p, lambda_pi4=0.0)
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
