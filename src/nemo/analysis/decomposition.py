"""
[ARK/NUM] Kalman-filter, RTS-smoother og historisk dekomposisjon for NEMO v3.

Referanse: Sargent (1989), Hamilton (1994) kap. 13.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from nemo.model.equations import NE, NZ
from nemo.analysis.irf import SHOCK_NAMES

logger = logging.getLogger(__name__)

OBS_NAMES: List[str] = [
    'dy_obs', 'dc_obs', 'dinv_obs', 'dx_obs', 'dm_obs', 'pi_obs', 'dw_obs',
    'i_R_obs', 'i_3m_obs', 'ds_obs', 'dpO_obs', 'dyS_obs', 'dh_obs', 'db_obs',
]

_SME: Dict[str, float] = {
    'dy_obs': 0.005, 'dc_obs': 0.008, 'dinv_obs': 0.015,
    'dx_obs': 0.010, 'dm_obs': 0.012, 'pi_obs': 0.008,
    'dw_obs': 0.004, 'i_R_obs': 0.0005, 'i_3m_obs': 0.0005,
    'ds_obs': 0.010, 'dpO_obs': 0.050, 'dyS_obs': 0.006,
    'dh_obs': 0.004, 'db_obs': 0.002,
}


def build_observation_system() -> Tuple[np.ndarray, np.ndarray]:
    """
    Bygg observasjonsmatrise H og målefeilkovariansmatrise Sv.

    Returns:
        H:  (14 × NZ) observasjonsmatrise
        Sv: (14 × 14) diagonal kovariansmatrise
    """
    from nemo.model.equations import (
        Y, C, INV, PI, W, I_R, RER, Q_H, B_NW, PO, YS,
    )
    H = np.zeros((14, NZ))
    H[0, Y] = 1
    H[1, C] = 1
    H[2, INV] = 1
    H[3, 4] = 1      # eksport
    H[4, 5] = 1      # import
    H[5, PI] = 4
    H[6, W] = 1
    H[7, I_R] = 4
    H[8, I_R] = 4    # i_3m mappes til I_R (kj52-valg)
    H[9, RER] = 1
    H[10, PO] = 1
    H[11, YS] = 1
    H[12, Q_H] = 1
    H[13, B_NW] = 1
    Sv = np.diag([_SME[n] ** 2 for n in OBS_NAMES])
    return H, Sv


def kalman_filter(
    T: np.ndarray,
    R: np.ndarray,
    H: np.ndarray,
    Q: np.ndarray,
    Sv: np.ndarray,
    Y_obs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fremover Kalman-filter med NaN-handtering (missing observations).

    Returns:
        z_filt: (T × NZ) filtrerte tilstander
        innov:  (T × NZ) tilstandsroms-innovasjoner
    """
    RQR = R @ Q @ R.T
    try:
        P0 = solve_discrete_lyapunov(T, RQR)
    except Exception:
        P0 = np.eye(NZ) * 0.01

    T_len = len(Y_obs)
    z = np.zeros(NZ)
    Pk = P0.copy()
    z_filt = np.zeros((T_len, NZ))
    innov = np.zeros((T_len, NZ))

    for t in range(T_len):
        zp = T @ z
        Pp = T @ Pk @ T.T + RQR
        yt = Y_obs[t]
        ms = np.isnan(yt)
        if not ms.all():
            Ht = H[~ms]
            yo = yt[~ms]
            Sv_t = Sv[np.ix_(~ms, ~ms)]
            innovation = yo - Ht @ zp
            S = Ht @ Pp @ Ht.T + Sv_t
            S = (S + S.T) / 2
            try:
                Kg = Pp @ Ht.T @ np.linalg.inv(S)
                z = zp + Kg @ innovation
                Pk = (np.eye(NZ) - Kg @ Ht) @ Pp
                Pk = (Pk + Pk.T) / 2
                innov[t] = z - zp
            except Exception:
                z = zp
                Pk = Pp
        else:
            z = zp
            Pk = Pp
        z_filt[t] = z

    logger.debug("Kalman-filter: %d perioder", T_len)
    return z_filt, innov


def rts_smoother(
    T: np.ndarray,
    R: np.ndarray,
    Q: np.ndarray,
    Y_obs: np.ndarray,
    H: np.ndarray,
    Sv: np.ndarray,
) -> np.ndarray:
    """
    Rauch-Tung-Striebel (RTS) bakover-glatter.

    Returns:
        z_smooth: (T × NZ) glattede tilstander
    """
    RQR = R @ Q @ R.T
    try:
        P0 = solve_discrete_lyapunov(T, RQR)
    except Exception:
        P0 = np.eye(NZ) * 0.01

    T_len = len(Y_obs)
    z_filt = np.zeros((T_len, NZ))
    P_filt = np.zeros((T_len, NZ, NZ))
    z_pred = np.zeros((T_len, NZ))
    P_pred = np.zeros((T_len, NZ, NZ))

    z = np.zeros(NZ)
    Pk = P0.copy()
    for t in range(T_len):
        zp = T @ z
        Pp = T @ Pk @ T.T + RQR
        z_pred[t] = zp
        P_pred[t] = Pp
        yt = Y_obs[t]
        ms = np.isnan(yt)
        if not ms.all():
            Ht = H[~ms]
            yo = yt[~ms]
            Sv_t = Sv[np.ix_(~ms, ~ms)]
            inn = yo - Ht @ zp
            S = Ht @ Pp @ Ht.T + Sv_t
            S = (S + S.T) / 2
            try:
                Kg = Pp @ Ht.T @ np.linalg.inv(S)
                z = zp + Kg @ inn
                Pk = (np.eye(NZ) - Kg @ Ht) @ Pp
                Pk = (Pk + Pk.T) / 2
            except Exception:
                z = zp
                Pk = Pp
        else:
            z = zp
            Pk = Pp
        z_filt[t] = z
        P_filt[t] = Pk

    # Bakover-glattingspass
    z_smooth = z_filt.copy()
    P_smooth = P_filt.copy()
    for t in range(T_len - 2, -1, -1):
        Pp_next = P_pred[t + 1]
        if np.abs(np.linalg.eigvals(Pp_next)).max() < 1e-12:
            continue
        L = P_filt[t] @ T.T @ np.linalg.pinv(Pp_next)
        z_smooth[t] = z_filt[t] + L @ (z_smooth[t + 1] - z_pred[t + 1])
        P_smooth[t] = P_filt[t] + L @ (P_smooth[t + 1] - Pp_next) @ L.T

    logger.debug("RTS-smoother: %d perioder", T_len)
    return z_smooth


def compute_historical_decomposition(
    T: np.ndarray,
    R: np.ndarray,
    z_smooth: np.ndarray,
    H: np.ndarray,
    obs_keys: Dict[int, str],
) -> Dict[str, Dict[str, List[float]]]:
    """
    Historisk dekomposisjon: sjokk-bidrag per observasjonsvariabel.

    Bruker residual-metoden: ε_t ≈ z_t − T·z_{t−1}, prosjektert på R-kolonner.

    Args:
        obs_keys: {obs_rad_indeks: variabelnøkkel}

    Returns:
        {variabelnøkkel: {sjoknavn: [bidrag per periode (t=1..T)]}}
    """
    T_hist = len(z_smooth)
    hist_decomp: Dict[str, Dict[str, List[float]]] = {}

    for obs_idx, vkey in obs_keys.items():
        h_row = H[obs_idx, :]
        shocks: Dict[str, List[float]] = {}
        for sidx, sname in SHOCK_NAMES.items():
            r_col = R[:, sidx]
            contrib = []
            for t in range(1, T_hist):
                eps_t = z_smooth[t] - T @ z_smooth[t - 1]
                obs_c = float(h_row @ (r_col * eps_t[sidx]))
                contrib.append(round(obs_c, 6))
            shocks[sname] = contrib
        hist_decomp[vkey] = shocks

    logger.info(
        "Historisk dekomposisjon: %d perioder, %d variabler",
        T_hist - 1, len(hist_decomp),
    )
    return hist_decomp
