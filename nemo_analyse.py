"""
================================================================================
NEMO FASE II v3 — ANALYSE
IRF, FEVD, historisk dekomposisjon og sjokk-betinget fremskrivning
================================================================================
Bruk:
    python nemo_analyse.py

Krever i samme mappe (eller tilgjengelig i Python-path):
    ../model/equations.py
    ../model/parameters.py
    ../solver/blanchard_kahn.py
    T_v3_estimated.npy, R_v3_estimated.npy   (fra estimering)
    chain_v3_v2_posterior.json               (fra estimering)
    nemo_data_faktisk_v2.csv                 (fra data-innhenting)
================================================================================
"""

import numpy as np
import pandas as pd
import json
import sys
import warnings
import os
from scipy.linalg import solve_discrete_lyapunov

# Legg til parent-dir i sti for importering
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'solver'))

from equations import (
    build_matrices_v3, NZ, NE,
    Y, PI, I_R, RER, Q_H, B_NW, C, INV, W, L, PO, YS,
    E_A, E_C, E_P, E_O, E_Ys, E_rp, E_i, E_H
)
from blanchard_kahn import solve as bk_solve
from parameters import Parameters as P

# ── KONFIGURASJON ─────────────────────────────────────────────────────────────
PARAM_NAMES = ['rho_A','rho_C','rho_O','rho_Ys','rho_rp','rho_H',
               'sigma_C','sigma_O','sigma_Ys','sigma_rp','sigma_i',
               'sigma_P','sigma_H','psi_R','psi_P1','psi_Y','h_c']
SIGMA_A_FIXED = 0.006

OBS_NAMES = ['dy_obs','dc_obs','dinv_obs','dx_obs','dm_obs','pi_obs','dw_obs',
             'i_R_obs','i_3m_obs','ds_obs','dpO_obs','dyS_obs','dh_obs','db_obs']

SHOCK_NAMES = {
    E_A:'TFP', E_C:'Konsum', E_P:'Prismarkup', E_O:'Oljepris',
    E_Ys:'Ettersp.', E_rp:'Risikopremie', E_i:'Pengepol.', E_H:'Bolig'
}

VAR_NAMES = {Y:'BNP-gap', PI:'KPI-inflasjon', I_R:'Styringsrente',
             RER:'RER-gap', Q_H:'Boligpris-gap'}


def load_posterior(json_path: str) -> tuple:
    """Last posterior fra JSON og returner theta_post, sigma_vals, rho_vals."""
    with open(json_path) as f:
        post = json.load(f)
    summ = post['summary']
    theta_post = np.array([summ[n]['mean'] for n in PARAM_NAMES])

    sigma_vals = {
        E_A: SIGMA_A_FIXED,
        E_C: theta_post[PARAM_NAMES.index('sigma_C')],
        E_P: theta_post[PARAM_NAMES.index('sigma_P')],
        E_O: theta_post[PARAM_NAMES.index('sigma_O')],
        E_Ys:theta_post[PARAM_NAMES.index('sigma_Ys')],
        E_rp:theta_post[PARAM_NAMES.index('sigma_rp')],
        E_i: theta_post[PARAM_NAMES.index('sigma_i')],
        E_H: theta_post[PARAM_NAMES.index('sigma_H')],
    }
    rho_vals = {
        E_A: theta_post[PARAM_NAMES.index('rho_A')],
        E_C: theta_post[PARAM_NAMES.index('rho_C')],
        E_P: 0.0,
        E_O: theta_post[PARAM_NAMES.index('rho_O')],
        E_Ys:theta_post[PARAM_NAMES.index('rho_Ys')],
        E_rp:theta_post[PARAM_NAMES.index('rho_rp')],
        E_i: 0.0,
        E_H: theta_post[PARAM_NAMES.index('rho_H')],
    }
    return theta_post, sigma_vals, rho_vals


def build_estimated_model(theta_post: np.ndarray) -> tuple:
    """Bygg og løs modellen fra posterior mean. Returner (T, R, stable)."""
    class Pt(P): pass
    for i, n in enumerate(PARAM_NAMES):
        setattr(Pt, n, float(theta_post[i]))
    setattr(Pt, 'sigma_A', SIGMA_A_FIXED)
    G0, G1, Psi, Pi = build_matrices_v3(Pt, theta_H=0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        T, R, d = bk_solve(G0, G1, Psi, Pi, verbose=False)
    return T, R, d['stable']


def compute_irf(T: np.ndarray, R: np.ndarray, shock_idx: int,
                shock_size: float, n_periods: int = 20) -> np.ndarray:
    """
    Beregn impulsrespons for ett sjokk.

    Returns:
        irf: (n_periods × NZ) array — log-%-avvik fra SS
    """
    irf = np.zeros((n_periods, NZ))
    state = R[:, shock_idx] * shock_size
    for t in range(n_periods):
        irf[t] = state
        state = T @ state
    return irf


def compute_fevd(T: np.ndarray, R: np.ndarray, sigma_vals: dict,
                 n_periods: int = 20) -> dict:
    """
    Beregn korrekt FEVD med σ²-normalisering.

    Returns:
        fevd_pct: {variabelnavn: {sjokkavn: [andel per horisont]}}
    """
    var_map = {'bnp':Y, 'pi':PI, 'rente':I_R, 'rer':RER, 'bolig':Q_H}
    cumvars = {vn: {} for vn in var_map}

    for sidx, sname in SHOCK_NAMES.items():
        sigma = sigma_vals[sidx]
        irf = compute_irf(T, R, sidx, sigma, n_periods)
        for vn, vidx in var_map.items():
            cumvars[vn][sname] = np.cumsum(irf[:, vidx]**2)

    fevd_pct = {}
    for vn in var_map:
        fevd_pct[vn] = {}
        for h in range(n_periods):
            tot = sum(cumvars[vn][s][h] for s in cumvars[vn])
            for s in cumvars[vn]:
                if s not in fevd_pct[vn]:
                    fevd_pct[vn][s] = []
                fevd_pct[vn][s].append(
                    round(float(cumvars[vn][s][h] / max(tot, 1e-15) * 100), 2)
                )
    return fevd_pct


def build_observation_system() -> tuple:
    """Bygg observasjonsmatrise H og målefeilkovarians Sv."""
    H = np.zeros((14, NZ))
    H[0,Y]=1; H[1,C]=1; H[2,INV]=1; H[3,4]=1; H[4,5]=1
    H[5,PI]=4; H[6,W]=1; H[7,I_R]=4; H[8,I_R]=4; H[9,RER]=1
    H[10,PO]=1; H[11,YS]=1; H[12,Q_H]=1; H[13,B_NW]=1
    sme = {
        'dy_obs':0.005, 'dc_obs':0.008, 'dinv_obs':0.015,
        'dx_obs':0.010, 'dm_obs':0.012, 'pi_obs':0.008,
        'dw_obs':0.004, 'i_R_obs':0.0005, 'i_3m_obs':0.0005,
        'ds_obs':0.010, 'dpO_obs':0.050, 'dyS_obs':0.006,
        'dh_obs':0.004, 'db_obs':0.002
    }
    Sv = np.diag([sme[n]**2 for n in OBS_NAMES])
    return H, Sv


def kalman_filter(T: np.ndarray, R: np.ndarray, H: np.ndarray,
                  Q: np.ndarray, Sv: np.ndarray,
                  Y_obs: np.ndarray) -> tuple:
    """
    Forward Kalman-filter.

    Returns:
        z_filt: (T × NZ) filtrerte tilstander
        innov:  (T × NZ) reduced-form innovasjoner i tilstandsrom
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
            Ht = H[~ms]; yo = yt[~ms]; Sv_t = Sv[np.ix_(~ms, ~ms)]
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
                z = zp; Pk = Pp
        else:
            z = zp; Pk = Pp
        z_filt[t] = z

    return z_filt, innov


def rts_smoother(T: np.ndarray, R: np.ndarray, Q: np.ndarray,
                 Y_obs: np.ndarray, H: np.ndarray,
                 Sv: np.ndarray) -> np.ndarray:
    """
    RTS (Rauch-Tung-Striebel) Kalman-smoother.

    Returns:
        z_smooth: (T × NZ) bakover-glattede tilstander
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

    z = np.zeros(NZ); Pk = P0.copy()
    for t in range(T_len):
        zp = T @ z; Pp = T @ Pk @ T.T + RQR
        z_pred[t] = zp; P_pred[t] = Pp
        yt = Y_obs[t]; ms = np.isnan(yt)
        if not ms.all():
            Ht = H[~ms]; yo = yt[~ms]; Sv_t = Sv[np.ix_(~ms, ~ms)]
            inn = yo - Ht @ zp; S = Ht @ Pp @ Ht.T + Sv_t; S = (S+S.T)/2
            try:
                Kg = Pp @ Ht.T @ np.linalg.inv(S)
                z = zp + Kg @ inn
                Pk = (np.eye(NZ) - Kg @ Ht) @ Pp; Pk = (Pk+Pk.T)/2
            except Exception:
                z = zp; Pk = Pp
        else:
            z = zp; Pk = Pp
        z_filt[t] = z; P_filt[t] = Pk

    # Bakover-glattingspass
    z_smooth = z_filt.copy(); P_smooth = P_filt.copy()
    for t in range(T_len - 2, -1, -1):
        Pp_next = P_pred[t+1]
        if np.abs(np.linalg.eigvals(Pp_next)).max() < 1e-12:
            continue
        L = P_filt[t] @ T.T @ np.linalg.pinv(Pp_next)
        z_smooth[t] = z_filt[t] + L @ (z_smooth[t+1] - z_pred[t+1])
        P_smooth[t] = P_filt[t] + L @ (P_smooth[t+1] - Pp_next) @ L.T

    return z_smooth


def shock_conditional_forecast(T: np.ndarray, R: np.ndarray,
                                sigma_vals: dict, rho_vals: dict,
                                last_state: np.ndarray,
                                last_innov: np.ndarray,
                                n_fcst: int = 16) -> dict:
    """
    Sjokk-betinget fremskrivning.

    Formel: z(T+h) = T^h·z_T + Σ_k ε_k,T · Σ_{s=0}^{h-1} ρ_k^(h-1-s) · T^s·r_k

    Returns:
        dict med 'baseline', 'conditional', 'shock_contrib', 'std_P'
    """
    # Identifiser aktive sjokk
    eps_T = {}
    for sidx in SHOCK_NAMES:
        r = R[:, sidx]
        eps = float(r @ last_innov) / (float(r @ r) + 1e-12)
        # Klipp til ±2σ for robusthet
        ratio = eps / sigma_vals[sidx]
        if abs(ratio) > 2.0:
            eps = np.sign(ratio) * 2.0 * sigma_vals[sidx]
        eps_T[sidx] = eps

    # Preberegn T^s matriser
    Ts = [np.eye(NZ)]
    for s in range(1, n_fcst):
        Ts.append(T @ Ts[-1])

    # Baseline: T^h · z_T
    baseline = np.zeros((n_fcst, NZ))
    s = last_state.copy()
    for h in range(n_fcst):
        s = T @ s
        baseline[h] = s

    # Sjokk-bidrag
    shock_contrib = {}
    total_shock = np.zeros((n_fcst, NZ))
    for sidx, sname in SHOCK_NAMES.items():
        r = R[:, sidx]
        rho = rho_vals[sidx]
        eps = eps_T[sidx]
        contrib = np.zeros((n_fcst, NZ))
        for h in range(n_fcst):
            accum = np.zeros(NZ)
            for s_idx in range(h):
                accum += (rho**(h-1-s_idx)) * (Ts[s_idx] @ r)
            contrib[h] = eps * accum
        shock_contrib[sname] = contrib
        total_shock += contrib

    conditional = baseline + total_shock

    # Prognosebånd P(h) fra P(0)=0
    Q_mat = np.zeros((NE, NE))
    for sidx, sigma in sigma_vals.items():
        Q_mat[sidx, sidx] = sigma**2
    RQR = R @ Q_mat @ R.T

    std_P = {vn: [] for vn in ['y','pi','i','rer','bolig']}
    var_std_map = {'y':Y, 'pi':PI, 'i':I_R, 'rer':RER, 'bolig':Q_H}
    Ph = np.zeros((NZ, NZ))
    for h in range(n_fcst):
        Ph = T @ Ph @ T.T + RQR
        for vn, vidx in var_std_map.items():
            std_P[vn].append(float(np.sqrt(max(Ph[vidx, vidx], 0))))

    return {
        'baseline': baseline,
        'conditional': conditional,
        'shock_contrib': shock_contrib,
        'eps_T': eps_T,
        'std_P': std_P,
    }


def print_fevd_table(fevd_pct: dict, horizons: list = [1, 4, 8, 20]):
    """Skriv ut FEVD-tabell for alle variabler."""
    for vn, vdata in fevd_pct.items():
        print(f"\nFEVD — {vn.upper()}")
        h_strs = [f"h={h:2d}" for h in horizons]
        print(f"  {'Sjokk':<16} " + "  ".join(f"{h:>6}" for h in h_strs))
        print("  " + "─"*52)
        for sname, vals in vdata.items():
            row = "  ".join(f"{vals[h-1]:>6.1f}%" for h in horizons)
            print(f"  {sname:<16} {row}")


# ── KJØRING ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='NEMO v3 analyse')
    parser.add_argument('--posterior', default='chain_v3_v2_posterior.json')
    parser.add_argument('--data',      default='nemo_data_faktisk_v2.csv')
    parser.add_argument('--n-irf',     type=int, default=20)
    parser.add_argument('--n-fevd',    type=int, default=20)
    parser.add_argument('--n-fcst',    type=int, default=16)
    parser.add_argument('--output',    default='analyse_resultater.json')
    args = parser.parse_args()

    print("Laster posterior...")
    theta_post, sigma_vals, rho_vals = load_posterior(args.posterior)

    print("Bygger og løser modell...")
    T, R, stable = build_estimated_model(theta_post)
    if not stable:
        raise RuntimeError("Modellen er ikke stabil — sjekk parametere.")
    print(f"  Stabil. max|eig(T)| = {np.abs(np.linalg.eigvals(T)).max():.6f}")

    print("Beregner IRF...")
    irf_results = {}
    for sidx, sname in SHOCK_NAMES.items():
        sigma = sigma_vals[sidx]
        irf = compute_irf(T, R, sidx, sigma, args.n_irf)
        irf_results[sname] = {
            vname: [round(float(irf[t, vidx] * 100), 4) for t in range(args.n_irf)]
            for vidx, vname in VAR_NAMES.items()
        }
    print(f"  Beregnet {len(irf_results)} IRF-er, {args.n_irf} perioder.")

    print("Beregner FEVD...")
    fevd_pct = compute_fevd(T, R, sigma_vals, args.n_fevd)
    print_fevd_table(fevd_pct)

    print("\nLaster data og kjører Kalman-filter...")
    obs_df = pd.read_csv(args.data, index_col=0, parse_dates=True)
    obs_pre  = obs_df[obs_df.index <= '2019-12-31'][OBS_NAMES]
    obs_post = obs_df[obs_df.index >= '2022-01-01'][OBS_NAMES]
    Y_comb   = np.concatenate([obs_pre.values, obs_post.values])
    dates    = list(obs_pre.index) + list(obs_post.index)

    H, Sv = build_observation_system()
    Q_mat = np.zeros((NE, NE))
    for sidx, sigma in sigma_vals.items():
        Q_mat[sidx, sidx] = sigma**2

    z_filt, innov = kalman_filter(T, R, H, Q_mat, Sv, Y_comb)
    last_state = z_filt[-1]
    last_innov = innov[-1]
    print(f"  Kalman-filter kjørt: {len(Y_comb)} observasjoner.")

    print("\nSjokk-betinget fremskrivning...")
    fcst = shock_conditional_forecast(
        T, R, sigma_vals, rho_vals,
        last_state, last_innov, args.n_fcst
    )
    print("  Aktive sjokk ved siste kvartal:")
    for sidx, sname in SHOCK_NAMES.items():
        eps = fcst['eps_T'][sidx]
        std = sigma_vals[sidx]
        print(f"    {sname:<15}: {eps/std:+.2f}σ")

    # ── Obs-rom-indekser ──────────────────────────────────────────────────────
    _DY = OBS_NAMES.index('dy_obs')   # 0
    _PI = OBS_NAMES.index('pi_obs')   # 5
    _IR = OBS_NAMES.index('i_R_obs')  # 7
    _DS = OBS_NAMES.index('ds_obs')   # 9
    _DH = OBS_NAMES.index('dh_obs')   # 12
    _OBS_KEYS = {_DY: 'y', _PI: 'pi', _IR: 'i', _DS: 'rer', _DH: 'bolig'}

    # ── Les demean-verdier for å konvertere tilbake til faktiske nivåer ───────
    _meta_path = os.path.join(os.path.dirname(os.path.abspath(args.data)), 'crosscheck_meta.json')
    _dmeans = {}
    if os.path.exists(_meta_path):
        try:
            with open(_meta_path) as _f:
                _dmeans = json.load(_f).get('demean_values', {})
        except Exception:
            pass
    _dm = {
        _DY: _dmeans.get('dy_obs',   0.0),
        _PI: _dmeans.get('pi_obs',   0.0),
        _IR: _dmeans.get('i_R_obs',  0.0),
        _DS: _dmeans.get('ds_obs',   0.0),
        _DH: _dmeans.get('dh_obs',   0.0),
    }
    # Rente skaleres: kvartalsnivå i pst. × 4 = % p.a.
    _IR_SCALE = 4.0

    # ── RTS-smoother → glattede tilstander ───────────────────────────────────
    print("\nKjører RTS-smoother...")
    z_smooth = rts_smoother(T, R, Q_mat, Y_comb, H, Sv)
    print(f"  RTS-smoother kjørt: {len(z_smooth)} perioder.")

    # ── hist_level: observerte verdier + demean-korreksjon ───────────────────
    def _safe(v, dm=0.0, scale=1.0):
        return round((float(v) + dm) * scale, 4) if not np.isnan(v) else None

    date_strs = [str(d.date()) if hasattr(d, 'date') else str(d) for d in dates]

    # ── Last NOK/EUR-nivå fra crosscheck_data.csv (ds_obs_level = ln(NOK/EUR)) ─
    _rer_levels = None
    _ck_csv = os.path.join(os.path.dirname(os.path.abspath(args.data)), 'crosscheck_data.csv')
    if not os.path.exists(_ck_csv):
        _ck_csv = 'crosscheck_data.csv'
    if os.path.exists(_ck_csv):
        try:
            _ck_df = pd.read_csv(_ck_csv, index_col=0, parse_dates=True)
            if 'ds_obs_level' in _ck_df.columns:
                _ck_pre  = _ck_df[_ck_df.index <= '2019-12-31']['ds_obs_level'].values
                _ck_post = _ck_df[_ck_df.index >= '2022-01-01']['ds_obs_level'].values
                _ck_rer  = np.concatenate([_ck_pre, _ck_post])
                _rer_levels = [
                    round(float(np.exp(v)), 4) if not np.isnan(v) else None
                    for v in _ck_rer
                ]
        except Exception:
            pass

    hist_level = {
        'dates': date_strs,
        'y':     [_safe(v, _dm[_DY])              for v in Y_comb[:, _DY]],
        'pi':    [_safe(v, _dm[_PI])              for v in Y_comb[:, _PI]],
        'i':     [_safe(v, _dm[_IR], _IR_SCALE)   for v in Y_comb[:, _IR]],
        'rer':   _rer_levels if _rer_levels is not None
                 else [_safe(v, _dm[_DS])          for v in Y_comb[:, _DS]],
        'bolig': [_safe(v, _dm[_DH])              for v in Y_comb[:, _DH]],
    }

    # ── forecast_level: tilstandsrom → obs-rom + demean-korreksjon ───────────
    _fcst_obs  = fcst['baseline']    @ H.T  # (n_fcst, 14)
    _fcst_cond = fcst['conditional'] @ H.T  # (n_fcst, 14)

    last_date = pd.Timestamp(dates[-1])
    fcst_dates = [
        str((last_date + pd.DateOffset(months=3 * (k + 1))).to_period('Q').to_timestamp().date())
        for k in range(args.n_fcst)
    ]

    def _fser(arr, col, dm=0.0, scale=1.0):
        return [round((float(v) + dm) * scale, 4) for v in arr[:, col]]

    forecast_level = {
        'dates': fcst_dates,
        'y':     _fser(_fcst_obs,  _DY, _dm[_DY]),
        'pi':    _fser(_fcst_obs,  _PI, _dm[_PI]),
        'i':     _fser(_fcst_obs,  _IR, _dm[_IR], _IR_SCALE),
        'rer':   _fser(_fcst_obs,  _DS, _dm[_DS]),
        'bolig': _fser(_fcst_obs,  _DH, _dm[_DH]),
        'y_bl':  _fser(_fcst_cond, _DY, _dm[_DY]),
        'pi_bl': _fser(_fcst_cond, _PI, _dm[_PI]),
        'i_bl':  _fser(_fcst_cond, _IR, _dm[_IR], _IR_SCALE),
    }

    # ── hist_decomp: sjokk-bidrag per periode via smoothede tilstander ────────
    print("Beregner historisk dekomposisjon...")
    T_hist = len(z_smooth)
    hist_decomp: dict = {}
    for obs_idx, vkey in _OBS_KEYS.items():
        h_row = H[obs_idx, :]
        shocks: dict = {}
        for sidx, sname in SHOCK_NAMES.items():
            r_col = R[:, sidx]
            contrib = []
            for t in range(1, T_hist):
                eps_t = z_smooth[t] - T @ z_smooth[t - 1]
                obs_c = float(h_row @ (r_col * eps_t[sidx]))
                contrib.append(round(obs_c, 6))
            shocks[sname] = contrib
        hist_decomp[vkey] = shocks
    print(f"  Historisk dekomposisjon: {T_hist - 1} perioder, {len(SHOCK_NAMES)} sjokk.")

    # ── Lagre ─────────────────────────────────────────────────────────────────
    results = {
        'irf':  irf_results,
        'fevd': fevd_pct,
        'forecast': {
            'conditional': fcst['conditional'].tolist(),
            'baseline':    fcst['baseline'].tolist(),
            'std_P':       fcst['std_P'],
        },
        'forecast_level': forecast_level,
        'hist_level':     hist_level,
        'hist_decomp':    hist_decomp,
        'active_shocks': {
            SHOCK_NAMES[k]: round(float(v/sigma_vals[k]), 3)
            for k, v in fcst['eps_T'].items()
        },
        'meta': {
            'eig_max':       round(float(np.abs(np.linalg.eigvals(T)).max()), 6),
            'n_states':      NZ,
            'n_shocks':      NE,
            'last_obs_date': date_strs[-1] if date_strs else None,
        }
    }
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResultater lagret: {args.output}")
