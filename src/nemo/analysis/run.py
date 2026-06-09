"""
[ARK/PL] CLI-entrypoint for NEMO-analyse.

Bruk:
    python -m nemo.analysis.run --posterior data/results/chain_kj41_prod_posterior.json
    python -m nemo.analysis.run --posterior ... --output analyse.json

Produserer en JSON-fil med IRF, FEVD, historisk dekomposisjon og prognose,
identisk med strukturen fra det gamle analyse.json-formatet.
"""

import argparse
import json
import logging
import os
import sys
from typing import List

import numpy as np
import pandas as pd

from nemo.analysis.irf import load_posterior, build_estimated_model, compute_all_irf
from nemo.analysis.fevd import compute_fevd, print_fevd_table
from nemo.analysis.decomposition import (
    OBS_NAMES, build_observation_system, kalman_filter, rts_smoother,
    compute_historical_decomposition,
)
from nemo.analysis.forecast import shock_conditional_forecast
from nemo.model.equations import Y, PI, I_R, RER, Q_H, NZ

logger = logging.getLogger(__name__)

_OBS_KEYS = {
    OBS_NAMES.index('dy_obs'):  'y',
    OBS_NAMES.index('pi_obs'):  'pi',
    OBS_NAMES.index('i_R_obs'): 'i',
    OBS_NAMES.index('ds_obs'):  'rer',
    OBS_NAMES.index('dh_obs'):  'bolig',
}
_DY = OBS_NAMES.index('dy_obs')
_PI = OBS_NAMES.index('pi_obs')
_IR = OBS_NAMES.index('i_R_obs')
_DS = OBS_NAMES.index('ds_obs')
_DH = OBS_NAMES.index('dh_obs')
_IR_SCALE = 4.0   # kvartalsnivå → prosent per år


def _load_demean(data_path: str) -> dict:
    """Last demean-verdier fra nemo_demean_kpi_jae.json (eller crosscheck_meta.json)."""
    data_dir = os.path.dirname(os.path.abspath(data_path))
    for fname in ('nemo_demean_kpi_jae.json', 'crosscheck_meta.json'):
        candidate = os.path.join(data_dir, fname)
        if os.path.exists(candidate):
            with open(candidate) as f:
                meta = json.load(f)
            return meta.get('demean_values', meta)
    return {}


def _load_rer_levels(data_path: str) -> List:
    """Last NOK/EUR-nivå fra crosscheck_data.csv hvis tilgjengelig."""
    data_dir = os.path.dirname(os.path.abspath(data_path))
    csv = os.path.join(data_dir, 'crosscheck_data.csv')
    if not os.path.exists(csv):
        return []
    try:
        df = pd.read_csv(csv, index_col=0, parse_dates=True)
        if 'ds_obs_level' not in df.columns:
            return []
        pre  = df[df.index <= '2019-12-31']['ds_obs_level'].values
        post = df[df.index >= '2022-01-01']['ds_obs_level'].values
        combined = np.concatenate([pre, post])
        return [
            round(float(np.exp(v)), 4) if not np.isnan(v) else None
            for v in combined
        ]
    except Exception:
        return []


def run(
    posterior_path: str,
    data_path: str,
    output_path: str,
    n_irf: int = 20,
    n_fevd: int = 20,
    n_fcst: int = 16,
    verbose: bool = True,
) -> dict:
    """
    Kjør full NEMO-analyse og lagre til JSON.

    Returns:
        Resultat-dict (identisk med lagret JSON).
    """
    if verbose:
        print("Laster posterior...")
    theta_post, sigma_vals, rho_vals = load_posterior(posterior_path)

    if verbose:
        print("Bygger og løser modell...")
    T, R, stable = build_estimated_model(theta_post)
    if not stable:
        raise RuntimeError("Modellen er ikke stabil — sjekk parametere.")
    eig_max = float(np.abs(np.linalg.eigvals(T)).max())
    if verbose:
        print(f"  Stabil. max|eig(T)| = {eig_max:.6f}")

    if verbose:
        print("Beregner IRF...")
    irf_results = compute_all_irf(T, R, sigma_vals, n_irf)

    if verbose:
        print("Beregner FEVD...")
    fevd_pct = compute_fevd(T, R, sigma_vals, n_fevd)
    if verbose:
        print_fevd_table(fevd_pct)

    if verbose:
        print("\nLaster data og kjører Kalman-filter...")
    obs_df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    obs_pre  = obs_df[obs_df.index <= '2019-12-31'][OBS_NAMES]
    obs_post = obs_df[obs_df.index >= '2022-01-01'][OBS_NAMES]
    Y_comb   = np.concatenate([obs_pre.values, obs_post.values])
    dates    = list(obs_pre.index) + list(obs_post.index)

    H, Sv = build_observation_system()
    Q_mat = np.zeros((R.shape[1], R.shape[1]))
    for sidx, sigma in sigma_vals.items():
        Q_mat[sidx, sidx] = sigma ** 2

    z_filt, innov = kalman_filter(T, R, H, Q_mat, Sv, Y_comb)
    last_state = z_filt[-1]
    last_innov = innov[-1]
    if verbose:
        print(f"  Kalman-filter kjørt: {len(Y_comb)} observasjoner.")

    if verbose:
        print("\nSjokk-betinget fremskrivning...")
    fcst = shock_conditional_forecast(
        T, R, sigma_vals, rho_vals, last_state, last_innov, n_fcst
    )
    if verbose:
        from nemo.analysis.irf import SHOCK_NAMES
        print("  Aktive sjokk ved siste kvartal:")
        for sidx, sname in SHOCK_NAMES.items():
            eps = fcst['eps_T'][sidx]
            std = sigma_vals[sidx]
            print(f"    {sname:<15}: {eps / std:+.2f}σ")

    if verbose:
        print("\nKjører RTS-smoother...")
    z_smooth = rts_smoother(T, R, Q_mat, Y_comb, H, Sv)
    if verbose:
        print(f"  RTS-smoother kjørt: {len(z_smooth)} perioder.")

    # ── Obs-rom-konvertering ──────────────────────────────────────────────────
    dmeans = _load_demean(data_path)
    dm = {
        _DY: dmeans.get('dy_obs',  0.0),
        _PI: dmeans.get('pi_obs',  0.0),
        _IR: dmeans.get('i_R_obs', 0.0),
        _DS: dmeans.get('ds_obs',  0.0),
        _DH: dmeans.get('dh_obs',  0.0),
    }

    def _safe(v, dm_val=0.0, scale=1.0):
        return round((float(v) + dm_val) * scale, 4) if not np.isnan(v) else None

    date_strs = [
        str(d.date()) if hasattr(d, 'date') else str(d) for d in dates
    ]

    rer_levels = _load_rer_levels(data_path)
    hist_level = {
        'dates': date_strs,
        'y':     [_safe(v, dm[_DY])             for v in Y_comb[:, _DY]],
        'pi':    [_safe(v, dm[_PI])             for v in Y_comb[:, _PI]],
        'i':     [_safe(v, dm[_IR], _IR_SCALE)  for v in Y_comb[:, _IR]],
        'rer':   rer_levels if rer_levels else [_safe(v, dm[_DS]) for v in Y_comb[:, _DS]],
        'bolig': [_safe(v, dm[_DH])             for v in Y_comb[:, _DH]],
    }

    last_date = pd.Timestamp(dates[-1])
    fcst_dates = [
        str((last_date + pd.DateOffset(months=3 * (k + 1))).to_period('Q').to_timestamp().date())
        for k in range(n_fcst)
    ]

    _fcst_obs  = fcst['baseline']    @ H.T
    _fcst_cond = fcst['conditional'] @ H.T

    def _fser(arr, col, dm_val=0.0, scale=1.0):
        return [round((float(v) + dm_val) * scale, 4) for v in arr[:, col]]

    forecast_level = {
        'dates': fcst_dates,
        'y':     _fser(_fcst_obs,  _DY, dm[_DY]),
        'pi':    _fser(_fcst_obs,  _PI, dm[_PI]),
        'i':     _fser(_fcst_obs,  _IR, dm[_IR], _IR_SCALE),
        'rer':   _fser(_fcst_obs,  _DS, dm[_DS]),
        'bolig': _fser(_fcst_obs,  _DH, dm[_DH]),
        'y_bl':  _fser(_fcst_cond, _DY, dm[_DY]),
        'pi_bl': _fser(_fcst_cond, _PI, dm[_PI]),
        'i_bl':  _fser(_fcst_cond, _IR, dm[_IR], _IR_SCALE),
    }

    if verbose:
        print("Beregner historisk dekomposisjon...")
    hist_decomp = compute_historical_decomposition(T, R, z_smooth, H, _OBS_KEYS)

    from nemo.analysis.irf import SHOCK_NAMES
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
            SHOCK_NAMES[k]: round(float(v / sigma_vals[k]), 3)
            for k, v in fcst['eps_T'].items()
        },
        'meta': {
            'eig_max':       round(eig_max, 6),
            'n_states':      NZ,
            'n_shocks':      R.shape[1],
            'last_obs_date': date_strs[-1] if date_strs else None,
        },
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    if verbose:
        print(f"\nResultater lagret: {output_path}")

    return results


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description='NEMO v3 analyse')
    parser.add_argument('--posterior', required=True,
                        help='Sti til posterior JSON (f.eks. chain_kj41_prod_posterior.json)')
    parser.add_argument('--data',
                        default='data/processed/nemo_data_kpi_jae.csv')
    parser.add_argument('--n-irf',  type=int, default=20)
    parser.add_argument('--n-fevd', type=int, default=20)
    parser.add_argument('--n-fcst', type=int, default=16)
    parser.add_argument('--output',
                        default='data/results/analyse_resultater.json')
    args = parser.parse_args()

    run(
        posterior_path=args.posterior,
        data_path=args.data,
        output_path=args.output,
        n_irf=args.n_irf,
        n_fevd=args.n_fevd,
        n_fcst=args.n_fcst,
    )


if __name__ == '__main__':
    main()
