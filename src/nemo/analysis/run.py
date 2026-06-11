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
_IR_SCALE = 4.0   # kvartalsrente → annualisert
# Observasjonsdataene er demeanede ANDELER (f.eks. i_R_obs+demean ≈ 0,01 per kvartal).
# Dashbordet og IRF-ene (compute_all_irf ×100) er i PROSENT, så nivåseriene må også
# skaleres med 100. Styringsrenten annualiseres i tillegg (×4). Uten dette vises
# f.eks. styringsrenten som «0,0 %» i stedet for «4,0 %» (QA-funn: datastørrelser).
_PCT = 100.0


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


def _build_diagnostics(posterior_path: str, irf_results: dict) -> dict:
    """
    Samle diagnostikk-data til dashbordet: MCMC-konvergens (sammendrag), prior/posterior-
    tabell, måleresfeil-struktur og NB Memo 3/2024-sammenligning for pengepolitisk sjokk.

    Robust mot manglende filer (CLAUDE.md: fil-/API-feil skal ikke krasje pipelinen).
    """
    diag: dict = {}
    try:
        with open(posterior_path) as f:
            summ = json.load(f)['summary']
    except Exception:
        summ = {}

    # 1) Konvergens-sammendrag fra meta-filen (..._posterior.json → ..._meta.json).
    #    Per-parameter ESS/R-hat og trace plots krever lagrede MCMC-trekk (ikke tilgjengelig).
    conv: dict = {}
    meta_path = posterior_path.replace('_posterior.json', '_meta.json')
    if os.path.exists(meta_path):
        try:
            m = json.load(open(meta_path))
            conv = {
                'psrf':     round(float(m.get('psrf_final', float('nan'))), 4),
                'ess_min':  int(round(float(m.get('ess_min', 0)))),
                'n_draws':  int(m.get('n_production', 0)),
                'burnin':   int(m.get('burnin', 0)),
                'acc_rate': round(float(m.get('acc_rate', 0)), 3),
            }
        except Exception:
            conv = {}
    diag['convergence'] = conv

    # 2) Prior/posterior-tabell.
    try:
        from nemo.estimation.mcmc import PARAM_PRIORS
    except Exception:
        PARAM_PRIORS = {}

    def _prior_str(name: str) -> str:
        spec = PARAM_PRIORS.get(name)
        if not spec:
            return '—'
        return f'{spec[0]}({spec[1]:g}, {spec[2]:g})'

    estimated = [
        {
            'name':  name,
            'prior': _prior_str(name),
            'mean':  round(float(q['mean']), 4),
            'lo':    round(float(q.get('q025', float('nan'))), 4),
            'hi':    round(float(q.get('q975', float('nan'))), 4),
        }
        for name, q in summ.items()
    ]
    fixed = [
        {'name': 'sigma_rp', 'value': 0.006,  'note': 'K&M (PE-godkjent 2026-05-24)'},
        {'name': 'h_c',      'value': 0.938,  'note': 'K&M (C2 Alt A)'},
        {'name': 'sigma_A',  'value': 0.006,  'note': 'K&M'},
        {'name': 'phi_PQ',   'value': 150.0,  'note': 'kj41 priskostnad'},
        {'name': 'phi_u',    'value': 0.2192, 'note': 'K&M Tabell 8'},
    ]
    diag['prior_posterior'] = {'estimated': estimated, 'fixed': fixed}

    # 3) Måleresfeil-struktur (standardavvik per observabel). 14 obs > 13 sjokk er
    #    veldefinert nettopp fordi alle observasjoner har måleresfeil.
    try:
        from nemo.analysis.decomposition import _SME
        diag['meas_error'] = [{'series': k, 'sd': v} for k, v in _SME.items()]
    except Exception:
        diag['meas_error'] = []

    # 5) Parameteroversikt med pedagogiske navn. Effektiv kj41-verdi = posterior mean for
    #    estimerte; ellers kj41-kalibrering; ellers Parameters-default. Gruppert for lesbarhet.
    _PDESC = [
        # (navn, gruppe, pedagogisk beskrivelse)
        ('rho_A',  'Sjokk – persistens', 'Teknologisjokk (TFP): tregheten'),
        ('rho_C',  'Sjokk – persistens', 'Konsum-/preferansesjokk: tregheten'),
        ('rho_O',  'Sjokk – persistens', 'Oljeprissjokk: tregheten'),
        ('rho_Ys', 'Sjokk – persistens', 'Utenlandsk etterspørselssjokk: tregheten'),
        ('rho_rp', 'Sjokk – persistens', 'Risikopremiesjokk: tregheten'),
        ('rho_H',  'Sjokk – persistens', 'Boligsjokk: tregheten'),
        ('rho_s',  'Sjokk – persistens', 'Valutakurssjokk: tregheten'),
        ('sigma_A',  'Sjokk – størrelse', 'Teknologisjokk: standardavvik'),
        ('sigma_C',  'Sjokk – størrelse', 'Konsumsjokk: standardavvik'),
        ('sigma_O',  'Sjokk – størrelse', 'Oljeprissjokk: standardavvik'),
        ('sigma_Ys', 'Sjokk – størrelse', 'Utenlandsk etterspørselssjokk: standardavvik'),
        ('sigma_rp', 'Sjokk – størrelse', 'Risikopremiesjokk: standardavvik'),
        ('sigma_i',  'Sjokk – størrelse', 'Pengepolitisk sjokk: standardavvik'),
        ('sigma_P',  'Sjokk – størrelse', 'Prismarkup-sjokk: standardavvik'),
        ('sigma_H',  'Sjokk – størrelse', 'Boligsjokk: standardavvik'),
        ('psi_R',  'Pengepolitisk regel', 'Rentegltting (treghet i styringsrenten)'),
        ('psi_R2', 'Pengepolitisk regel', 'AR(2)-ledd i renteregelen (0 = ren AR(1))'),
        ('psi_P1', 'Pengepolitisk regel', 'Inflasjonsvekt i renteregelen'),
        ('psi_Y',  'Pengepolitisk regel', 'Produksjonsgap-vekt i renteregelen'),
        ('psi_PL', 'Pengepolitisk regel', 'Prisnivåmål-vekt (0 = rent inflasjonsmål)'),
        ('phi_PQ', 'Nominelle rigiditeter', 'Prisjusteringskostnad innenlands (Rotemberg)'),
        ('phi_PM', 'Nominelle rigiditeter', 'Prisjusteringskostnad import'),
        ('phi_W',  'Nominelle rigiditeter', 'Lønnsjusteringskostnad'),
        ('gamma_p','Nominelle rigiditeter', 'Bakoverskuende prisindeksering'),
        ('kappa_M','Nominelle rigiditeter', 'Helning i importpris-Phillipskurven'),
        ('phi_I1', 'Realøkonomiske rigiditeter', 'Investeringsjusteringskostnad (nivå)'),
        ('phi_I2', 'Realøkonomiske rigiditeter', 'Investeringsjusteringskostnad (endring)'),
        ('phi_H1', 'Realøkonomiske rigiditeter', 'Boliginvesteringskostnad (nivå)'),
        ('phi_H2', 'Realøkonomiske rigiditeter', 'Boliginvesteringskostnad (endring)'),
        ('phi_u',  'Realøkonomiske rigiditeter', 'Kapitalutnyttelses-elastisitet'),
        ('phi_O',  'Realøkonomiske rigiditeter', 'Olje→valutakurs-kanal i UIP'),
        ('h_c',    'Preferanser / teknologi', 'Konsumvane (habit)'),
        ('beta',   'Preferanser / teknologi', 'Diskonteringsfaktor'),
        ('phi_L',  'Preferanser / teknologi', 'Invers Frisch-elastisitet (arbeidstilbud)'),
        ('alpha_K','Preferanser / teknologi', 'Kapitalandel i produksjonen'),
        ('alpha_l','Preferanser / teknologi', 'Arbeidskraftandel i produksjonen'),
        ('delta',  'Preferanser / teknologi', 'Kapitalslitasje (depresiering)'),
        ('CY',     'Steady state-andeler', 'Konsum som andel av BNP'),
        ('IY',     'Steady state-andeler', 'Investering som andel av BNP'),
        ('GY',     'Steady state-andeler', 'Offentlig etterspørsel som andel av BNP'),
        ('XY',     'Steady state-andeler', 'Eksport som andel av BNP'),
        ('MY',     'Steady state-andeler', 'Import som andel av BNP'),
        ('pi_ss',  'Steady state-nivåer', 'Inflasjonsmål (kvartalsvis bruttofaktor)'),
    ]
    _PSYM = {
        'rho_A': 'ρ_A', 'rho_C': 'ρ_C', 'rho_O': 'ρ_O', 'rho_Ys': 'ρ_Y*',
        'rho_rp': 'ρ_rp', 'rho_H': 'ρ_H', 'rho_s': 'ρ_s',
        'sigma_A': 'σ_A', 'sigma_C': 'σ_C', 'sigma_O': 'σ_O', 'sigma_Ys': 'σ_Y*',
        'sigma_rp': 'σ_rp', 'sigma_i': 'σ_i', 'sigma_P': 'σ_P', 'sigma_H': 'σ_H',
        'psi_R': 'ψ_R', 'psi_R2': 'ψ_R2', 'psi_P1': 'ψ_π', 'psi_Y': 'ψ_y', 'psi_PL': 'ψ_PL',
        'phi_PQ': 'φ_PQ', 'phi_PM': 'φ_PM', 'phi_W': 'φ_W', 'gamma_p': 'γ_p', 'kappa_M': 'κ_M',
        'phi_I1': 'φ_I1', 'phi_I2': 'φ_I2', 'phi_H1': 'φ_H1', 'phi_H2': 'φ_H2',
        'phi_u': 'φ_u', 'phi_O': 'φ_O', 'h_c': 'h_c', 'beta': 'β', 'phi_L': 'φ_L',
        'alpha_K': 'α_K', 'alpha_l': 'α_l', 'delta': 'δ',
        'CY': 'C/Y', 'IY': 'I/Y', 'GY': 'G/Y', 'XY': 'X/Y', 'MY': 'M/Y', 'pi_ss': 'π̄',
    }
    try:
        from nemo.model.parameters import Parameters as _P
        _kj41_fixed = {'phi_PQ': 150.0, 'sigma_rp': 0.006, 'sigma_A': 0.006,
                       'h_c': 0.938, 'phi_u': 0.2192, 'psi_R2': 0.0}

        def _eff(name):
            if name in summ:        return float(summ[name]['mean'])
            if name in _kj41_fixed: return _kj41_fixed[name]
            return getattr(_P, name, None)

        diag['param_overview'] = [
            {'name': nm, 'symbol': _PSYM.get(nm, nm), 'group': grp, 'label': desc,
             'value': (round(float(_eff(nm)), 5) if _eff(nm) is not None else None),
             'est': nm in summ}
            for nm, grp, desc in _PDESC
        ]
    except Exception:
        diag['param_overview'] = []

    # 4) NB Memo 3/2024-sammenligning, pengepolitisk sjokk, normalisert til +1 pp topp i I_R.
    try:
        data_dir = os.path.dirname(os.path.abspath(posterior_path))
        nb = json.load(open(os.path.join(data_dir, 'B5_nb_benchmark.json')))['nb_referanse']
        mp = irf_results.get('Pengepol.', {})
        peak = max((abs(x) for x in mp.get('Styringsrente', [1.0])), default=1.0) or 1.0
        scale = 1.0 / peak
        vmap = {'Y': 'BNP-gap', 'PI': 'KPI-inflasjon', 'I_R': 'Styringsrente', 'RER': 'RER-gap'}
        rows = []
        for v, key in vmap.items():
            arr = mp.get(key, [])
            row = {'var': v}
            for q in (1, 4, 8, 12):
                row[f'q{q}'] = {
                    'kj41': round(arr[q - 1] * scale, 2) if len(arr) >= q else None,
                    'nb':   nb.get(v, {}).get(f'q{q}'),
                }
            rows.append(row)
        diag['nb_irf'] = {'rows': rows, 'note': 'Normalisert til +1 pp topp i styringsrenten.'}
    except Exception:
        diag['nb_irf'] = {}

    return diag


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
    param_means, sigma_vals, rho_vals = load_posterior(posterior_path)

    if verbose:
        print("Bygger og løser modell...")
    T, R, stable = build_estimated_model(param_means)
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
    # kj41 estimerte på kjerne-KPI (pi_core_obs) i PI-observasjonsraden — bruk samme
    # serie her slik at Kalman-filter og historisk dekomposisjon er konsistent med
    # estimeringen. Samme posisjon (indeks 5) som pi_obs, så H-matrisen treffer riktig.
    obs_cols = [
        'pi_core_obs' if c == 'pi_obs' and 'pi_core_obs' in obs_df.columns else c
        for c in OBS_NAMES
    ]
    obs_pre  = obs_df[obs_df.index <= '2019-12-31'][obs_cols]
    obs_post = obs_df[obs_df.index >= '2022-01-01'][obs_cols]

    # Trim haleskvartaler der HELE observasjonsraden mangler. Forecast-origo må være
    # et faktisk observert kvartal; ellers blir siste Kalman-innovasjon ≈ 0, og den
    # betingede prognosen blir identisk med den ubetingede (jf. QA-funn A).
    def _drop_trailing_allnan(frame: pd.DataFrame) -> pd.DataFrame:
        valid = ~frame.isna().all(axis=1)
        if valid.any():
            last = int(np.where(valid.values)[0][-1])
            return frame.iloc[:last + 1]
        return frame
    obs_post = _drop_trailing_allnan(obs_post)

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
        'y':     [_safe(v, dm[_DY], _PCT)              for v in Y_comb[:, _DY]],
        'pi':    [_safe(v, dm[_PI], _PCT)              for v in Y_comb[:, _PI]],
        'i':     [_safe(v, dm[_IR], _IR_SCALE * _PCT)  for v in Y_comb[:, _IR]],
        'rer':   rer_levels if rer_levels else [_safe(v, dm[_DS], _PCT) for v in Y_comb[:, _DS]],
        'bolig': [_safe(v, dm[_DH], _PCT)              for v in Y_comb[:, _DH]],
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
        'y':     _fser(_fcst_obs,  _DY, dm[_DY], _PCT),
        'pi':    _fser(_fcst_obs,  _PI, dm[_PI], _PCT),
        'i':     _fser(_fcst_obs,  _IR, dm[_IR], _IR_SCALE * _PCT),
        'rer':   _fser(_fcst_obs,  _DS, dm[_DS], _PCT),
        'bolig': _fser(_fcst_obs,  _DH, dm[_DH], _PCT),
        'y_bl':  _fser(_fcst_cond, _DY, dm[_DY], _PCT),
        'pi_bl': _fser(_fcst_cond, _PI, dm[_PI], _PCT),
        'i_bl':  _fser(_fcst_cond, _IR, dm[_IR], _IR_SCALE * _PCT),
    }

    if verbose:
        print("Beregner historisk dekomposisjon...")
    hist_decomp = compute_historical_decomposition(T, R, z_smooth, H, _OBS_KEYS)

    # ── 4-kvartalersvekst (y/y) for Oversikt/Prognose ─────────────────────────
    # dy_obs/ds_obs/dh_obs er kvartalsvise log-diff; pi_core_obs er ×4-annualisert
    # (→ /4 gir kvartalsvis). y/y = rullende sum av 4 kvartalsvise log-diff, pluss
    # 4× kvartalssnittet (demean), ×100. Beregnes på sammenhengende kalender (full
    # obs_df, inkl. COVID-kvartaler som faktisk har data) + prognosen, slik at
    # vinduene blir korrekte. Styringsrenten er et nivå og forblir uendret.
    # Historikk-fanen beholder kvartalsserier (hist_level) så nivå og dekomponering
    # er konsistente; Oversikt/Prognose bruker y/y-blokken under.
    def _yoy(q_hist: pd.Series, q_fcst, mean_q: float):
        idx  = list(q_hist.index) + [pd.Timestamp(d) for d in fcst_dates]
        vals = list(np.asarray(q_hist.values, dtype=float)) + [float(v) for v in q_fcst]
        s = (pd.Series(vals, index=idx).rolling(4).sum() + 4.0 * mean_q) * 100.0
        def g(d):
            v = s.get(pd.Timestamp(d))
            return round(float(v), 4) if v is not None and np.isfinite(v) else None
        return [g(d) for d in dates], [g(d) for d in fcst_dates]

    _pi_col = 'pi_core_obs' if 'pi_core_obs' in obs_df.columns else 'pi_obs'
    oy_h, oy_f = _yoy(obs_df['dy_obs'],          _fcst_obs[:, _DY],          dm[_DY])
    op_h, op_f = _yoy(obs_df[_pi_col] / 4.0,     _fcst_obs[:, _PI] / 4.0,    dm[_PI] / 4.0)
    or_h, or_f = _yoy(obs_df['ds_obs'],          _fcst_obs[:, _DS],          dm[_DS])
    ob_h, ob_f = _yoy(obs_df['dh_obs'],          _fcst_obs[:, _DH],          dm[_DH])
    _,    oy_fbl = _yoy(obs_df['dy_obs'],        _fcst_cond[:, _DY],         dm[_DY])
    _,    op_fbl = _yoy(obs_df[_pi_col] / 4.0,   _fcst_cond[:, _PI] / 4.0,   dm[_PI] / 4.0)

    overview_hist = {'dates': date_strs, 'y': oy_h, 'pi': op_h, 'i': hist_level['i'],
                     'rer': or_h, 'bolig': ob_h}
    overview_fcst = {'dates': fcst_dates, 'y': oy_f, 'pi': op_f, 'i': forecast_level['i'],
                     'rer': or_f, 'bolig': ob_f,
                     'y_bl': oy_fbl, 'pi_bl': op_fbl, 'i_bl': forecast_level['i_bl']}

    diagnostics = _build_diagnostics(posterior_path, irf_results)

    from nemo.analysis.irf import SHOCK_NAMES

    # ── IRF-normalisering per sjokk (tolkbare sjokkstørrelser) ────────────────
    # Pengepolitisk sjokk normaliseres til +1 pp topp i styringsrenten (NB-konvensjon);
    # øvrige sjokk til en fast innovasjonsstørrelse i sjokkets egne enheter. I en lineær
    # modell skalerer responsen proporsjonalt: skala = mål / σ (1-std-IRF → mål-IRF).
    _irf_targets = {
        'TFP':          (0.007, '+0,7 %'),
        'Konsum':       (0.01,  '+1 %'),
        'Prismarkup':   (0.003, '+0,3 pp'),
        'Oljepris':     (0.10,  '+10 %'),
        'Ettersp.':     (0.01,  '+1 %'),
        'Risikopremie': (0.01,  '+1 pp'),
        'Bolig':        (0.10,  '+10 %'),
    }
    irf_norm = {}
    for sidx, sname in SHOCK_NAMES.items():
        if sname == 'Pengepol.':
            rate = irf_results.get(sname, {}).get('Styringsrente', [])
            peak = max((abs(x) for x in rate), default=0.0)
            irf_norm[sname] = {'scale': round(1.0 / peak, 6) if peak > 1e-9 else 1.0,
                               'label': '+1 pp'}
        elif sname in _irf_targets:
            t, lab = _irf_targets[sname]
            sig = sigma_vals.get(sidx, 0.0)
            irf_norm[sname] = {'scale': round(t / sig, 6) if sig > 1e-12 else 1.0,
                               'label': lab}
        else:
            irf_norm[sname] = {'scale': 1.0, 'label': '1 std'}

    results = {
        'irf':  irf_results,
        'irf_norm': irf_norm,
        'fevd': fevd_pct,
        'forecast': {
            'conditional': fcst['conditional'].tolist(),
            'baseline':    fcst['baseline'].tolist(),
            'std_P':       fcst['std_P'],
        },
        'forecast_level': forecast_level,
        'hist_level':     hist_level,
        'hist_decomp':    hist_decomp,
        'overview_hist':  overview_hist,
        'overview_fcst':  overview_fcst,
        'diagnostics':    diagnostics,
        'active_shocks': {
            SHOCK_NAMES[k]: round(float(v / sigma_vals[k]), 3)
            for k, v in fcst['eps_T'].items()
        },
        'meta': {
            'eig_max':       round(eig_max, 6),
            'n_states':      int(T.shape[0]),
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
