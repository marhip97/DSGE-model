"""
[QA/ARK] Røyktest av nemo.analysis-pakken.

Verifiserer at alle moduler eksporterer forventede symboler og at
kjerneberegningene (IRF, FEVD, Kalman, RTS, prognose) kjøres uten feil
mot kalibrert modell.
"""

import numpy as np
import pytest

from nemo.analysis import (
    PARAM_NAMES, SHOCK_NAMES, VAR_NAMES, OBS_NAMES,
    build_estimated_model, compute_irf, compute_all_irf,
    compute_fevd, build_observation_system,
    kalman_filter, rts_smoother, compute_historical_decomposition,
    shock_conditional_forecast,
)
from nemo.model.equations import NZ, NE, Y, PI, I_R, RER, Q_H
from nemo.model.parameters import Parameters


@pytest.fixture(scope="module")
def modell(kalibrert_modell):
    T, R, _ = kalibrert_modell
    return T, R


def test_compute_irf_shape(modell):
    T, R = modell
    irf = compute_irf(T, R, 0, 0.01, n_periods=12)
    assert irf.shape == (12, NZ)


def test_compute_all_irf_keys(modell):
    T, R = modell
    from nemo.model.equations import E_A
    sigma_vals = {k: 0.01 for k in SHOCK_NAMES}
    irfs = compute_all_irf(T, R, sigma_vals, n_periods=8)
    assert set(irfs.keys()) == set(SHOCK_NAMES.values())
    for sname, vdict in irfs.items():
        assert set(vdict.keys()) == set(VAR_NAMES.values())
        for vname, vals in vdict.items():
            assert len(vals) == 8


def test_compute_fevd_sums(modell):
    T, R = modell
    sigma_vals = {k: 0.01 for k in SHOCK_NAMES}
    fevd = compute_fevd(T, R, sigma_vals, n_periods=8)
    for vn, vdata in fevd.items():
        for h_idx in range(8):
            total = sum(vals[h_idx] for vals in vdata.values())
            assert abs(total - 100.0) < 2.0, f"{vn} h={h_idx+1}: sum={total:.2f}"


def test_build_observation_system():
    H, Sv = build_observation_system()
    assert H.shape == (14, NZ)
    assert Sv.shape == (14, 14)
    assert (np.diag(Sv) > 0).all()


def test_kalman_filter(modell):
    T, R = modell
    H, Sv = build_observation_system()
    sigma_vals = {k: 0.01 for k in SHOCK_NAMES}
    Q = np.zeros((NE, NE))
    for idx, s in sigma_vals.items():
        Q[idx, idx] = s ** 2

    rng = np.random.default_rng(42)
    n = 40
    Y_obs = rng.standard_normal((n, 14)) * 0.01
    z_filt, innov = kalman_filter(T, R, H, Q, Sv, Y_obs)
    assert z_filt.shape == (n, NZ)
    assert innov.shape == (n, NZ)
    assert not np.any(np.isnan(z_filt))


def test_rts_smoother(modell):
    T, R = modell
    H, Sv = build_observation_system()
    sigma_vals = {k: 0.01 for k in SHOCK_NAMES}
    Q = np.zeros((NE, NE))
    for idx, s in sigma_vals.items():
        Q[idx, idx] = s ** 2

    rng = np.random.default_rng(43)
    n = 30
    Y_obs = rng.standard_normal((n, 14)) * 0.01
    z_smooth = rts_smoother(T, R, Q, Y_obs, H, Sv)
    assert z_smooth.shape == (n, NZ)
    assert not np.any(np.isnan(z_smooth))


def test_historical_decomposition(modell):
    T, R = modell
    H, Sv = build_observation_system()
    sigma_vals = {k: 0.01 for k in SHOCK_NAMES}
    Q = np.zeros((NE, NE))
    for idx, s in sigma_vals.items():
        Q[idx, idx] = s ** 2

    rng = np.random.default_rng(44)
    n = 20
    Y_obs = rng.standard_normal((n, 14)) * 0.01
    z_smooth = rts_smoother(T, R, Q, Y_obs, H, Sv)

    from nemo.analysis.decomposition import OBS_NAMES
    obs_keys = {
        OBS_NAMES.index('dy_obs'):  'y',
        OBS_NAMES.index('pi_obs'):  'pi',
    }
    decomp = compute_historical_decomposition(T, R, z_smooth, H, obs_keys)
    assert set(decomp.keys()) == {'y', 'pi'}
    for vkey, shocks in decomp.items():
        assert set(shocks.keys()) == set(SHOCK_NAMES.values())
        for s, contrib in shocks.items():
            assert len(contrib) == n - 1


def test_shock_conditional_forecast(modell):
    T, R = modell
    sigma_vals = {k: 0.01 for k in SHOCK_NAMES}
    rho_vals   = {k: 0.80 for k in SHOCK_NAMES}
    last_state = np.zeros(NZ)
    last_innov = np.zeros(NZ)

    result = shock_conditional_forecast(T, R, sigma_vals, rho_vals, last_state, last_innov, n_fcst=8)
    assert result['baseline'].shape == (8, NZ)
    assert result['conditional'].shape == (8, NZ)
    for vn in ['y', 'pi', 'i', 'rer', 'bolig']:
        assert len(result['std_P'][vn]) == 8
