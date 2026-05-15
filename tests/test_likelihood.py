"""
[NUM/ARK] Test av Kalman-filter og likelihood-beregning.

Dekker:
- Kalman-filter returnerer endelig log-likelihood på simulerte data
- COVID-gap: kalman_hull = kf_pre + kf_post (ingen blødning mellom blokker)
- log_posterior er endelig i kalibrasjonspunktet
- NaN-robusthet: én observasjon med NaN håndteres korrekt
"""

import warnings

import numpy as np
import pytest

from nemo.estimation.mcmc import (
    PARAM_NAMES,
    KM,
    _kf_block,
    kalman_hull,
    log_posterior,
    build_H,
    build_Sv,
    build_Q,
)


@pytest.fixture(scope="module")
def kalman_input(kalibrert_modell, syntetisk_obs):
    T, R, _ = kalibrert_modell
    Y_pre, Y_post = syntetisk_obs
    H  = build_H()
    Sv = build_Sv()
    theta_km = np.array([KM[n] for n in PARAM_NAMES])
    Q  = build_Q(theta_km)
    return T, R, H, Q, Sv, Y_pre, Y_post


def test_kf_block_endelig(kalman_input):
    """_kf_block returnerer endelig log-likelihood på syntetiske data."""
    T, R, H, Q, Sv, Y_pre, _ = kalman_input
    ll = _kf_block(T, R, H, Q, Sv, Y_pre)
    assert np.isfinite(ll), f"_kf_block returnerte {ll}"
    assert ll < 0, "Log-likelihood skal være negativ"


def test_kalman_hull_endelig(kalman_input):
    """kalman_hull (COVID-split) returnerer endelig log-likelihood."""
    T, R, H, Q, Sv, Y_pre, Y_post = kalman_input
    ll = kalman_hull(T, R, H, Q, Sv, Y_pre, Y_post)
    assert np.isfinite(ll), f"kalman_hull returnerte {ll}"


def test_kalman_hull_splitt(kalman_input):
    """
    kalman_hull = kf_pre + kf_post (COVID-gap gir uavhengige blokker).
    Differansen skal være ≈ 0 (numerisk presisjon).
    """
    T, R, H, Q, Sv, Y_pre, Y_post = kalman_input
    ll_hull  = kalman_hull(T, R, H, Q, Sv, Y_pre, Y_post)
    ll_pre   = _kf_block(T, R, H, Q, Sv, Y_pre)
    ll_post  = _kf_block(T, R, H, Q, Sv, Y_post)
    assert abs(ll_hull - (ll_pre + ll_post)) < 1e-8, (
        f"kalman_hull ({ll_hull:.4f}) ≠ kf_pre + kf_post ({ll_pre + ll_post:.4f})"
    )


def test_log_posterior_endelig_i_km(kalman_input):
    """log_posterior er endelig i K&M-kalibrasjonspunktet."""
    T, R, H, Q, Sv, Y_pre, Y_post = kalman_input
    Sv_obs = build_Sv()
    theta_km = np.array([KM[n] for n in PARAM_NAMES])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lp = log_posterior(theta_km, H, Sv_obs, Y_pre, Y_post)
    assert np.isfinite(lp), f"log_posterior = {lp} ved K&M-parametre"


def test_kf_nan_robusthet(kalman_input):
    """
    Kalman-filter håndterer én periode der alle observasjoner er NaN.
    (Tilsvarer manglende data, ikke COVID-gap.)
    """
    T, R, H, Q, Sv, Y_pre, _ = kalman_input
    Y_med_nan = Y_pre.copy().astype(float)
    Y_med_nan[10, :] = np.nan   # periode 10: ingen observasjoner

    ll = _kf_block(T, R, H, Q, Sv, Y_med_nan)
    assert np.isfinite(ll), f"_kf_block med NaN-rad returnerte {ll}"


def test_kf_utenfor_prior_gir_minus_inf(kalman_input):
    """
    log_posterior returnerer -inf når theta er utenfor priorgrensene.
    """
    T, R, H, Q, Sv, Y_pre, Y_post = kalman_input
    Sv_obs = build_Sv()
    theta_ugyldig = np.array([KM[n] for n in PARAM_NAMES])
    # Sett rho_A = 1.5 (over øvre priorbegrensning 0.9995)
    theta_ugyldig[PARAM_NAMES.index('rho_A')] = 1.5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lp = log_posterior(theta_ugyldig, H, Sv_obs, Y_pre, Y_post)
    assert lp == -np.inf, f"Forventet -inf utenfor prior, fikk {lp}"
