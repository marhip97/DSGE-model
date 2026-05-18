"""Tester for logit-reparametrisering (Fase 2, C5 §2)."""

from __future__ import annotations

import numpy as np
import pytest

from nemo.estimation.mcmc import KM, PARAM_NAMES, PARAM_PRIORS, log_posterior
from nemo.estimation.reparam import (
    REPARAM_PARAMS,
    log_jacobian,
    to_natural,
    to_unconstrained,
    wrap_log_posterior,
)


def _theta_km() -> np.ndarray:
    return np.array([KM.get(n, 0.5) for n in PARAM_NAMES], dtype=float)


def test_reparam_params_finnes():
    for n in REPARAM_PARAMS:
        assert n in PARAM_NAMES, f"{n} ikke i PARAM_NAMES"


def test_roundtrip_naturlig_til_unc_og_tilbake():
    """to_natural(to_unconstrained(x)) == x for x i (lb, ub)."""
    theta = _theta_km()
    theta_unc = to_unconstrained(theta)
    theta_back = to_natural(theta_unc)
    np.testing.assert_allclose(theta_back, theta, rtol=1e-10, atol=1e-12)


def test_kun_reparam_params_endres():
    """Indekser utenfor REPARAM_PARAMS skal være uendret etter transformasjon."""
    theta = _theta_km()
    theta_unc = to_unconstrained(theta)
    for i, name in enumerate(PARAM_NAMES):
        if name not in REPARAM_PARAMS:
            assert theta_unc[i] == theta[i], f"{name} endret av to_unconstrained"


def test_unc_er_endelig_for_indre_punkter():
    theta = _theta_km()
    theta_unc = to_unconstrained(theta)
    assert np.all(np.isfinite(theta_unc))


@pytest.mark.parametrize("name", list(REPARAM_PARAMS))
def test_grensetilfeller_klippes(name: str):
    """Verdier ved priorgrensa skal håndteres uten NaN."""
    theta = _theta_km()
    i = PARAM_NAMES.index(name)
    lb, ub = PARAM_PRIORS[name][-2], PARAM_PRIORS[name][-1]

    # Test ved nedre grense
    theta[i] = lb
    unc = to_unconstrained(theta)
    assert np.isfinite(unc[i])

    # Test ved øvre grense
    theta[i] = ub
    unc = to_unconstrained(theta)
    assert np.isfinite(unc[i])


def test_jacobian_har_riktig_fortegn_og_størrelse():
    """log|dx/dy| = log(ub-lb) + log(u) + log(1-u). Skal være negativ siden u(1-u) ≤ 0.25."""
    theta = _theta_km()
    log_jac = log_jacobian(theta)
    assert np.isfinite(log_jac)
    # psi_R=0.667 i (0.01, 0.92) → u ≈ 0.727, log|jac| = log(0.91) + log(0.727) + log(0.273) ≈ -1.6
    assert log_jac < 0


def test_jacobian_returnerer_minusinf_utenfor_støtte():
    theta = _theta_km()
    i = PARAM_NAMES.index("psi_R")
    theta[i] = 0.001  # under nedre grense 0.01
    assert log_jacobian(theta) == -np.inf


def test_jacobian_numerisk_vs_analytisk():
    """Sammenlikn analytisk log|dx/dy| med numerisk derivat."""
    theta = _theta_km()
    theta_unc = to_unconstrained(theta)

    h = 1e-6
    for name in REPARAM_PARAMS:
        i = PARAM_NAMES.index(name)
        unc_plus = theta_unc.copy()
        unc_minus = theta_unc.copy()
        unc_plus[i] += h
        unc_minus[i] -= h
        x_plus = to_natural(unc_plus)[i]
        x_minus = to_natural(unc_minus)[i]
        dx_dy_num = (x_plus - x_minus) / (2 * h)

        lb, ub = PARAM_PRIORS[name][-2], PARAM_PRIORS[name][-1]
        u = (theta[i] - lb) / (ub - lb)
        dx_dy_ana = (ub - lb) * u * (1 - u)

        np.testing.assert_allclose(
            dx_dy_num, dx_dy_ana, rtol=1e-5,
            err_msg=f"{name}: numerisk {dx_dy_num} vs analytisk {dx_dy_ana}",
        )


def test_wrap_log_posterior_invarians():
    """log_post_unc(unc) = log_post_nat(nat) + log|dx/dy|.

    Sjekker at wrapperen produserer riktig densitet i unc-rom.
    """
    import pandas as pd
    from pathlib import Path
    from nemo.estimation.mcmc import build_H, build_Sv

    rot = Path(__file__).resolve().parents[1]
    data_sti = rot / "data" / "processed" / "nemo_data_faktisk_v2.csv"
    if not data_sti.exists():
        pytest.skip("Datafil mangler — skip likelihood-test")

    df = pd.read_csv(data_sti, index_col=0)
    df.index = pd.to_datetime(df.index)
    Y_pre = df[df.index <= "2019-12-31"].values
    Y_post = df[df.index >= "2022-01-01"].values
    H = build_H()
    Sv = build_Sv()

    theta = _theta_km()
    theta_unc = to_unconstrained(theta)

    lp_nat = log_posterior(theta, H, Sv, Y_pre, Y_post)
    if not np.isfinite(lp_nat):
        pytest.skip("log_posterior ikke endelig ved K&M-startverdier")

    log_post_unc = wrap_log_posterior(log_posterior)
    lp_unc = log_post_unc(theta_unc, H, Sv, Y_pre, Y_post)
    jac = log_jacobian(theta)

    np.testing.assert_allclose(lp_unc, lp_nat + jac, rtol=1e-10)
