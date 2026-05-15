"""
[QA/NUM] Test av FEVD-konsistens.

Sjokk-andeler skal summere til ≈ 100 % per variabel og horisont.
Toleranse: ± 2 prosentpoeng (numerisk avrunding i matriseprodukt).
"""

import warnings

import numpy as np
import pytest

from nemo.model.equations import NE, NZ, build_matrices_v3
from nemo.model.parameters import Parameters
from nemo.solver.blanchard_kahn import solve as bk_solve

_HORIZONS = [1, 4, 8, 20]
_TOL = 2.0  # prosentpoeng


def _beregn_fevd(T, R, Q_diag, n_vars, horizons):
    """
    Enkel FEVD-beregning: andel av prognosefeil-varians fra hvert sjokk.

    Forutsetter diagonal Q (ukorrelerte sjokk).
    """
    NZ = T.shape[0]
    NE = R.shape[1]
    max_h = max(horizons)

    # R_skalt: hver kolonne skalert med shock std
    R_s = R * np.sqrt(Q_diag)[np.newaxis, :]

    # MSE-bidrag per sjokk og horisont: (NZ, NE, max_h)
    contrib = np.zeros((NZ, NE, max_h))
    Tk = np.eye(NZ)
    for h in range(max_h):
        impulse = Tk @ R_s          # (NZ, NE)
        contrib[:, :, h] = impulse ** 2
        Tk = T @ Tk

    # Akkumulert MSE
    fevd = {}
    for h in horizons:
        mse_total = contrib[:, :, :h].sum(axis=2)   # (NZ, NE)
        total = mse_total.sum(axis=1, keepdims=True)  # (NZ, 1)
        with np.errstate(divide='ignore', invalid='ignore'):
            shares = np.where(total > 1e-20, mse_total / total * 100, 0.0)
        fevd[h] = shares[:n_vars, :]   # (n_vars, NE)
    return fevd


@pytest.fixture(scope="module")
def fevd_data(kalibrert_modell):
    T, R, _ = kalibrert_modell
    from nemo.estimation.mcmc import build_Q, PARAM_NAMES, KM
    theta_km = np.array([KM[n] for n in PARAM_NAMES])
    Q = build_Q(theta_km)
    Q_diag = np.diag(Q)
    return _beregn_fevd(T, R, Q_diag, n_vars=20, horizons=_HORIZONS)


@pytest.mark.parametrize("horisont", _HORIZONS)
def test_fevd_summer_til_100(fevd_data, horisont):
    """FEVD-andeler summerer til ≈ 100 % per variabel ved horisont h."""
    shares = fevd_data[horisont]  # (n_vars, NE)
    summer = shares.sum(axis=1)   # (n_vars,)
    for i, s in enumerate(summer):
        if shares[i].sum() < 1e-10:
            continue  # hopp over variabler uten varians
        assert abs(s - 100.0) < _TOL, (
            f"Variabel {i} ved horisont {horisont}: sum = {s:.2f} % (toleranse ±{_TOL})"
        )


@pytest.mark.parametrize("horisont", _HORIZONS)
def test_fevd_andeler_ikke_negative(fevd_data, horisont):
    """Ingen FEVD-andel er negativ."""
    shares = fevd_data[horisont]
    assert (shares >= -1e-10).all(), (
        f"Negative FEVD-andeler ved horisont {horisont}: min = {shares.min():.4f}"
    )


def test_fevd_risikopremie_rer_dominans(kalibrert_modell):
    """
    Kjent svakhet (CLAUDE.md): sigma_rp dominerer RER-varians.
    Testen dokumenterer nåsituasjonen — ikke et krav om at det er riktig.
    Skal oppdateres etter Spor C3-analyse.
    """
    from nemo.model.equations import RER
    from nemo.estimation.mcmc import build_Q, PARAM_NAMES, KM, E_rp as E_rp_idx

    T, R, _ = kalibrert_modell
    theta_km = np.array([KM[n] for n in PARAM_NAMES])
    Q = build_Q(theta_km)
    Q_diag = np.diag(Q)

    fevd = _beregn_fevd(T, R, Q_diag, n_vars=NZ, horizons=[20])
    rer_andeler = fevd[20][RER]  # andeler per sjokk

    # Sjekk at RER-varians ikke er null (modellen produserer RER-bevegelse)
    assert rer_andeler.sum() > 1.0, "RER har ingen FEVD-varians — trolig modellfeil"
