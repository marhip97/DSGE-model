"""
Pytest fixtures for NEMO-testpakken.

Alle fixtures bruker kalibrerte parametere (ingen syntetisk fallback i
produksjonsfunksjonene). Syntetiske data i fixtures er utelukkende for
testformål og produseres deterministisk via numpy seed.
"""

import warnings

import numpy as np
import pytest

from nemo.model.equations import (
    NE, NZ,
    build_matrices_v3,
)
from nemo.model.parameters import Parameters
from nemo.solver.blanchard_kahn import solve_klein


@pytest.fixture(scope="session")
def kalibrert_modell():
    """
    Returnerer (T, R, diag) for kalibrert v3-modell via Klein (2000) løser.

    Med K&M-kalibrering (mimicking rule) er modellen indeterminert i Klein-forstand
    (n_explosive=5 ≠ rank(Pi)=7). solve_klein faller tilbake til MSV-likevekten
    (direkte løsning), som er identisk med Sims direkte inversjon.
    """
    G0, G1, Psi, Pi = build_matrices_v3(Parameters)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        T, R, diag = solve_klein(G0, G1, Psi, Pi, verbose=False)
    return T, R, diag


@pytest.fixture(scope="session")
def kalibrert_modell_v3():
    """Returnerer (T, R, diag) for v3-modell (bakoverkompatibel fixture)."""
    G0, G1, Psi, Pi = build_matrices_v3(Parameters)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        T, R, diag = solve_klein(G0, G1, Psi, Pi, verbose=False)
    return T, R, diag


@pytest.fixture(scope="session")
def syntetisk_obs(kalibrert_modell):
    """
    Genererer 80 kvartalers syntetiske observasjoner fra kalibrert v3-modell.
    Brukes i Kalman-filter-tester.

    Returnerer (Y_pre, Y_post) der Y_post er 20 perioder etter en COVID-gap
    (8 kvartalers hopp i tid, ingen observasjoner).
    """
    T_mat, R_mat, _ = kalibrert_modell
    rng = np.random.default_rng(2024)

    from nemo.estimation.mcmc import N_OBS, build_H, build_Sv

    H  = build_H()
    Sv = build_Sv()

    n_pre  = 60
    n_post = 20
    n_total = n_pre + n_post

    # Simuler tilstandsbane (NZ=49 dimensjoner)
    z = np.zeros(NZ)
    states = []
    for _ in range(n_total):
        eps = rng.standard_normal(NE) * 0.01
        z = T_mat @ z + R_mat @ eps
        states.append(z.copy())

    states = np.array(states)  # (n_total, NZ)

    # Observasjoner med målefeil
    obs_noise = rng.standard_normal((n_total, N_OBS)) * 0.001
    Y_all = states @ H.T + obs_noise  # (n_total, N_OBS)

    Y_pre  = Y_all[:n_pre]
    Y_post = Y_all[n_pre:]
    return Y_pre, Y_post
