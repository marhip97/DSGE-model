"""
[DSGE/NUM] Test av GEORG-politikkregelen (Staff Memo 15/2025, kj-GEORG).

Sjekker:
- build_matrices_georg returnerer (NZ_GEORG×NZ_GEORG)-matriser
- Lagg-identiteter og AR(1)-Z er korrekt definert
- BK-stabilitet med Tabell 4-koeffisientene
- IRF-fortegn for pengepolitikksjokk (BNP(−), π(−))
- Exitstrategi: use_georg=False gir eksakt v3_forward-kjernebane
- Indikatorkobling i rad 20 (renteglatting, Z-ledd)
- build_H_georg har korrekt form
"""

import warnings
import numpy as np
import pytest

from nemo.model.equations import (
    build_matrices_georg, build_matrices_v3_forward,
    NZ, NZ_GEORG, I_R, I_R_L, PI, PIW, S, A, Y, E_i,
    GEORG_Z, GEORG_PI_L2, GEORG_PIW_L1, GEORG_PIW_L2,
    GEORG_A_L1, GEORG_A_L3, GEORG_S_L1, GEORG_S_L7, PI_L,
)
from nemo.model.parameters import Parameters
from nemo.solver.blanchard_kahn import solve, compute_irf
from nemo.estimation.mcmc import build_H_georg, build_H, N_OBS


@pytest.fixture(scope="module")
def georg_system():
    G0, G1, Psi, Pi = build_matrices_georg(Parameters, theta_H=0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        T, R, d = solve(G0, G1, Psi, Pi, verbose=False)
    return G0, G1, Psi, Pi, T, R, d


def test_matrisedimensjoner(georg_system):
    G0, G1, Psi, Pi, T, R, d = georg_system
    assert G0.shape == (NZ_GEORG, NZ_GEORG), f"G0.shape={G0.shape}"
    assert G1.shape == (NZ_GEORG, NZ_GEORG)
    assert Psi.shape == (NZ_GEORG, 13)
    assert Pi.shape == (NZ_GEORG, NZ_GEORG)
    assert T.shape == (NZ_GEORG, NZ_GEORG)
    assert R.shape == (NZ_GEORG, 13)


def test_lagg_identiteter():
    """Nye GEORG-tilstander er korrekte 1-periodes lagg: k_t = src_{t-1}."""
    G0, G1, Psi, Pi = build_matrices_georg(Parameters, theta_H=0.05)
    for k, src in [
        (GEORG_PI_L2,  PI_L),
        (GEORG_PIW_L1, PIW),
        (GEORG_PIW_L2, GEORG_PIW_L1),
        (GEORG_A_L1,   A),
        (GEORG_S_L1,   S),
        (GEORG_S_L7,   GEORG_S_L7 - 1),
    ]:
        assert G0[k, k] == pytest.approx(1.0)
        assert G1[k, src] == pytest.approx(1.0)


def test_ar1_policy_shock():
    """Z_t = λ_Z·Z_{t-1} + ε_i; sjokket går via Z, ikke direkte i.i.d. på rad 20.

    Merk: rad 20-kolonnene perturberes av fixed-point-forventningsleddet (samme
    mekanisme som v3_forward), så vi tester wiringen på Z-raden (ren) + at Z
    entrer regelen tilnærmet én-til-én. Eksakt basis-wiring dekkes av exit-testen.
    """
    G0, G1, Psi, Pi = build_matrices_georg(Parameters, theta_H=0.05)
    lam = Parameters.georg_lambda_Z
    assert G0[GEORG_Z, GEORG_Z] == pytest.approx(1.0)
    assert G1[GEORG_Z, GEORG_Z] == pytest.approx(lam)
    assert Psi[GEORG_Z, E_i] == pytest.approx(1.0)
    # Rad 20 mottar sjokket via Z-ledd, ikke via i.i.d. Psi
    assert Psi[20, E_i] == pytest.approx(0.0)
    # Z entrer regelen (basis −1.0, lett perturbert av forventningsleddet)
    assert G0[20, GEORG_Z] < 0.0
    assert abs(G0[20, GEORG_Z] - (-1.0)) < 0.1


def test_rentglatting_koeffisient():
    """Rad 20 har renteglatting (negativ koeffisient på i_R_{t-1}) og dominant
    egen-ledd på i_R. Eksakt ω_r dekkes av basis-wiring (exit-test); her sjekkes
    fortegn/størrelsesorden siden fixed-point-forventningsleddet perturberer raden."""
    G0, G1, Psi, Pi = build_matrices_georg(Parameters, theta_H=0.05)
    assert G0[20, I_R] > 0.5, "egen-ledd på i_R skal være dominant positivt (~1)"
    assert G0[20, I_R_L] < 0.0, "renteglatting: i_R_{t-1}-koeffisient skal være negativ"


def test_bk_stabilitet(georg_system):
    """BK-stabil med Tabell 4-koeffisientene."""
    _, _, _, _, T, R, d = georg_system
    assert d["stable"], f"GEORG ustabil: max|eig|={d.get('max_eig_T')}"
    assert d["max_eig_T"] < 1.0, f"max|eig(T)|={d['max_eig_T']:.6f} ≥ 1.0"


def test_irf_fortegn(georg_system):
    """Pengepolitikksjokk (+) → kumulativ BNP(−) og π(−)."""
    _, _, _, _, T, R, d = georg_system
    irf = compute_irf(T, R, E_i, 0.0025, T_periods=20)
    assert irf[:, Y].sum()  < 0.0, "BNP-respons skal være negativ kumulativt"
    assert irf[:, PI].sum() < 0.0, "π-respons skal være negativ kumulativt"
    assert float(np.max(irf[:, I_R])) > 0.0, "Styringsrenten skal stige initialt"


def test_exit_use_georg_false():
    """use_georg=False gir eksakt samme kjernebane (første NZ) som v3_forward."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        G0_g, G1_g, Psi_g, Pi_g = build_matrices_georg(
            Parameters, theta_H=0.05, use_georg=False
        )
        G0_f, G1_f, Psi_f, Pi_f = build_matrices_v3_forward(Parameters, theta_H=0.05)
        T_g, R_g, d_g = solve(G0_g, G1_g, Psi_g, Pi_g, verbose=False)
        T_f, R_f, d_f = solve(G0_f, G1_f, Psi_f, Pi_f, verbose=False)

    irf_g = compute_irf(T_g, R_g, E_i, 0.0025, T_periods=20)
    irf_f = compute_irf(T_f, R_f, E_i, 0.0025, T_periods=20)
    np.testing.assert_allclose(
        irf_g[:, :NZ], irf_f[:, :NZ],
        atol=1e-8, err_msg="use_georg=False avviker fra v3_forward-kjernen",
    )


def test_georg_avviker_fra_mimicking():
    """GEORG-IRF skal faktisk skille seg fra mimicking rule (ellers er regelen inaktiv)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        G0_g, G1_g, Psi_g, Pi_g = build_matrices_georg(Parameters, theta_H=0.05)
        G0_f, G1_f, Psi_f, Pi_f = build_matrices_v3_forward(Parameters, theta_H=0.05)
        T_g, R_g, _ = solve(G0_g, G1_g, Psi_g, Pi_g, verbose=False)
        T_f, R_f, _ = solve(G0_f, G1_f, Psi_f, Pi_f, verbose=False)
    irf_g = compute_irf(T_g, R_g, E_i, 0.0025, T_periods=20)
    irf_f = compute_irf(T_f, R_f, E_i, 0.0025, T_periods=20)
    # I_R-banen skal avvike (AR(1)-Z + GEORG-indikatorer gir annen dynamikk)
    assert np.max(np.abs(irf_g[:, I_R] - irf_f[:, I_R])) > 1e-6, (
        "GEORG-IRF er identisk med mimicking — regelen ser inaktiv ut"
    )


def test_build_H_georg_form():
    """build_H_georg har riktig form (N_OBS × NZ_GEORG) med null-kolonner for nye states."""
    H = build_H_georg()
    assert H.shape == (N_OBS, NZ_GEORG), f"H.shape={H.shape}"
    assert np.all(H[:, NZ:] == 0.0), "GEORG-tilstandskolonner skal være null (ikke observert)"
    H_50 = build_H()
    np.testing.assert_array_equal(H[:, :NZ], H_50, err_msg="Første NZ kolonner skal matche build_H()")
