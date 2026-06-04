"""
[STAT/NUM] Test av PLT-kanalen (prisnivåmål, kj46).

Sjekker:
- build_matrices_v3_plt returnerer (NZ_PLT×NZ_PLT)-matriser
- P_STAR_GAP-likning er korrekt definert
- BK-stabilitet med psi_PL=0.10 (estimeringsscenario)
- I_R.q12 synker monotont med økt psi_PL
- Exitstrategi: psi_PL=0 gir eksakt same I_R-bane som v3_forward
- build_H_plt har korrekt form
"""

import warnings
import numpy as np
import pytest

from nemo.model.equations import (
    build_matrices_v3_plt, build_matrices_v3_forward,
    NZ, NZ_PLT, P_STAR_GAP, I_R, PI, E_i,
)
from nemo.model.parameters import Parameters
from nemo.solver.blanchard_kahn import solve, compute_irf
from nemo.estimation.mcmc import build_H_plt, N_OBS, build_H


class P_PLT(Parameters):
    """Testparametere med typisk PLT-styrke."""
    psi_R  = 0.9490
    phi_PQ = 150.0
    psi_PL = 0.10


@pytest.fixture(scope="module")
def plt_system():
    G0, G1, Psi, Pi = build_matrices_v3_plt(P_PLT, theta_H=0.05, lambda_pi4=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        T, R, d = solve(G0, G1, Psi, Pi, verbose=False)
    return G0, G1, T, R, d


def test_matrisedimensjoner(plt_system):
    G0, G1, T, R, d = plt_system
    assert G0.shape == (NZ_PLT, NZ_PLT), f"G0.shape={G0.shape}"
    assert G1.shape == (NZ_PLT, NZ_PLT)
    assert T.shape  == (NZ_PLT, NZ_PLT)
    assert R.shape  == (NZ_PLT, 13)


def test_bk_stabilitet_med_plt(plt_system):
    """BK-stabil med psi_PL=0.10."""
    _, _, T, R, d = plt_system
    assert d["stable"], f"PLT ustabil: max|eig|={d['max_eig_T']:.6f}"
    assert d["max_eig_T"] < 1.0, f"max|eig(T)|={d['max_eig_T']:.6f} ≥ 1.0"


def test_p_star_gap_likning():
    """P_STAR_GAP-likning: G0[P_STAR_GAP, P_STAR_GAP]=1, G0[P_STAR_GAP, PI]=-1, G1[P_STAR_GAP, P_STAR_GAP]=1."""
    G0, G1, Psi, Pi = build_matrices_v3_plt(P_PLT, theta_H=0.05, lambda_pi4=0.0)
    assert G0[P_STAR_GAP, P_STAR_GAP] == pytest.approx(1.0)
    assert G0[P_STAR_GAP, PI]         == pytest.approx(-1.0)
    assert G1[P_STAR_GAP, P_STAR_GAP] == pytest.approx(1.0)


def test_plt_ledd_i_taylor():
    """Taylor-regel (rad I_R) har psi_PL-koeffisient for P_STAR_GAP."""
    G0, G1, Psi, Pi = build_matrices_v3_plt(P_PLT, theta_H=0.05, lambda_pi4=0.0)
    psi_R  = P_PLT.psi_R
    psi_R2 = P_PLT.psi_R2
    psi_PL = P_PLT.psi_PL
    _scale = 1.0 - psi_R - psi_R2
    forventet = -_scale * psi_PL
    assert G0[I_R, P_STAR_GAP] == pytest.approx(forventet, rel=1e-6)


def test_exit_psi_PL_0():
    """psi_PL=0 gir eksakt samme I_R-bane som v3_forward."""
    class P_exit(Parameters):
        psi_R  = 0.9490
        phi_PQ = 150.0
        psi_PL = 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        G0_plt,  G1_plt,  Psi_plt,  Pi_plt  = build_matrices_v3_plt(P_exit, theta_H=0.05, lambda_pi4=0.0)
        G0_fwd,  G1_fwd,  Psi_fwd,  Pi_fwd  = build_matrices_v3_forward(P_exit, theta_H=0.05, lambda_pi4=0.0)
        T_plt, R_plt, d_plt = solve(G0_plt, G1_plt, Psi_plt, Pi_plt, verbose=False)
        T_fwd, R_fwd, d_fwd = solve(G0_fwd, G1_fwd, Psi_fwd, Pi_fwd, verbose=False)

    irf_plt = compute_irf(T_plt, R_plt, E_i, 0.0025, T_periods=13)
    irf_fwd = compute_irf(T_fwd, R_fwd, E_i, 0.0025, T_periods=13)
    np.testing.assert_allclose(
        irf_plt[:, :NZ], irf_fwd[:, :NZ],
        atol=1e-8, err_msg="psi_PL=0 gir annen I_R-bane enn v3_forward",
    )


def test_ir_q12_synker_med_psi_PL():
    """I_R.q12 synker monotont med økt psi_PL."""
    vals = []
    for ppl in [0.0, 0.10, 0.20, 0.40]:
        class P_t(Parameters):
            psi_R  = 0.9490
            phi_PQ = 150.0
        P_t.psi_PL = ppl
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            G0, G1, Psi, Pi = build_matrices_v3_plt(P_t, theta_H=0.05, lambda_pi4=0.0)
            T, R, d = solve(G0, G1, Psi, Pi, verbose=False)
        if not d["stable"]:
            continue
        irf = compute_irf(T, R, E_i, 0.0025, T_periods=13)
        peak = float(np.max(irf[:, I_R]))
        irf_n = irf / peak if peak > 0 else irf
        vals.append(float(irf_n[11, I_R]))

    for i in range(len(vals) - 1):
        assert vals[i] > vals[i + 1], (
            f"I_R.q12 synker ikke monotont: {vals}"
        )


def test_build_H_plt_form():
    """build_H_plt har riktig form (N_OBS × NZ_PLT) med null-kolonne for P_STAR_GAP."""
    H = build_H_plt()
    assert H.shape == (N_OBS, NZ_PLT), f"H.shape={H.shape}"
    assert np.all(H[:, P_STAR_GAP] == 0.0), "P_STAR_GAP-kolonnen skal være null (ikke observert)"
    H_50 = build_H()
    np.testing.assert_array_equal(H[:, :NZ], H_50, err_msg="Første NZ kolonner skal matche build_H()")
