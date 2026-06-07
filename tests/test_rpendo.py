"""
[DSGE/NUM] Test av endogen risikopremie i UIP (build_matrices_rpendo).

Sjekker:
- (NZ_RPENDO×NZ_RPENDO)-matriser
- RP_ENDO-loven: RP_ENDO_t = ρ_pe·RP_ENDO_{t-1} + κ_pe·(i_D − i*)
- UIP-kobling i rad 15
- BK-stabilitet i kalibreringsområdet
- IRF-fortegn for pengepolitikksjokk (BNP(−), π(−))
- Premien forsterker RER-appresiering (større |RER| ved κ>0)
- Exitstrategi: κ_pe=0 gir eksakt v3_forward-kjernebane
"""

import warnings
import numpy as np
import pytest

from nemo.model.equations import (
    build_matrices_rpendo, build_matrices_v3_forward,
    NZ, NZ_RPENDO, RP_ENDO, I_R, I_D, I_STAR, Y, PI, RER, E_i,
)
from nemo.model.parameters import Parameters
from nemo.solver.blanchard_kahn import solve, compute_irf


class P_RP(Parameters):
    """Kalibrert endogen premie i NB-treffende område."""
    kappa_rp_endo = 0.30
    rho_rp_endo   = 0.50


@pytest.fixture(scope="module")
def rp_system():
    G0, G1, Psi, Pi = build_matrices_rpendo(P_RP, theta_H=0.05, lambda_pi4=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        T, R, d = solve(G0, G1, Psi, Pi, verbose=False)
    return G0, G1, Psi, Pi, T, R, d


def test_matrisedimensjoner(rp_system):
    G0, G1, Psi, Pi, T, R, d = rp_system
    assert G0.shape == (NZ_RPENDO, NZ_RPENDO)
    assert Psi.shape == (NZ_RPENDO, 13)
    assert T.shape == (NZ_RPENDO, NZ_RPENDO)


def test_rp_endo_lov():
    """RP_ENDO_t = ρ_pe·RP_ENDO_{t-1} + κ_pe·(i_D − i*)."""
    G0, G1, Psi, Pi = build_matrices_rpendo(P_RP, theta_H=0.05, lambda_pi4=0.0)
    assert G0[RP_ENDO, RP_ENDO] == pytest.approx(1.0)
    assert G0[RP_ENDO, I_D]    == pytest.approx(-P_RP.kappa_rp_endo)
    assert G0[RP_ENDO, I_STAR] == pytest.approx(+P_RP.kappa_rp_endo)
    assert G1[RP_ENDO, RP_ENDO] == pytest.approx(P_RP.rho_rp_endo)


def test_uip_kobling():
    """UIP-rad 15 har premie-ledd (1−ρ_s) på RP_ENDO."""
    G0, G1, Psi, Pi = build_matrices_rpendo(P_RP, theta_H=0.05, lambda_pi4=0.0)
    _w = 1.0 - float(getattr(P_RP, 'rho_s', 0.0))
    assert G0[15, RP_ENDO] == pytest.approx(_w)


def test_bk_stabilitet(rp_system):
    _, _, _, _, T, R, d = rp_system
    assert d["stable"], f"rpendo ustabil: max|eig|={d.get('max_eig_T')}"
    assert d["max_eig_T"] < 1.0


def test_irf_fortegn(rp_system):
    """Pengepolitikksjokk (+) → kumulativ BNP(−) og π(−)."""
    _, _, _, _, T, R, d = rp_system
    irf = compute_irf(T, R, E_i, 0.0025, T_periods=20)
    assert irf[:, Y].sum() < 0.0
    assert irf[:, PI].sum() < 0.0


def test_premie_forsterker_appresiering():
    """κ>0 gir større RER-appresiering på impact enn κ=0 (v3_forward)."""
    def rer_q1(kappa):
        class Pt(Parameters):
            rho_rp_endo = 0.50
        Pt.kappa_rp_endo = kappa
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            G0, G1, Psi, Pi = build_matrices_rpendo(Pt, theta_H=0.05, lambda_pi4=0.0)
            T, R, d = solve(G0, G1, Psi, Pi, verbose=False)
        irf = compute_irf(T, R, E_i, 0.0025, T_periods=20)
        return irf[0, RER] / np.max(irf[:, I_R])
    assert rer_q1(0.30) < rer_q1(0.0), "premien skal forsterke appresiering (mer negativ RER)"


def test_exit_kappa_0():
    """κ_pe=0 gir eksakt samme kjernebane (første NZ) som v3_forward."""
    class P0(Parameters):
        kappa_rp_endo = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        G0r, G1r, Psir, Pir = build_matrices_rpendo(P0, theta_H=0.05, lambda_pi4=0.0)
        G0f, G1f, Psif, Pif = build_matrices_v3_forward(P0, theta_H=0.05, lambda_pi4=0.0)
        Tr, Rr, _ = solve(G0r, G1r, Psir, Pir, verbose=False)
        Tf, Rf, _ = solve(G0f, G1f, Psif, Pif, verbose=False)
    irf_r = compute_irf(Tr, Rr, E_i, 0.0025, T_periods=20)
    irf_f = compute_irf(Tf, Rf, E_i, 0.0025, T_periods=20)
    np.testing.assert_allclose(
        irf_r[:, :NZ], irf_f[:, :NZ], atol=1e-8,
        err_msg="κ_pe=0 avviker fra v3_forward-kjernen",
    )
