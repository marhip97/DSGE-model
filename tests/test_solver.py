"""
[NUM/ARK] Test av Blanchard-Kahn-løser og modellstabilitet.

Dekker:
- BK-stabilitet (max|eig(T)| < 1) for alle tre modellversjoner
- Dimensjonskontroll (T er NZ×NZ, R er NZ×NE)
- G0-rang (full rang = NZ)
"""

import warnings

import numpy as np
import pytest

from nemo.model.equations import NE, NZ, build_matrices, build_matrices_v2, build_matrices_v3
from nemo.model.parameters import Parameters
from nemo.solver.blanchard_kahn import solve as bk_solve


@pytest.mark.parametrize("builder,label", [
    pytest.param(build_matrices, "v1", marks=pytest.mark.xfail(
        reason="v1 marginalt ustabil etter A4a bank-fix (2026-05-18). "
               "v1/v2 er deprecated — kun v3 brukes i produksjon.",
        strict=True,
    )),
    pytest.param(build_matrices_v2, "v2", marks=pytest.mark.xfail(
        reason="v2 marginalt ustabil etter A4a bank-fix (2026-05-18). "
               "v1/v2 er deprecated — kun v3 brukes i produksjon.",
        strict=True,
    )),
    (build_matrices_v3, "v3"),
])
def test_bk_stabilitet(builder, label):
    """max|eig(T)| < 1 for produksjonsmodellversjon (v3)."""
    G0, G1, Psi, Pi = builder(Parameters)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        T, R, diag = bk_solve(G0, G1, Psi, Pi, verbose=False)
    assert diag["stable"], (
        f"v{label}: T er ikke stabil — max|eig| = {diag['max_eig_T']:.6f}"
    )
    assert diag["max_eig_T"] < 1.0, (
        f"v{label}: max|eig(T)| = {diag['max_eig_T']:.6f} ≥ 1.0"
    )


@pytest.mark.parametrize("builder,label", [
    (build_matrices,    "v1"),
    (build_matrices_v2, "v2"),
    (build_matrices_v3, "v3"),
])
def test_dimensjoner(builder, label):
    """T er (NZ×NZ), R er (NZ×NE)."""
    G0, G1, Psi, Pi = builder(Parameters)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        T, R, diag = bk_solve(G0, G1, Psi, Pi, verbose=False)
    assert T.shape == (NZ, NZ), f"{label}: T.shape = {T.shape}, forventet ({NZ}, {NZ})"
    assert R.shape == (NZ, NE), f"{label}: R.shape = {R.shape}, forventet ({NZ}, {NE})"


def test_g0_full_rang():
    """G0 er ikke singulær (rang = NZ)."""
    G0, G1, Psi, Pi = build_matrices_v3(Parameters)
    rang = np.linalg.matrix_rank(G0)
    assert rang == NZ, f"G0-rang = {rang}, forventet {NZ}"


def test_modell_dimensjonskonstanter():
    """NZ=49 (etter Alt. A: variabel kapitalutnyttelse) og NE=13."""
    assert NZ == 49, f"NZ = {NZ}, forventet 49 (Alt. A 2026-05-15)"
    assert NE == 13, f"NE = {NE}, forventet 13"


def test_psi_dimensjon():
    """Psi har riktig dimensjon (NZ×NE)."""
    G0, G1, Psi, Pi = build_matrices_v3(Parameters)
    assert Psi.shape == (NZ, NE), f"Psi.shape = {Psi.shape}"


def test_t_matrise_er_reell(kalibrert_modell):
    """T-matrisen er reell etter Schur-dekomposisjon."""
    T, R, _ = kalibrert_modell
    assert np.isreal(T).all(), "T inneholder komplekse tall"
    assert np.isreal(R).all(), "R inneholder komplekse tall"
