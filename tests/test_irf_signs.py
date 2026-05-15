"""
[QA/DSGE] Kvalitative IRF-krav — 15 tegn som modellen må tilfredsstille.

Referanse: Kravik & Mimir (2019), NB Memo 3/2024 Figur 1.
Alle krav er kumulativt (sum over 8 kvartaler) for å unngå
kortvarige numeriske artefakter.

Kravene:
 1–4  : Pengepolitikkssjokk
 5–6  : Risikopremiesjokk
 7–8  : Prismarkupsjokk (kostnadsdrevet)
 9–10 : TFP-sjokk
11    : Oljeprisssjokk
12–13 : Utenlandsk etterspørselssjokk
14    : Boligpreferansesjokk
15    : LTV-sjokk (husholdning)
"""

import warnings

import numpy as np
import pytest

from nemo.model.equations import (
    C_NW, E_H, E_Ys, E_i, E_O, E_P, E_A, E_rp, E_phi_h,
    I_R, PI, Q_H, RER, Y,
)
from nemo.solver.blanchard_kahn import compute_irf

_HORIZONS = 8  # kvartaler for kumulativ sum


def _cum(T, R, shock_idx, shock_size):
    irf = compute_irf(T, R, shock_idx, shock_size, T_periods=_HORIZONS)
    return irf.sum(axis=0)


@pytest.fixture(scope="module")
def krav_data(kalibrert_modell):
    T, R, _ = kalibrert_modell
    return T, R


# ── Pengepolitikkssjokk (+25 bp) ─────────────────────────────────────────────

def test_01_pengepol_rente_opp(krav_data):
    """Krav 1: Pengepol → styringsrente opp."""
    T, R = krav_data
    cum = _cum(T, R, E_i, 0.0025)
    assert cum[I_R] > 0, f"I_R kumulativ = {cum[I_R]:.4f}"


def test_02_pengepol_bnp_ned(krav_data):
    """Krav 2: Pengepol → BNP ned."""
    T, R = krav_data
    cum = _cum(T, R, E_i, 0.0025)
    assert cum[Y] < 0, f"Y kumulativ = {cum[Y]:.4f}"


def test_03_pengepol_inflasjon_ned(krav_data):
    """Krav 3: Pengepol → inflasjon ned."""
    T, R = krav_data
    cum = _cum(T, R, E_i, 0.0025)
    assert cum[PI] < 0, f"PI kumulativ = {cum[PI]:.4f}"


def test_04_pengepol_rer_ned(krav_data):
    """Krav 4: Pengepol → NOK appresierer (RER ned)."""
    T, R = krav_data
    cum = _cum(T, R, E_i, 0.0025)
    assert cum[RER] < 0, f"RER kumulativ = {cum[RER]:.4f}"


# ── Risikopremiesjokk ─────────────────────────────────────────────────────────

def test_05_risikopremie_rer_opp(krav_data):
    """Krav 5: Risikopremie → NOK depresiererer (RER opp)."""
    T, R = krav_data
    cum = _cum(T, R, E_rp, 0.01)
    assert cum[RER] > 0, f"RER kumulativ = {cum[RER]:.4f}"


def test_06_risikopremie_inflasjon_opp(krav_data):
    """Krav 6: Risikopremie → importprisinflasjon driver PI opp."""
    T, R = krav_data
    cum = _cum(T, R, E_rp, 0.01)
    assert cum[PI] > 0, f"PI kumulativ = {cum[PI]:.4f}"


# ── Prismarkupsjokk (kostnadsdrevet) ─────────────────────────────────────────

def test_07_kostnad_inflasjon_opp(krav_data):
    """Krav 7: Prismarkup → inflasjon opp."""
    T, R = krav_data
    cum = _cum(T, R, E_P, 0.003)
    assert cum[PI] > 0, f"PI kumulativ = {cum[PI]:.4f}"


def test_08_kostnad_bnp_ned(krav_data):
    """Krav 8: Prismarkup → BNP ned (stagflasjonseffekt)."""
    T, R = krav_data
    cum = _cum(T, R, E_P, 0.003)
    assert cum[Y] < 0, f"Y kumulativ = {cum[Y]:.4f}"


# ── TFP-sjokk ────────────────────────────────────────────────────────────────

@pytest.mark.xfail(
    strict=True,
    reason=(
        "KJENT MODELLFEIL (Spor A): TFP-sjokk gir negativ BNP-respons i alle 20 kvartaler. "
        "Bygger opp positivt etter t=11, men kumulativt < 0. "
        "Sannsynlig årsak: manglende kanal eller koeffisientfeil i produksjonsfunksjonen. "
        "Må utredes i Spor A (likningstransparens) og Spor C4 (sigma_A identifikasjon)."
    ),
)
def test_09_tfp_bnp_opp(krav_data):
    """Krav 9: TFP → BNP opp."""
    T, R = krav_data
    cum = _cum(T, R, E_A, 0.007)
    assert cum[Y] > 0, f"Y kumulativ = {cum[Y]:.4f}"


def test_10_tfp_inflasjon_ned(krav_data):
    """Krav 10: TFP → inflasjon ned (marginal kostnad faller)."""
    T, R = krav_data
    cum = _cum(T, R, E_A, 0.007)
    assert cum[PI] < 0, f"PI kumulativ = {cum[PI]:.4f}"


# ── Oljeprisssjokk ───────────────────────────────────────────────────────────

def test_11_oljepris_bnp_opp(krav_data):
    """Krav 11: Oljeprissjokk → norsk BNP (Fastlands) opp via velferdskanal."""
    T, R = krav_data
    cum = _cum(T, R, E_O, 0.10)
    assert cum[Y] > 0, f"Y kumulativ = {cum[Y]:.4f}"


# ── Utenlandsk etterspørselssjokk ────────────────────────────────────────────

def test_12_ettersp_bnp_opp(krav_data):
    """Krav 12: Utenl. etterspørsel → eksport og BNP opp."""
    T, R = krav_data
    cum = _cum(T, R, E_Ys, 0.01)
    assert cum[Y] > 0, f"Y kumulativ = {cum[Y]:.4f}"


def test_13_ettersp_rente_opp(krav_data):
    """Krav 13: Utenl. etterspørsel → styringsrente opp (Taylor-respons)."""
    T, R = krav_data
    cum = _cum(T, R, E_Ys, 0.01)
    assert cum[I_R] > 0, f"I_R kumulativ = {cum[I_R]:.4f}"


# ── Boligpreferansesjokk ─────────────────────────────────────────────────────

def test_14_bolig_boligpris_opp(krav_data):
    """Krav 14: Boligpreferansesjokk → boligpris opp."""
    T, R = krav_data
    cum = _cum(T, R, E_H, 0.01)
    assert cum[Q_H] > 0, f"Q_H kumulativ = {cum[Q_H]:.4f}"


# ── LTV-sjokk ────────────────────────────────────────────────────────────────

@pytest.mark.xfail(
    strict=True,
    reason=(
        "KJENT MODELLFEIL (Spor A4c): LTV-sjokk gir positiv C_NW fra t=2. "
        "t=1 er korrekt negativ, men kollateral-oppgangseffekten via q_H dominerer. "
        "Mulig feil: EPS_PHI_H-fortegn i utlånsrente-likning (G0[22,EPS_PHI_H]=-1.0 "
        "og G0[23,EPS_PHI_H]=-1.0 senker renter ved positivt sjokk, øker kreditt). "
        "Krever review av LTV-sjokk-konvensjon mot K&M (2019) seksjon 2.2 / Gerali et al. 2010."
    ),
)
def test_15_ltv_laantakerkonsum_ned(krav_data):
    """Krav 15: LTV-sjokk → låntakerkonsum (c_NW) ned (strammet belåningsgrense)."""
    T, R = krav_data
    cum = _cum(T, R, E_phi_h, 0.01)
    assert cum[C_NW] < 0, f"C_NW kumulativ = {cum[C_NW]:.4f}"
