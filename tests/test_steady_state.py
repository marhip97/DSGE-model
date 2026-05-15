"""
[DSGE/QA] Steady-state konsistenssjekk (Spor A5).

Verifiserer at de kalibrerte steady-state-forholdene tilfredsstiller
ressursbetingelsen i den åpne økonomien:

    Y = C + I + G + X - M
  ⇒ 1 = CY + IY + GY + XY - MY

Med boliginvestering separert (IHY):
    1 = CY + IY + IHY + GY + XY - MY
"""

import pytest

from nemo.model.parameters import Parameters as p


@pytest.mark.xfail(
    strict=True,
    reason=(
        "KJENT KALIBRERINGSFEIL (Spor A5): CY+IY+GY+XY-MY = 0.84, "
        "avvik -0.16 fra ressursbetingelsen. Kommentar i parameters.py linje 142 "
        "sier 'IY inkluderer nå kapitalakkumulering', men summen blir ikke 1. "
        "Bør utredes mot K&M (2019) Tabell 8 og SSB nasjonalregnskap."
    ),
)
def test_ressursbetingelse_uten_bolig():
    """1 = CY + IY + GY + XY - MY (uten separat boliginvestering)."""
    sum_ = p.CY + p.IY + p.GY + p.XY - p.MY
    assert abs(sum_ - 1.0) < 0.05, (
        f"CY+IY+GY+XY-MY = {sum_:.3f}, avvik {sum_-1.0:+.3f}"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "KJENT KALIBRERINGSFEIL (Spor A5): CY+IY+IHY+GY+XY-MY = 0.94, "
        "avvik -0.06. Nærmere 1 enn uten-bolig-versjonen (0.84), så IY+IHY-"
        "tolkningen er trolig riktig. Resterende 6 pp gap må utredes: "
        "enten reell miskalibrering eller statistisk diskrepans i kilden."
    ),
)
def test_ressursbetingelse_med_bolig():
    """1 = CY + IY + IHY + GY + XY - MY (med separat boliginvestering)."""
    sum_ = p.CY + p.IY + p.IHY + p.GY + p.XY - p.MY
    assert abs(sum_ - 1.0) < 0.05, (
        f"CY+IY+IHY+GY+XY-MY = {sum_:.3f}, avvik {sum_-1.0:+.3f}"
    )


def test_andeler_positive():
    """Alle steady-state andeler er positive (sanity)."""
    for navn in ("CY", "IY", "IHY", "GY", "XY", "MY"):
        verdi = getattr(p, navn)
        assert verdi > 0, f"{navn} = {verdi}, må være > 0"


def test_andeler_realistiske_grenser():
    """
    Steady-state andeler er innenfor rimelige grenser for norsk økonomi.
    Grensene er romslige — formålet er å fange konfigurerings-tabber.
    """
    grenser = {
        "CY":  (0.40, 0.65),   # konsum/BNP
        "IY":  (0.10, 0.30),   # investering/BNP
        "IHY": (0.03, 0.15),   # boliginvest./BNP
        "GY":  (0.15, 0.30),   # offentlig/BNP
        "XY":  (0.20, 0.50),   # eksport/BNP
        "MY":  (0.20, 0.45),   # import/BNP
    }
    for navn, (nedre, ovre) in grenser.items():
        verdi = getattr(p, navn)
        assert nedre <= verdi <= ovre, (
            f"{navn} = {verdi:.3f}, utenfor [{nedre:.2f}, {ovre:.2f}]"
        )
