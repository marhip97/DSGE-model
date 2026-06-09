"""
[QA/ARK] Røyktest av nemo.dashboard.

Verifiserer at build_dashboard produserer gyldig HTML med alle seksjoner.
"""

import json
import os
import tempfile

import pytest

from nemo.dashboard.build import build_dashboard


@pytest.fixture(scope="module")
def analyse_json(tmp_path_factory):
    """Lag minimalt analyse_resultater.json for testing."""
    data = {
        "irf": {
            "Pengepol.": {
                "BNP-gap": [round(-0.01 * (0.9**t), 4) for t in range(20)],
                "KPI-inflasjon": [round(-0.005 * (0.9**t), 4) for t in range(20)],
                "Styringsrente": [round(0.025 * (0.95**t), 4) for t in range(20)],
                "RER-gap": [round(-0.02 * (0.9**t), 4) for t in range(20)],
                "Boligpris-gap": [round(-0.008 * (0.9**t), 4) for t in range(20)],
            }
        },
        "fevd": {
            "bnp": {"TFP": [50.0]*20, "Pengepol.": [50.0]*20},
            "rer": {"Risikopremie": [80.0]*20, "Pengepol.": [20.0]*20},
        },
        "forecast": {"conditional": [], "baseline": [], "std_P": {}},
        "forecast_level": {
            "dates": ["2025-04-01", "2025-07-01"],
            "y": [0.1, 0.2], "pi": [0.3, 0.4], "i": [4.5, 4.3],
            "rer": [11.5, 11.6], "bolig": [0.5, 0.6],
            "y_bl": [0.12, 0.22], "pi_bl": [0.31, 0.41], "i_bl": [4.4, 4.2],
        },
        "hist_level": {
            "dates": ["2001-01-01", "2001-04-01"],
            "y": [0.1, -0.1], "pi": [2.5, 2.6], "i": [7.0, 6.5],
            "rer": [8.5, 8.6], "bolig": [0.2, 0.3],
        },
        "hist_decomp": {
            "y": {"TFP": [0.01, 0.02], "Pengepol.": [-0.01, -0.02]},
        },
        "active_shocks": {"TFP": 0.2, "Pengepol.": -0.5},
        "meta": {
            "eig_max": 0.9944, "n_states": 50, "n_shocks": 13,
            "last_obs_date": "2025-12-31",
        },
    }
    tmp = tmp_path_factory.mktemp("data")
    path = str(tmp / "test_analyse.json")
    with open(path, "w") as f:
        json.dump(data, f)
    return path


def test_build_creates_html(analyse_json, tmp_path):
    out = str(tmp_path / "dashboard.html")
    build_dashboard(analyse_json, out)
    assert os.path.exists(out)
    html = open(out).read()
    assert html.strip().startswith("<!DOCTYPE html")


def test_no_placeholders(analyse_json, tmp_path):
    out = str(tmp_path / "dashboard.html")
    build_dashboard(analyse_json, out)
    html = open(out).read()
    for ph in ["__IRF_DATA__", "__FEVD_DATA__", "__HIST_LEVEL_DATA__",
               "__HIST_DECOMP_DATA__", "__FCST_LEVEL_DATA__",
               "__ACTIVE_SHOCKS__", "__META_DATA__"]:
        assert ph not in html, f"Uerstattede placeholder: {ph}"


def test_html_contains_tabs(analyse_json, tmp_path):
    out = str(tmp_path / "dashboard.html")
    build_dashboard(analyse_json, out)
    html = open(out).read()
    for tab in ["Oversikt", "IRF", "FEVD", "Historikk", "Prognose", "Diagnostikk"]:
        assert tab in html, f"Mangler fane: {tab}"


def test_html_contains_data(analyse_json, tmp_path):
    out = str(tmp_path / "dashboard.html")
    build_dashboard(analyse_json, out)
    html = open(out).read()
    assert "Pengepol." in html
    assert "0.9944" in html
