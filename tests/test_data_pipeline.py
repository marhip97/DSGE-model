"""
Tester for NEMO datapipeline.

Alle tester bruker syntetiske data eller mocked API-kall.
Ingen nettverkstilgang kreves.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from nemo.data.pipeline import (
    OBSERVASJONSVARIABLER,
    demean,
    hp_filter,
    log_diff,
    transformer_til_obs,
)
from nemo.data.ssb import ssb_kode_til_dato, _parse_json_stat2


# ─── Hjelpefunksjoner for syntetiske testdata ──────────────────────────────────

def _lag_syntetisk_nr(n: int = 100, start: str = "2001-03-31") -> pd.DataFrame:
    """Genererer syntetiske nasjonalregnskapsdata for testing."""
    rng = np.random.default_rng(42)
    indeks = pd.date_range(start, periods=n, freq="QE")
    base = 100.0 * np.exp(np.cumsum(rng.normal(0.005, 0.01, n)))
    return pd.DataFrame(
        {
            "bnp_fastland": base,
            "privat_konsum": 0.5 * base + rng.normal(0, 0.5, n),
            "bruttoinvesteringer": 0.2 * base + rng.normal(0, 1.0, n),
            "eksport": 0.23 * base + rng.normal(0, 0.8, n),
            "import_": 0.34 * base + rng.normal(0, 0.8, n),
        },
        index=indeks,
    )


def _lag_syntetisk_serie(n: int = 100, start: str = "2001-03-31", base: float = 100.0) -> pd.Series:
    """Genererer en syntetisk positiv nivå-tidsserie."""
    rng = np.random.default_rng(42)
    indeks = pd.date_range(start, periods=n, freq="QE")
    verdier = base * np.exp(np.cumsum(rng.normal(0.005, 0.01, n)))
    return pd.Series(verdier, index=indeks)


def _lag_syntetisk_rente(n: int = 100, start: str = "2001-03-31", nivaa: float = 2.5) -> pd.Series:
    """Genererer en syntetisk renteserie (annualisert %)."""
    rng = np.random.default_rng(99)
    indeks = pd.date_range(start, periods=n, freq="QE")
    verdier = np.clip(
        nivaa + np.cumsum(rng.normal(0, 0.1, n)),
        0.0, 10.0
    )
    return pd.Series(verdier, index=indeks)


def _lag_syntetisk_obs_df(n: int = 98) -> pd.DataFrame:
    """Bygger komplett syntetisk observasjonsmatrise med alle 14 variabler."""
    rng = np.random.default_rng(42)
    indeks = pd.date_range("2001-06-30", periods=n, freq="QE")
    data = {col: rng.normal(0, 0.01, n) for col in OBSERVASJONSVARIABLER}
    return pd.DataFrame(data, index=indeks)


# ─── Test: Kvartalsdatokonvertering ───────────────────────────────────────────

class TestSsbKodeTilDato:
    """Tester SSB-kvartalskode konvertering."""

    def test_k1_gir_31_mars(self):
        assert ssb_kode_til_dato("2001K1") == pd.Timestamp("2001-03-31")

    def test_k2_gir_30_juni(self):
        assert ssb_kode_til_dato("2001K2") == pd.Timestamp("2001-06-30")

    def test_k3_gir_30_september(self):
        assert ssb_kode_til_dato("2001K3") == pd.Timestamp("2001-09-30")

    def test_k4_gir_31_desember(self):
        assert ssb_kode_til_dato("2001K4") == pd.Timestamp("2001-12-31")

    def test_ulike_aar(self):
        assert ssb_kode_til_dato("2025K2") == pd.Timestamp("2025-06-30")


# ─── Test: HP-filter ──────────────────────────────────────────────────────────

class TestHpFilter:
    """Tester HP-filter-implementasjonen."""

    def test_syklus_har_gjennomsnitt_nær_null(self):
        """HP-syklus skal ha gjennomsnitt ≈ 0."""
        rng = np.random.default_rng(1)
        trend = np.linspace(0, 1, 80)
        syklus_input = 0.1 * rng.standard_normal(80)
        y = trend + syklus_input

        _, syklus = hp_filter(y, lam=1600)
        assert abs(syklus.mean()) < 0.05, f"Syklusgjennomsnitt = {syklus.mean():.4f}"

    def test_trend_og_syklus_summerer_til_original(self):
        """Trend + syklus = original tidsserie."""
        rng = np.random.default_rng(2)
        y = rng.standard_normal(60)
        trend, syklus = hp_filter(y, lam=1600)
        np.testing.assert_allclose(trend + syklus, y, atol=1e-10)

    def test_returnerer_to_arrays(self):
        """Funksjonen returnerer tuple med to arrays."""
        y = np.arange(1.0, 51.0)
        result = hp_filter(y)
        assert len(result) == 2
        assert result[0].shape == (50,)
        assert result[1].shape == (50,)

    def test_krever_minst_4_obs(self):
        """For kort tidsserie skal gi ValueError."""
        with pytest.raises(ValueError, match="minst 4"):
            hp_filter(np.array([1.0, 2.0, 3.0]))

    def test_konstant_serie_gir_null_syklus(self):
        """En konstant serie skal gi nær-null syklus."""
        y = np.ones(50) * 5.0
        _, syklus = hp_filter(y)
        np.testing.assert_allclose(syklus, 0.0, atol=1e-8)


# ─── Test: log_diff ───────────────────────────────────────────────────────────

class TestLogDiff:
    """Tester log-differanse-funksjonen."""

    def test_log_diff_gir_nan_i_første_obs(self):
        """Første observasjon skal alltid være NaN."""
        s = pd.Series([100.0, 102.0, 101.0, 103.0])
        result = log_diff(s)
        assert pd.isna(result.iloc[0])

    def test_log_diff_korrekt_verdi(self):
        """log_diff(e, e²) = 1."""
        s = pd.Series([np.e, np.e ** 2])
        result = log_diff(s)
        np.testing.assert_allclose(result.iloc[1], 1.0, atol=1e-10)

    def test_log_diff_bevarer_indeks(self):
        """Indeksen skal bevares."""
        indeks = pd.date_range("2001-03-31", periods=4, freq="QE")
        s = pd.Series([100.0, 102.0, 101.0, 104.0], index=indeks)
        result = log_diff(s)
        assert (result.index == indeks).all()


# ─── Test: Output-format ─────────────────────────────────────────────────────

class TestOutputFormat:
    """Tester at observasjonsmatrisen har korrekt format."""

    def test_14_kolonner_med_riktige_navn(self):
        """Observasjonsmatrise skal ha eksakt 14 kolonner med riktige navn."""
        df = _lag_syntetisk_obs_df()
        assert list(df.columns) == OBSERVASJONSVARIABLER, (
            f"Kolonner: {list(df.columns)}"
        )

    def test_kvartalsdatoer_som_timestamps(self):
        """Indeks skal være pd.Timestamp (siste dag i kvartal)."""
        df = _lag_syntetisk_obs_df()
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_kvartalsdatoer_er_kvartalsslutt(self):
        """Alle indeksdatoer skal være siste dag i et kvartal."""
        df = _lag_syntetisk_obs_df()
        for dato in df.index:
            periode = pd.Period(dato, "Q")
            forventet = periode.end_time.normalize()
            assert dato == forventet, f"{dato} er ikke kvartalsslutt"

    def test_ingen_nan_i_syntetisk_data(self):
        """Syntetisk observasjonsmatrise skal ikke ha NaN."""
        df = _lag_syntetisk_obs_df()
        assert not df.isna().any().any(), (
            f"NaN funnet i kolonner: {df.columns[df.isna().any()].tolist()}"
        )


# ─── Test: Demeaning ─────────────────────────────────────────────────────────

class TestDemeaning:
    """Tester at demeaning fungerer korrekt."""

    def test_alle_kolonner_gjennomsnitt_nær_null(self):
        """Etter demeaning skal alle kolonnegjennomsnnitt ≈ 0."""
        df = _lag_syntetisk_obs_df(n=80)
        demeaned, _ = demean(df)
        for kol in demeaned.columns:
            mean_val = demeaned[kol].mean()
            assert abs(mean_val) < 1e-10, (
                f"Kolonne {kol} har gjennomsnitt {mean_val:.2e} etter demeaning"
            )

    def test_demean_returnerer_korrekte_gjennomsnitt(self):
        """demean() skal returnere de faktiske gjennomsnittene som dict."""
        rng = np.random.default_rng(7)
        indeks = pd.date_range("2001-06-30", periods=40, freq="QE")
        data = {col: rng.normal(float(i), 0.1, 40) for i, col in enumerate(OBSERVASJONSVARIABLER)}
        df = pd.DataFrame(data, index=indeks)

        _, means = demean(df)
        for kol in OBSERVASJONSVARIABLER:
            np.testing.assert_allclose(
                means[kol], df[kol].mean(), atol=1e-12,
                err_msg=f"Feil gjennomsnitt for {kol}"
            )

    def test_demeaning_idempotent(self):
        """Dobbel demeaning skal gi samme resultat som én gang."""
        df = _lag_syntetisk_obs_df()
        demeaned1, _ = demean(df)
        demeaned2, _ = demean(demeaned1)
        pd.testing.assert_frame_equal(demeaned1, demeaned2)


# ─── Test: demean.json ────────────────────────────────────────────────────────

class TestDemeanJson:
    """Tester at nemo_demean.json-filen har korrekt innhold."""

    def test_demean_json_har_14_nokler(self, tmp_path: Path):
        """nemo_demean.json skal ha eksakt 14 nøkler."""
        means = {kol: float(i * 0.001) for i, kol in enumerate(OBSERVASJONSVARIABLER)}
        json_fil = tmp_path / "nemo_demean.json"
        json_fil.write_text(json.dumps(means), encoding="utf-8")

        innhold = json.loads(json_fil.read_text(encoding="utf-8"))
        assert len(innhold) == 14, f"Forventet 14 nøkler, fikk {len(innhold)}"

    def test_demean_json_har_riktige_nokler(self, tmp_path: Path):
        """nemo_demean.json skal ha alle 14 NEMO-variabelnavn som nøkler."""
        means = {kol: 0.0 for kol in OBSERVASJONSVARIABLER}
        json_fil = tmp_path / "nemo_demean.json"
        json_fil.write_text(json.dumps(means), encoding="utf-8")

        innhold = json.loads(json_fil.read_text(encoding="utf-8"))
        for kol in OBSERVASJONSVARIABLER:
            assert kol in innhold, f"Nøkkel '{kol}' mangler i demean.json"

    def test_demean_pipeline_lager_json(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """
        Kontroller at demean-operasjonen produserer gyldig JSON med 14 nøkler.

        Bruker demean() direkte uten nettverkskall.
        """
        df = _lag_syntetisk_obs_df(n=60)
        _, means = demean(df)

        json_fil = tmp_path / "nemo_demean.json"
        with open(json_fil, "w", encoding="utf-8") as f:
            json.dump(means, f)

        innhold = json.loads(json_fil.read_text(encoding="utf-8"))
        assert len(innhold) == 14


# ─── Test: Periodedekning ─────────────────────────────────────────────────────

class TestPeriodedekning:
    """Tester at dataserien dekker nødvendig periode."""

    def test_starter_ikke_etter_2001Q3(self):
        """Data skal starte senest 2001Q3 (juli–september 2001)."""
        df = _lag_syntetisk_obs_df(n=100)
        grense = pd.Timestamp("2001-09-30")
        assert df.index[0] <= grense, (
            f"Data starter {df.index[0].date()}, forventet ≤ {grense.date()}"
        )

    def test_slutter_ikke_foer_2020Q4(self):
        """Data skal slutte senest 2020Q4 (desember 2020)."""
        df = _lag_syntetisk_obs_df(n=100)
        # 100 kvartaler fra 2001-06-30 = ~2026
        grense = pd.Timestamp("2020-12-31")
        assert df.index[-1] >= grense, (
            f"Data slutter {df.index[-1].date()}, forventet ≥ {grense.date()}"
        )


# ─── Test: transformer_til_obs ────────────────────────────────────────────────

class TestTransformerTilObs:
    """Tester transformasjonslogikken."""

    def _lag_inputs(self, n: int = 80):
        """Hjelpemetode: lag syntetiske inndata til transformer_til_obs."""
        nr = _lag_syntetisk_nr(n=n)
        kpi = _lag_syntetisk_serie(n=n + 4)  # Litt lengre for sikkerhet
        lonn = _lag_syntetisk_serie(n=n)
        bolig = _lag_syntetisk_serie(n=n)
        styringsrente = _lag_syntetisk_rente(n=n, nivaa=2.5)
        nibor = _lag_syntetisk_rente(n=n, nivaa=3.0)
        nok_eur = _lag_syntetisk_serie(n=n, base=9.0)
        k2 = _lag_syntetisk_serie(n=n, base=3000.0)
        brent = _lag_syntetisk_serie(n=n, base=70.0)
        euro_bnp = _lag_syntetisk_serie(n=n, base=100.0)
        return nr, kpi, lonn, bolig, styringsrente, nibor, nok_eur, k2, brent, euro_bnp

    def test_returnerer_14_kolonner(self):
        """transformer_til_obs skal returnere 14 kolonner."""
        inputs = self._lag_inputs()
        obs = transformer_til_obs(*inputs)
        assert len(obs.columns) == 14

    def test_kolonnerekkefølge_korrekt(self):
        """Kolonner skal følge OBSERVASJONSVARIABLER-rekkefølgen."""
        inputs = self._lag_inputs()
        obs = transformer_til_obs(*inputs)
        assert list(obs.columns) == OBSERVASJONSVARIABLER

    def test_i_R_obs_skalert_med_400(self):
        """i_R_obs skal være styringsrente / 400."""
        nr, kpi, lonn, bolig, styringsrente, nibor, nok_eur, k2, brent, euro_bnp = self._lag_inputs()
        obs = transformer_til_obs(nr, kpi, lonn, bolig, styringsrente, nibor, nok_eur, k2, brent, euro_bnp)

        sr_aligned = styringsrente.reindex(nr.index)
        forventet = sr_aligned / 400.0
        mask = obs["i_R_obs"].notna() & forventet.notna()
        np.testing.assert_allclose(
            obs.loc[mask, "i_R_obs"].values,
            forventet.loc[mask].values,
            atol=1e-12,
        )

    def test_pi_obs_annualisert(self):
        """pi_obs skal være log_diff(KPI) × 4."""
        nr, kpi, lonn, bolig, styringsrente, nibor, nok_eur, k2, brent, euro_bnp = self._lag_inputs()
        obs = transformer_til_obs(nr, kpi, lonn, bolig, styringsrente, nibor, nok_eur, k2, brent, euro_bnp)

        kpi_aligned = kpi.reindex(nr.index)
        forventet_pi = log_diff(kpi_aligned) * 4
        mask = obs["pi_obs"].notna() & forventet_pi.notna()
        np.testing.assert_allclose(
            obs.loc[mask, "pi_obs"].values,
            forventet_pi.loc[mask].values,
            atol=1e-12,
        )


# ─── Test: API-fallback ───────────────────────────────────────────────────────

class TestApiFallback:
    """Tester at pipeline faller tilbake til cache hvis API feiler."""

    def _lag_ssb_cache_data(self, table_id: str, contents_code: str, n: int = 20) -> dict:
        """Bygger minimal gyldig JSON-stat2-respons for testing."""
        koder = [f"{2001 + t // 4}K{t % 4 + 1}" for t in range(n)]
        verdier = list(range(100, 100 + n))
        return {
            "id": ["ContentsCode", "Tid"],
            "size": [1, n],
            "value": verdier,
            "dimension": {
                "ContentsCode": {
                    "label": "ContentsCode",
                    "category": {
                        "index": {contents_code: 0},
                        "label": {contents_code: contents_code},
                    },
                },
                "Tid": {
                    "label": "Tid",
                    "category": {
                        "index": {k: i for i, k in enumerate(koder)},
                        "label": {k: k for k in koder},
                    },
                },
            },
        }

    def test_ssb_api_fallback_til_cache(self, tmp_path: Path):
        """Hvis SSB API feiler, skal cache-data brukes uten exception."""
        from nemo.data import ssb as ssb_modul

        table_id = "09189"
        contents_code = "BNPfastland"
        cache_data = self._lag_ssb_cache_data(table_id, contents_code, n=16)

        # Skriv cache-fil
        cache_fil = tmp_path / f"ssb_{table_id}_20240101.json"
        with open(cache_fil, "w", encoding="utf-8") as f:
            json.dump(cache_data, f)

        # Patch cache-søk og API-kall
        with (
            patch.object(ssb_modul, "_ensure_raw_dir", return_value=tmp_path),
            patch.object(ssb_modul, "_cache_path", return_value=tmp_path / f"ssb_{table_id}_99999999.json"),
            patch.object(ssb_modul, "_find_latest_cache", return_value=cache_fil),
            patch("requests.post", side_effect=Exception("Nettverksfeil")),
        ):
            # Skal ikke kaste exception
            result = ssb_modul.hent_ssb_tabell(table_id, {}, bruk_cache=False)

        assert result == cache_data

    def test_parse_json_stat2_returnerer_serie(self):
        """_parse_json_stat2 skal returnere pd.Series med korrekt lengde."""
        data = self._lag_ssb_cache_data("09189", "BNPfastland", n=12)
        s = _parse_json_stat2(data, "BNPfastland")
        assert isinstance(s, pd.Series)
        assert len(s) == 12

    def test_nb_api_fallback_til_cache(self, tmp_path: Path):
        """Hvis NB API feiler, skal cache-data brukes uten exception."""
        from nemo.data import norges_bank as nb_modul

        series_key = "POLICY_RATE"
        # Minimal SDMX-JSON-respons
        cache_data = {
            "data": {
                "dataSets": [
                    {"series": {"0:0": {"observations": {"0": [2.5], "1": [2.0]}}}}
                ],
                "structure": {
                    "dimensions": {
                        "observation": [
                            {"values": [{"id": "2020-01-01"}, {"id": "2020-04-01"}]}
                        ]
                    }
                },
            }
        }

        cache_fil = tmp_path / f"nb_{series_key}_20240101.json"
        with open(cache_fil, "w", encoding="utf-8") as f:
            json.dump(cache_data, f)

        with (
            patch.object(nb_modul, "_ensure_raw_dir", return_value=tmp_path),
            patch.object(nb_modul, "_cache_path", return_value=tmp_path / f"nb_{series_key}_99999999.json"),
            patch.object(nb_modul, "_finn_siste_cache", return_value=cache_fil),
            patch("requests.get", side_effect=Exception("Nettverksfeil")),
        ):
            result = nb_modul._hent_nb_api(series_key, "http://dummy", bruk_cache=False)

        assert result == cache_data

    def test_fred_api_fallback_til_cache(self, tmp_path: Path):
        """Hvis FRED API feiler, skal cache-data brukes uten exception."""
        from nemo.data import fred as fred_modul

        series_id = "DCOILBRENTEU"
        csv_innhold = "DATE,DCOILBRENTEU\n2001-01-01,25.0\n2001-04-01,27.0\n"

        cache_fil = tmp_path / f"fred_{series_id}_20240101.csv"
        cache_fil.write_text(csv_innhold, encoding="utf-8")

        with (
            patch.object(fred_modul, "_ensure_raw_dir", return_value=tmp_path),
            patch.object(fred_modul, "_cache_path", return_value=tmp_path / f"fred_{series_id}_99999999.csv"),
            patch.object(fred_modul, "_finn_siste_cache", return_value=cache_fil),
            patch("requests.get", side_effect=Exception("Nettverksfeil")),
        ):
            s = fred_modul._hent_fred_csv(series_id, bruk_cache=False)

        assert isinstance(s, pd.Series)
        assert len(s) == 2


# ─── Test: SSB JSON-stat2 parser ──────────────────────────────────────────────

class TestParseJsonStat2:
    """Tester JSON-stat2-parseren."""

    def test_parser_korrekte_verdier(self):
        """Parseren skal returnere korrekte numeriske verdier."""
        koder = ["2001K1", "2001K2", "2001K3"]
        data = {
            "id": ["ContentsCode", "Tid"],
            "size": [1, 3],
            "value": [100.0, 101.5, 99.8],
            "dimension": {
                "ContentsCode": {
                    "label": "ContentsCode",
                    "category": {
                        "index": {"BNPfastland": 0},
                        "label": {"BNPfastland": "BNP fastland"},
                    },
                },
                "Tid": {
                    "label": "Tid",
                    "category": {
                        "index": {k: i for i, k in enumerate(koder)},
                        "label": {k: k for k in koder},
                    },
                },
            },
        }
        s = _parse_json_stat2(data, "BNPfastland")
        np.testing.assert_allclose(s.values, [100.0, 101.5, 99.8])

    def test_parser_kaster_feil_for_ukjent_kode(self):
        """Parseren skal kaste ValueError for ukjente ContentsCode."""
        data = {
            "id": ["ContentsCode", "Tid"],
            "size": [1, 2],
            "value": [1.0, 2.0],
            "dimension": {
                "ContentsCode": {
                    "label": "ContentsCode",
                    "category": {"index": {"ANNEN_KODE": 0}, "label": {"ANNEN_KODE": "Annen"}},
                },
                "Tid": {
                    "label": "Tid",
                    "category": {
                        "index": {"2001K1": 0, "2001K2": 1},
                        "label": {"2001K1": "2001K1", "2001K2": "2001K2"},
                    },
                },
            },
        }
        with pytest.raises(ValueError, match="ContentsCode"):
            _parse_json_stat2(data, "BNPfastland")
