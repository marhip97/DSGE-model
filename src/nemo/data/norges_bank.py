"""
Norges Bank Data API-klient for NEMO-prosjektet.

Henter pengepolitiske data (styringsrente, NIBOR, valutakurs, kreditt)
fra Norges Banks SDMX-JSON API. Alle kall caches i data/raw/.

Referanse: https://data.norges-bank.no/api/
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ─── Konstanter ────────────────────────────────────────────────────────────────

NB_BASE_URL = "https://data.norges-bank.no/api/data"
START_PERIOD = "2001-01-01"


def _finn_repo_root() -> Path:
    """Finner repo-rotkatalog ved å lete etter pyproject.toml."""
    kandidat = Path(__file__).resolve()
    for _ in range(10):
        if (kandidat / "pyproject.toml").exists():
            return kandidat
        kandidat = kandidat.parent
    raise FileNotFoundError("Fant ikke pyproject.toml — kan ikke bestemme repo-rot.")


def _ensure_raw_dir() -> Path:
    """Sørger for at data/raw/ eksisterer, returnerer stien."""
    raw_dir = _finn_repo_root() / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir


def _cache_path(series_key: str) -> Path:
    """Bygger cache-filsti for NB-serie med dagens dato."""
    raw_dir = _ensure_raw_dir()
    dato = datetime.now().strftime("%Y%m%d")
    safe_key = re.sub(r"[^a-zA-Z0-9_]", "_", series_key)
    return raw_dir / f"nb_{safe_key}_{dato}.json"


def _finn_siste_cache(series_key: str) -> Optional[Path]:
    """Finner nyeste cache-fil for gitt serie-nøkkel."""
    raw_dir = _ensure_raw_dir()
    safe_key = re.sub(r"[^a-zA-Z0-9_]", "_", series_key)
    kandidater = sorted(raw_dir.glob(f"nb_{safe_key}_*.json"), reverse=True)
    return kandidater[0] if kandidater else None


def _hent_nb_api(
    series_key: str,
    url: str,
    bruk_cache: bool = True,
) -> dict:
    """
    Henter SDMX-JSON-data fra Norges Banks API med cache-fallback.

    Parametere
    ----------
    series_key : str
        Identifikator for serien (brukes som cache-nøkkel).
    url : str
        Full API-URL.
    bruk_cache : bool
        Om eksisterende cache skal brukes direkte.

    Returnerer
    ----------
    dict
        Rådata fra NB API i SDMX-JSON-format.
    """
    cache_fil = _cache_path(series_key)

    if bruk_cache and cache_fil.exists():
        logger.info("Leser NB-serie %s fra cache: %s", series_key, cache_fil)
        with open(cache_fil, encoding="utf-8") as f:
            return json.load(f)

    logger.info("Henter NB-serie %s fra API: %s", series_key, url)

    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        with open(cache_fil, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        logger.info("Cachet NB-serie %s → %s", series_key, cache_fil)
        return data

    except Exception as exc:
        logger.warning(
            "NB API-kall feilet for %s: %s. Prøver cache-fallback.",
            series_key, exc
        )
        fallback = _finn_siste_cache(series_key)
        if fallback is not None:
            logger.info("Bruker cache-fallback: %s", fallback)
            with open(fallback, encoding="utf-8") as f:
                return json.load(f)
        raise RuntimeError(
            f"NB API feilet og ingen cache finnes for serie '{series_key}'."
        ) from exc


def _parse_sdmx_json(data: dict) -> pd.Series:
    """
    Parser SDMX-JSON-respons fra Norges Bank til en pd.Series.

    Parametere
    ----------
    data : dict
        SDMX-JSON-respons med struktur data.dataSets[0].series og
        data.structure.dimensions.observation[0].values.

    Returnerer
    ----------
    pd.Series
        Tidsserie med datostrenger som indeks og float-verdier.
        Datostrenger er i format 'YYYY-MM-DD' eller 'YYYY-MM'.
    """
    dataset = data["data"]["dataSets"][0]
    time_dim = data["data"]["structure"]["dimensions"]["observation"][0]
    times = [p["id"] for p in time_dim["values"]]

    serie_dict = dataset["series"]
    result: dict[str, float] = {}

    for _serie_key, serie_val in serie_dict.items():
        for t_idx_str, val_list in serie_val["observations"].items():
            t_idx = int(t_idx_str)
            t = times[t_idx]
            val = val_list[0] if val_list[0] is not None else np.nan
            result[t] = float(val)

    s = pd.Series(result).sort_index()
    return s


def _dagsdata_til_kvartalssnitt(s: pd.Series) -> pd.Series:
    """
    Konverterer dagsdata (YYYY-MM-DD indeks) til kvartalssnitt.

    Parametere
    ----------
    s : pd.Series
        Dagsdata med datostrenger eller DatetimeIndex som indeks.

    Returnerer
    ----------
    pd.Series
        Kvartalsvise gjennomsnitt med pd.Timestamp-indeks (siste dag i kvartal).
    """
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")
        s = s[s.index.notna()]

    s_kvartal = s.resample("QE").mean()
    return s_kvartal


def _maanedlig_til_kvartalssnitt(s: pd.Series) -> pd.Series:
    """
    Konverterer månedlige data til kvartalssnitt.

    Parametere
    ----------
    s : pd.Series
        Månedlige data med datostrenger (f.eks. 'YYYY-MM') eller DatetimeIndex.

    Returnerer
    ----------
    pd.Series
        Kvartalsvise gjennomsnitt med pd.Timestamp-indeks.
    """
    if not isinstance(s.index, pd.DatetimeIndex):
        # Prøv å parse 'YYYY-MM' format
        parsed = pd.to_datetime(s.index.str[:7] + "-01", errors="coerce")
        s = s.copy()
        s.index = parsed
        s = s[s.index.notna()]
        # Sett til siste dag i måneden
        s.index = s.index + pd.offsets.MonthEnd(0)

    s_kvartal = s.resample("QE").mean()
    return s_kvartal


def hent_styringsrente(bruk_cache: bool = True) -> pd.Series:
    """
    Henter styringsrenten fra Norges Bank (POLICY_RATE) og beregner kvartalssnitt.

    Resultat gir i_R_obs (annualisert styringsrente, ikke skalert).
    Pipeline deler på 400 for å konvertere til kvartalsverdier.

    Parametere
    ----------
    bruk_cache : bool
        Om cache skal brukes.

    Returnerer
    ----------
    pd.Series
        Kvartalsvise styringsrenteverdier (annualisert, i %).
        Indeks: pd.Timestamp (siste dag i kvartal).
    """
    url = (
        f"{NB_BASE_URL}/IR/B.KPRA.SD."
        f"?startPeriod={START_PERIOD}&format=sdmx-json&locale=no"
    )
    data = _hent_nb_api("POLICY_RATE", url, bruk_cache=bruk_cache)
    s = _parse_sdmx_json(data)
    s_kvartal = _dagsdata_til_kvartalssnitt(s)
    s_kvartal.name = "styringsrente"
    logger.info("Styringsrente hentet: %d kvartaler", len(s_kvartal))
    return s_kvartal


def hent_nibor_3m(bruk_cache: bool = True) -> pd.Series:
    """
    Henter 3M NIBOR fra Norges Bank og beregner kvartalssnitt.

    Parametere
    ----------
    bruk_cache : bool
        Om cache skal brukes.

    Returnerer
    ----------
    pd.Series
        Kvartalsvise NIBOR 3M-verdier (annualisert, i %).
        Indeks: pd.Timestamp (siste dag i kvartal).
    """
    url = (
        f"{NB_BASE_URL}/NIBOR/3M"
        f"?startPeriod={START_PERIOD}&format=sdmx-json&locale=no"
    )
    data = _hent_nb_api("NIBOR_3M", url, bruk_cache=bruk_cache)
    s = _parse_sdmx_json(data)
    s_kvartal = _dagsdata_til_kvartalssnitt(s)
    s_kvartal.name = "nibor_3m"
    logger.info("NIBOR 3M hentet: %d kvartaler", len(s_kvartal))
    return s_kvartal


def hent_valutakurs_importveid(bruk_cache: bool = True) -> pd.Series:
    """
    Henter importveid valutakurs I-44 fra Norges Bank og beregner kvartalssnitt.

    I-44 er NB's importveide kursindeks (44 handelspartnere), brukt som
    ds_obs (nominell valutakursendring) i NEMO-estimeringen.

    Parametere
    ----------
    bruk_cache : bool
        Om cache skal brukes.

    Returnerer
    ----------
    pd.Series
        Kvartalsvise importveid valutakurs-gjennomsnitt (I-44, NOK per valutakurv).
        Indeks: pd.Timestamp (siste dag i kvartal).
    """
    url = (
        f"{NB_BASE_URL}/EXR/B.I44.NOK.SP"
        f"?startPeriod={START_PERIOD}&format=sdmx-json&locale=no"
    )
    data = _hent_nb_api("EXR_B_I44_NOK_SP", url, bruk_cache=bruk_cache)
    s = _parse_sdmx_json(data)
    s_kvartal = _dagsdata_til_kvartalssnitt(s)
    s_kvartal.name = "importveid_kurs"
    logger.info("Importveid valutakurs I-44 hentet: %d kvartaler", len(s_kvartal))
    return s_kvartal


def hent_k2_husholdning(bruk_cache: bool = True) -> pd.Series:
    """
    Henter K2 husholdningenes kredittvekst fra Norges Bank.

    Månedlige data konverteres til kvartalssnitt.

    Parametere
    ----------
    bruk_cache : bool
        Om cache skal brukes.

    Returnerer
    ----------
    pd.Series
        Kvartalsvise K2-verdier (kredittvekst, % el. nivå).
        Indeks: pd.Timestamp (siste dag i kvartal).
    """
    url = (
        f"{NB_BASE_URL}/CREDIT_INDICATOR/K2.HH"
        f"?startPeriod={START_PERIOD}&format=sdmx-json&locale=no"
    )
    data = _hent_nb_api("CREDIT_INDICATOR_K2_HH", url, bruk_cache=bruk_cache)
    s = _parse_sdmx_json(data)

    # K2 er månedlig — konverter til kvartalssnitt
    s_kvartal = _maanedlig_til_kvartalssnitt(s)
    s_kvartal.name = "k2_husholdning"
    logger.info("K2 husholdning hentet: %d kvartaler", len(s_kvartal))
    return s_kvartal
