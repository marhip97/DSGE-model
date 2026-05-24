"""
FRED (Federal Reserve Bank of St. Louis) API-klient for NEMO-prosjektet.

Henter oljepris (Brent) og handelspartner-BNP (eurosonens BNP som proxy).
Alle kall caches i data/raw/ for robusthet mot nettverksbrudd.

Referanse: https://fred.stlouisfed.org/
"""

from __future__ import annotations

import io
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ─── Konstanter ────────────────────────────────────────────────────────────────

FRED_CSV_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# Serie-IDer
BRENT_ID = "DCOILBRENTEU"          # Brent crude oil, daglig
EURO_GDP_ID = "CLVMNACSCAB1GQEU272020"   # Euro Area GDP, chain-linked, kvartal


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


def _cache_path(series_id: str) -> Path:
    """Bygger cache-filsti for FRED-serie med dagens dato."""
    raw_dir = _ensure_raw_dir()
    dato = datetime.now().strftime("%Y%m%d")
    return raw_dir / f"fred_{series_id}_{dato}.csv"


def _finn_siste_cache(series_id: str) -> Optional[Path]:
    """Finner nyeste cache-fil for gitt serie-ID."""
    raw_dir = _ensure_raw_dir()
    kandidater = sorted(raw_dir.glob(f"fred_{series_id}_*.csv"), reverse=True)
    return kandidater[0] if kandidater else None


def _hent_fred_csv(
    series_id: str,
    bruk_cache: bool = True,
    start_date: str = "2001-01-01",
) -> pd.Series:
    """
    Henter FRED CSV-data med cache-fallback.

    Bruker fredgraph.csv-endepunktet som ikke krever API-nøkkel.
    Hvis FRED_API_KEY er satt i miljøet, legges den til som parameter.

    Parametere
    ----------
    series_id : str
        FRED serie-ID, f.eks. 'DCOILBRENTEU'.
    bruk_cache : bool
        Om eksisterende cache skal brukes direkte.
    start_date : str
        Startdato for data (format: 'YYYY-MM-DD').

    Returnerer
    ----------
    pd.Series
        Tidsserie med pd.DatetimeIndex og float-verdier.
    """
    cache_fil = _cache_path(series_id)

    if bruk_cache and cache_fil.exists():
        logger.info("Leser FRED-serie %s fra cache: %s", series_id, cache_fil)
        df = pd.read_csv(cache_fil, index_col=0, parse_dates=True)
        s = df.iloc[:, 0].copy()
        s.index = pd.to_datetime(s.index)
        return s

    logger.info("Henter FRED-serie %s fra API", series_id)

    params: dict = {"id": series_id, "vintage_date": ""}
    api_key = os.environ.get("FRED_API_KEY", "")
    if api_key:
        params["api_key"] = api_key

    url = f"{FRED_CSV_BASE}?id={series_id}"
    if start_date:
        # fredgraph.csv støtter ikke startdato-parameter,
        # men FRED API gjør det. Filtrerer etter nedlasting.
        pass

    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()

        df = pd.read_csv(
            io.StringIO(resp.text),
            index_col=0,
            parse_dates=True,
            na_values=[".", ""],
        )
        s = df.iloc[:, 0].copy()
        s.index = pd.to_datetime(s.index)
        s.name = series_id

        # Filtrer fra startdato
        s = s[s.index >= pd.Timestamp(start_date)]

        # Cache til fil
        cache_fil.write_text(resp.text, encoding="utf-8")
        logger.info("Cachet FRED-serie %s → %s", series_id, cache_fil)
        return s

    except Exception as exc:
        logger.warning(
            "FRED API-kall feilet for %s: %s. Prøver cache-fallback.",
            series_id, exc
        )
        fallback = _finn_siste_cache(series_id)
        if fallback is not None:
            logger.info("Bruker cache-fallback: %s", fallback)
            df = pd.read_csv(fallback, index_col=0, parse_dates=True)
            s = df.iloc[:, 0].copy()
            s.index = pd.to_datetime(s.index)
            s = s[s.index >= pd.Timestamp(start_date)]
            return s
        raise RuntimeError(
            f"FRED API feilet og ingen cache finnes for serie '{series_id}'."
        ) from exc


def hent_brent_oljepris(bruk_cache: bool = True) -> pd.Series:
    """
    Henter Brent crude oil-pris (daglig) fra FRED og beregner kvartalssnitt.

    Serie: DCOILBRENTEU (Brent Crude Oil Prices: Europe, USD per barrel)

    Parametere
    ----------
    bruk_cache : bool
        Om cache skal brukes.

    Returnerer
    ----------
    pd.Series
        Kvartalsvise Brent-oljepriser (USD per fat).
        Indeks: pd.Timestamp (siste dag i kvartal).
    """
    s = _hent_fred_csv(BRENT_ID, bruk_cache=bruk_cache)

    # Dagsdata → kvartalssnitt
    s_kvartal = s.resample("QE").mean()
    s_kvartal.name = "brent_usd"
    logger.info("Brent oljepris hentet: %d kvartaler", len(s_kvartal))
    return s_kvartal


def hent_euro_bnp(bruk_cache: bool = True) -> pd.Series:
    """
    Henter euro-sone BNP (kvartalsvise chain-linked volumer) fra FRED.

    Brukes som proxy for norske handelspartneres BNP.
    Serie: CLVMNACSCAB1GQEU272020 (Euro Area GDP, chain-linked, 2020=100)

    Alternativt forsøker NAEXKP01EZQ661S (OECD Euro Area GDP) hvis primær feiler.

    Parametere
    ----------
    bruk_cache : bool
        Om cache skal brukes.

    Returnerer
    ----------
    pd.Series
        Kvartalsvise BNP-volumer for euro-sonen.
        Indeks: pd.Timestamp (siste dag i kvartal).
    """
    try:
        s = _hent_fred_csv(EURO_GDP_ID, bruk_cache=bruk_cache)
    except Exception as exc:
        logger.warning(
            "Primær euro-BNP-serie feilet (%s): %s. Prøver alternativ.",
            EURO_GDP_ID, exc
        )
        alt_id = "NAEXKP01EZQ661S"
        s = _hent_fred_csv(alt_id, bruk_cache=bruk_cache)
        s.name = EURO_GDP_ID

    # Kvartalsvise data trenger ikke resample, men sørg for kvartalsslutt-indeks
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)

    # Konverter til kvartalsslutt hvis nødvendig
    if not all(s.index == s.index.to_period("Q").to_timestamp("Q")):
        s = s.resample("QE").last()

    s.name = "bnp_handelspartner"
    logger.info("Euro-sone BNP hentet: %d kvartaler", len(s))
    return s
