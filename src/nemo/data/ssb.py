"""
SSB Statistikkbanken JSON-stat API-klient for NEMO-prosjektet.

Henter kvartalsvise makroøkonomiske data fra SSB via JSON-stat2-formatet.
Alle kall caches lokalt i data/raw/ for robusthet mot nettverksbrudd.

Kravik og Mimir (2019), Appendiks A — variabeldefinisjoner.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ─── Konstanter ────────────────────────────────────────────────────────────────

SSB_BASE_URL = "https://data.ssb.no/api/v0/no/table"

# Absolutt sti til data/raw/ relativt til dette modulet
_REPO_ROOT = Path(__file__).parents[3]  # src/nemo/data -> repo root (3 nivåer opp... men src/ er under repo root)
# Prøv begge mulige stier
_RAW_DIR = _REPO_ROOT / "data" / "raw"
if not _RAW_DIR.parent.exists():
    # Fallback: se etter data/ fra src/nemo/data/
    _RAW_DIR = Path(__file__).parents[4] / "data" / "raw"


def _ensure_raw_dir() -> Path:
    """Sørger for at data/raw/ eksisterer, returnerer stien."""
    raw_dir = _RAW_DIR
    # Finn repo-root dynamisk
    candidate = Path(__file__).resolve()
    for _ in range(10):
        if (candidate / "pyproject.toml").exists():
            raw_dir = candidate / "data" / "raw"
            break
        candidate = candidate.parent
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir


def _cache_path(table_id: str) -> Path:
    """Bygger cache-filsti for SSB-tabell med dagens dato."""
    raw_dir = _ensure_raw_dir()
    dato = datetime.now().strftime("%Y%m%d")
    return raw_dir / f"ssb_{table_id}_{dato}.json"


def _find_latest_cache(table_id: str) -> Optional[Path]:
    """Finner nyeste cache-fil for gitt tabell-ID."""
    raw_dir = _ensure_raw_dir()
    kandidater = sorted(raw_dir.glob(f"ssb_{table_id}_*.json"), reverse=True)
    return kandidater[0] if kandidater else None


def _ssb_kvartal_koder(start_year: int = 2001, end_year: int = 2026) -> list[str]:
    """Genererer SSB-tidskoder som '2001K1', '2001K2', ..., '2025K4'."""
    return [
        f"{y}K{q}"
        for y in range(start_year, end_year)
        for q in range(1, 5)
    ]


def ssb_kode_til_dato(kode: str) -> pd.Timestamp:
    """
    Konverterer SSB-kvartalskode til siste dag i kvartalet.

    Eksempel: '2001K1' → Timestamp('2001-03-31')

    Parametere
    ----------
    kode : str
        SSB-kvartalskode, f.eks. '2001K1'.

    Returnerer
    ----------
    pd.Timestamp
        Siste dag i det aktuelle kvartalet.
    """
    year_str, q_str = kode.split("K")
    year = int(year_str)
    quarter = int(q_str)
    periode = pd.Period(f"{year}Q{quarter}", "Q")
    return periode.end_time.normalize()


def hent_ssb_tabell(
    table_id: str,
    query: dict,
    bruk_cache: bool = True,
) -> dict:
    """
    Henter rådata fra SSB Statistikkbanken (JSON-stat2).

    Prøver API-kall først. Hvis det feiler, faller tilbake til nyeste
    lokale cache-fil. Kaster RuntimeError hvis verken API- eller cache-data
    er tilgjengelig.

    Parametere
    ----------
    table_id : str
        SSB-tabell-ID, f.eks. '09189'.
    query : dict
        SSB API-spørring i JSON-stat2-format.
    bruk_cache : bool
        Om eksisterende cache skal leses direkte (uten API-kall).

    Returnerer
    ----------
    dict
        Rådata fra SSB i JSON-stat2-format.
    """
    cache_fil = _cache_path(table_id)

    if bruk_cache and cache_fil.exists():
        logger.info("Leser SSB-tabell %s fra cache: %s", table_id, cache_fil)
        with open(cache_fil, encoding="utf-8") as f:
            return json.load(f)

    url = f"{SSB_BASE_URL}/{table_id}"
    logger.info("Henter SSB-tabell %s fra API: %s", table_id, url)

    siste_exc: Exception | None = None
    for forsok in range(1, 4):   # 3 forsøk med eksponentiell backoff (2s, 4s)
        try:
            resp = requests.post(url, json=query, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            with open(cache_fil, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            logger.info("Cachet SSB-tabell %s → %s", table_id, cache_fil)
            return data

        except Exception as exc:
            siste_exc = exc
            if forsok < 3:
                vent = 2 ** forsok   # 2s, 4s
                logger.warning(
                    "SSB API-kall feilet (forsøk %d/3) for tabell %s: %s. Venter %ds.",
                    forsok, table_id, exc, vent
                )
                time.sleep(vent)
            else:
                logger.warning(
                    "SSB API-kall feilet (forsøk 3/3) for tabell %s: %s. Prøver cache-fallback.",
                    table_id, exc
                )

    fallback = _find_latest_cache(table_id)
    if fallback is not None:
        logger.info("Bruker cache-fallback: %s", fallback)
        with open(fallback, encoding="utf-8") as f:
            return json.load(f)
    raise RuntimeError(
        f"SSB API feilet (3 forsøk) og ingen cache finnes for tabell {table_id}."
    ) from siste_exc


def _parse_json_stat2(data: dict, contents_code: str) -> pd.Series:
    """
    Parser JSON-stat2-respons og returnerer tidsserie for én variabel.

    Parametere
    ----------
    data : dict
        JSON-stat2-respons fra SSB.
    contents_code : str
        ContentsCode-verdi for ønsket variabel (f.eks. 'BNPfastland').

    Returnerer
    ----------
    pd.Series
        Tidsserie med SSB-kvartalskoder som indeks og float-verdier.
    """
    dims = data.get("id", [])
    sizes = data.get("size", [])
    values = data.get("value", [])
    dimension = data.get("dimension", {})

    # Finn indekser for ContentsCode og Tid
    contents_idx = dims.index("ContentsCode") if "ContentsCode" in dims else None
    tid_idx = dims.index("Tid") if "Tid" in dims else None

    if contents_idx is None or tid_idx is None:
        raise ValueError(f"Forventet dimensjonene 'ContentsCode' og 'Tid' i {dims}")

    # Hent kategorier for ContentsCode og Tid
    contents_cats = list(dimension["ContentsCode"]["category"]["index"].keys())
    tid_cats = list(dimension["Tid"]["category"]["index"].keys())

    # Finn posisjon til ønsket ContentsCode
    if contents_code not in contents_cats:
        tilgjengelig = ", ".join(contents_cats)
        raise ValueError(
            f"ContentsCode '{contents_code}' ikke funnet. Tilgjengelig: {tilgjengelig}"
        )
    code_pos = contents_cats.index(contents_code)

    n_contents = sizes[contents_idx]
    n_tid = sizes[tid_idx]

    # Ekstraher verdier for valgt ContentsCode
    result = {}
    for t_pos, tid_kode in enumerate(tid_cats):
        if contents_idx < tid_idx:
            flat_idx = code_pos * n_tid + t_pos
        else:
            flat_idx = t_pos * n_contents + code_pos
        val = values[flat_idx]
        result[tid_kode] = float(val) if val is not None else np.nan

    return pd.Series(result, name=contents_code)


def _lag_ssb_query(contents_codes: list[str], start_year: int = 2001) -> dict:
    """Bygger SSB JSON-stat2 API-spørring. Bruker filter=all for Tid."""
    return {
        "query": [
            {
                "code": "ContentsCode",
                "selection": {"filter": "item", "values": contents_codes},
            },
            {
                "code": "Tid",
                "selection": {"filter": "all", "values": ["*"]},
            },
        ],
        "response": {"format": "json-stat2"},
    }


def hent_nasjonalregnskap(bruk_cache: bool = True) -> pd.DataFrame:
    """
    Henter kvartalsvise nasjonalregnskapsdata fra SSB tabell 09189.

    Volumindekser (2020=100) for BNP fastland, privat konsum,
    bruttoinvesteringer, eksport og import.

    Parametere
    ----------
    bruk_cache : bool
        Om cache skal brukes fremfor nytt API-kall.

    Returnerer
    ----------
    pd.DataFrame
        Kolonner: bnp_fastland, privat_konsum, bruttoinvesteringer,
                  eksport, import_ — indeks som pd.Timestamp (siste dag i kvartal).
    """
    table_id = "09189"
    codes = ["BNPfastland", "PK", "BINV", "EKSPORT", "IMPORT"]
    query = _lag_ssb_query(codes)

    data = hent_ssb_tabell(table_id, query, bruk_cache=bruk_cache)

    serier = {}
    navn_map = {
        "BNPfastland": "bnp_fastland",
        "PK": "privat_konsum",
        "BINV": "bruttoinvesteringer",
        "EKSPORT": "eksport",
        "IMPORT": "import_",
    }
    for code, kolonne in navn_map.items():
        s = _parse_json_stat2(data, code)
        s.index = [ssb_kode_til_dato(k) for k in s.index]
        serier[kolonne] = s

    df = pd.DataFrame(serier)
    df = df.sort_index()
    logger.info("Nasjonalregnskap hentet: %d kvartaler (%s–%s)", len(df), df.index[0].date(), df.index[-1].date())
    return df


def hent_kpi(bruk_cache: bool = True) -> pd.Series:
    """
    Henter månedlig KPI fra SSB tabell 03013 og beregner kvartalssnitt.

    Parametere
    ----------
    bruk_cache : bool
        Om cache skal brukes.

    Returnerer
    ----------
    pd.Series
        Kvartalsvis KPI-indeks med pd.Timestamp-indeks (siste dag i kvartal).
    """
    table_id = "03013"

    # Månedlige tidskoder
    query = {
        "query": [
            {
                "code": "ContentsCode",
                "selection": {"filter": "item", "values": ["KPI"]},
            },
            {
                "code": "Tid",
                "selection": {"filter": "all", "values": ["*"]},
            },
        ],
        "response": {"format": "json-stat2"},
    }

    data = hent_ssb_tabell(table_id, query, bruk_cache=bruk_cache)
    s = _parse_json_stat2_maaned(data, "KPI")

    # Konverter til kvartalssnitt
    s_kvartal = s.resample("QE").mean()
    logger.info("KPI hentet: %d kvartaler", len(s_kvartal))
    return s_kvartal


def hent_kpi_jae(bruk_cache: bool = True) -> pd.Series:
    """
    Henter månedlig KPI-JAE fra SSB tabell 10235 og beregner kvartalssnitt.

    KPI-JAE = Konsumprisindeksen justert for avgiftsendringer og uten energivarer.
    Tilsvarer NB MEMOs «kjerneinflasjon» og er det NBs NEMO bruker som pi_obs.

    Parametere
    ----------
    bruk_cache : bool
        Om cache skal brukes.

    Returnerer
    ----------
    pd.Series
        Kvartalsvis KPI-JAE-indeks med pd.Timestamp-indeks (siste dag i kvartal).
    """
    table_id = "10235"

    query = {
        "query": [
            {
                "code": "ContentsCode",
                "selection": {"filter": "item", "values": ["KpiJAE"]},
            },
            {
                "code": "Tid",
                "selection": {"filter": "all", "values": ["*"]},
            },
        ],
        "response": {"format": "json-stat2"},
    }

    data = hent_ssb_tabell(table_id, query, bruk_cache=bruk_cache)
    s = _parse_json_stat2_maaned(data, "KpiJAE")

    s_kvartal = s.resample("QE").mean()
    logger.info("KPI-JAE hentet: %d kvartaler", len(s_kvartal))
    return s_kvartal


def _parse_json_stat2_maaned(data: dict, contents_code: str) -> pd.Series:
    """
    Parser JSON-stat2-respons med månedlige tidskoder (format: '2001M01').

    Returnerer pd.Series med pd.DatetimeIndex (siste dag i måneden).
    """
    dims = data.get("id", [])
    sizes = data.get("size", [])
    values = data.get("value", [])
    dimension = data.get("dimension", {})

    contents_idx = dims.index("ContentsCode") if "ContentsCode" in dims else None
    tid_idx = dims.index("Tid") if "Tid" in dims else None

    contents_cats = list(dimension["ContentsCode"]["category"]["index"].keys())
    tid_cats = list(dimension["Tid"]["category"]["index"].keys())

    code_pos = contents_cats.index(contents_code)
    n_contents = sizes[contents_idx]
    n_tid = sizes[tid_idx]

    result = {}
    for t_pos, tid_kode in enumerate(tid_cats):
        if contents_idx < tid_idx:
            flat_idx = code_pos * n_tid + t_pos
        else:
            flat_idx = t_pos * n_contents + code_pos
        val = values[flat_idx]
        # Konverter '2001M01' til dato
        try:
            dato = pd.Timestamp(tid_kode.replace("M", "-") + "-01") + pd.offsets.MonthEnd(0)
        except Exception:
            continue
        result[dato] = float(val) if val is not None else np.nan

    s = pd.Series(result, name=contents_code).sort_index()
    return s


def hent_lonnsinndeks(bruk_cache: bool = True) -> pd.Series:
    """
    Henter kvartalsvise lønnsindeksdata fra SSB tabell 09786.

    Parametere
    ----------
    bruk_cache : bool
        Om cache skal brukes.

    Returnerer
    ----------
    pd.Series
        Kvartalsvise lønnsindeksverdier med pd.Timestamp-indeks.
    """
    table_id = "09786"

    # Prøv å hente metadata for å finne riktig ContentsCode
    # Bruk 'Lonnsinndeks' eller tilsvarende kode
    query_meta = {
        "query": [],
        "response": {"format": "json-stat2"},
    }

    # Kjente koder for lønnsindeks i 09786
    # Prøver LCI (Labour Cost Index) eller tilsvarende
    lonn_koder = ["LCI", "LONN", "Lonnsinndeks", "LI"]

    # Hent med en enkelt spørring
    kvartal_koder = _ssb_kvartal_koder(2001)
    query = {
        "query": [
            {
                "code": "Tid",
                "selection": {"filter": "item", "values": kvartal_koder},
            },
        ],
        "response": {"format": "json-stat2"},
    }

    try:
        data = hent_ssb_tabell(table_id, query, bruk_cache=bruk_cache)
    except Exception as exc:
        logger.warning("Feil ved henting av lønnsindeks: %s", exc)
        raise

    # Finn tilgjengelig ContentsCode
    dims = data.get("id", [])
    dimension = data.get("dimension", {})
    sizes = data.get("size", [])
    values = data.get("value", [])

    if "ContentsCode" in dims:
        contents_cats = list(dimension["ContentsCode"]["category"]["index"].keys())
        logger.info("Tilgjengelige lønnsindeks-koder: %s", contents_cats)
        code = contents_cats[0]  # Ta første tilgjengelige
        s = _parse_json_stat2(data, code)
    else:
        # Én-dimensjonal serie
        tid_cats = list(dimension["Tid"]["category"]["index"].keys())
        s = pd.Series(
            [float(v) if v is not None else np.nan for v in values],
            index=tid_cats,
            name="lonnsinndeks",
        )

    s.index = [ssb_kode_til_dato(k) for k in s.index]
    s = s.sort_index()
    logger.info("Lønnsindeks hentet: %d kvartaler", len(s))
    return s


def hent_boligprisindeks(bruk_cache: bool = True) -> pd.Series:
    """
    Henter kvartalsvise boligprisindeksdata fra SSB tabell 07241.

    Parametere
    ----------
    bruk_cache : bool
        Om cache skal brukes.

    Returnerer
    ----------
    pd.Series
        Kvartalsvise boligprisindeksverdier med pd.Timestamp-indeks.
    """
    table_id = "07241"
    kvartal_koder = _ssb_kvartal_koder(2001)

    query = {
        "query": [
            {
                "code": "Tid",
                "selection": {"filter": "item", "values": kvartal_koder},
            },
        ],
        "response": {"format": "json-stat2"},
    }

    try:
        data = hent_ssb_tabell(table_id, query, bruk_cache=bruk_cache)
    except Exception as exc:
        logger.warning("Feil ved henting av boligprisindeks: %s", exc)
        raise

    dims = data.get("id", [])
    dimension = data.get("dimension", {})
    sizes = data.get("size", [])
    values = data.get("value", [])

    if "ContentsCode" in dims:
        contents_cats = list(dimension["ContentsCode"]["category"]["index"].keys())
        logger.info("Tilgjengelige boligpris-koder: %s", contents_cats)
        code = contents_cats[0]
        s = _parse_json_stat2(data, code)
    else:
        tid_cats = list(dimension["Tid"]["category"]["index"].keys())
        s = pd.Series(
            [float(v) if v is not None else np.nan for v in values],
            index=tid_cats,
            name="boligprisindeks",
        )

    s.index = [ssb_kode_til_dato(k) for k in s.index]
    s = s.sort_index()
    s.name = "boligprisindeks"
    logger.info("Boligprisindeks hentet: %d kvartaler", len(s))
    return s
