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

# PxWeb v2 API — brukes for nasjonalregnskap (09190) fra og med NR23-revisjonen
_SSB_PXWEB_V2 = "https://data.ssb.no/api/pxwebapi/v2/tables"

# Bekreftet aggregat-koder i SSB tabell 09190 (NR23-revisjon, PxWeb v2)
_NR_KODER = {
    "bnp_fastland": "bnpb.nr23_9fn",
}

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
            # Logg HTTP-statuskode og responskropp for diagnose
            body = ""
            try:
                body = resp.text[:500]   # type: ignore[possibly-undefined]
            except Exception:
                pass
            if forsok < 3:
                vent = 2 ** forsok   # 2s, 4s
                logger.warning(
                    "SSB API-kall feilet (forsøk %d/3) for tabell %s: %s | body: %s. Venter %ds.",
                    forsok, table_id, exc, body, vent
                )
                time.sleep(vent)
            else:
                logger.warning(
                    "SSB API-kall feilet (forsøk 3/3) for tabell %s: %s | body: %s. Prøver cache-fallback.",
                    table_id, exc, body
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
        # Logg alle tilgjengelige dimensjoner for diagnose
        alle_dims = {d: list(dimension.get(d, {}).get("category", {}).get("index", {}).keys())[:10]
                     for d in dims}
        logger.error("Mangler ContentsCode/Tid. Tilgjengelige dims: %s", alle_dims)
        raise ValueError(f"Forventet dimensjonene 'ContentsCode' og 'Tid' i {dims}")

    # Hent kategorier for ContentsCode og Tid
    contents_cats = list(dimension["ContentsCode"]["category"]["index"].keys())
    tid_cats = list(dimension["Tid"]["category"]["index"].keys())

    # Finn posisjon til ønsket ContentsCode
    if contents_code not in contents_cats:
        tilgjengelig = ", ".join(contents_cats)
        # Dump ALLE koder i ALLE dimensjoner for å kunne oppdatere hent_nasjonalregnskap
        alle_dims = {d: list(dimension.get(d, {}).get("category", {}).get("index", {}).keys())
                     for d in dims if d not in ("Tid",)}
        label_dims = {d: list(dimension.get(d, {}).get("category", {}).get("label", {}).values())
                      for d in dims if d not in ("Tid",)}
        logger.error(
            "ContentsCode '%s' ikke funnet. Tilgjengelig: [%s]. "
            "ALLE dims/koder: %s. Labels: %s",
            contents_code, tilgjengelig, alle_dims, label_dims
        )
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
    """Bygger SSB JSON-stat2 API-spørring uten filtre — returnerer all data fra tabellen."""
    return {"query": [], "response": {"format": "json-stat2"}}


def hent_nasjonalregnskap(bruk_cache: bool = True) -> pd.DataFrame:
    """
    Henter kvartalsvise nasjonalregnskapsdata fra SSB tabell 09190 (PxWeb v2).

    Bruker GET-kall med ContentsCode=Faste (faste 2023-priser, mill. kr).
    Tabell 09190 = kvartalsvise tall; 09189 = årslige; 11721 = månedlige.

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
    table_id = "09190"
    cache_fil = _cache_path(table_id)

    if bruk_cache and cache_fil.exists():
        logger.info("Leser SSB-tabell %s fra cache: %s", table_id, cache_fil)
        with open(cache_fil, encoding="utf-8") as f:
            return _parse_nr_ny_struktur(json.load(f))

    url = f"{_SSB_PXWEB_V2}/{table_id}/data"
    params = {
        "lang": "no",
        "valueCodes[Makrost]": "*",           # alle aggregater, filtrer lokalt
        "valueCodes[ContentsCode]": "Faste",  # faste 2023-priser (rene kvartalstall)
        "valueCodes[Tid]": "top(200)",         # siste 200 kvartaler (> 50 år)
        "outputFormat": "json-stat2",
    }

    logger.info("Henter SSB-tabell %s (PxWeb v2 GET): %s", table_id, url)
    siste_exc: Exception | None = None
    for forsok in range(1, 4):
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            with open(cache_fil, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            logger.info("Cachet SSB-tabell %s → %s", table_id, cache_fil)
            return _parse_nr_ny_struktur(data)
        except Exception as exc:
            siste_exc = exc
            body = ""
            try:
                body = resp.text[:500]  # type: ignore[possibly-undefined]
            except Exception:
                pass
            if forsok < 3:
                vent = 2 ** forsok
                logger.warning(
                    "SSB PxWeb v2 feilet (forsøk %d/3) for %s: %s | body: %s. Venter %ds.",
                    forsok, table_id, exc, body, vent,
                )
                time.sleep(vent)
            else:
                logger.warning(
                    "SSB PxWeb v2 feilet (forsøk 3/3) for %s: %s | body: %s. Prøver cache.",
                    table_id, exc, body,
                )

    fallback = _find_latest_cache(table_id)
    if fallback is not None:
        logger.info("Cache-fallback: %s", fallback)
        with open(fallback, encoding="utf-8") as f:
            return _parse_nr_ny_struktur(json.load(f))
    raise RuntimeError(
        f"SSB PxWeb v2 feilet (3 forsøk) og ingen cache for tabell {table_id}."
    ) from siste_exc


def _parse_nr_ny_struktur(data: dict) -> pd.DataFrame:
    """
    Parser SSB nasjonalregnskap (tabell 09190) i NR23-struktur.

    ContentsCode = måletype (Faste/Priser/Volum/Endringer),
    aggregater i separat dimensjon (Makrost).
    Brukes med PxWeb v2 GET-kall der ContentsCode=Faste er forhåndsfiltrert.
    """
    dims = data.get("id", [])
    sizes = data.get("size", [])
    values = data.get("value", [])
    dimension = data.get("dimension", {})

    if "Tid" not in dims:
        raise ValueError(f"Fant ikke Tid-dimensjon. Dims: {dims}")
    tid_idx = dims.index("Tid")
    tid_cats = list(dimension["Tid"]["category"]["index"].keys())

    if "ContentsCode" not in dims:
        raise ValueError(f"Fant ikke ContentsCode. Dims: {dims}")
    cc_idx = dims.index("ContentsCode")
    cc_cats = list(dimension["ContentsCode"]["category"]["index"].keys())
    # Foretrekk 'Faste' (faste priser) fremfor 'Volum' (årslig endring)
    volum_code = next((c for c in cc_cats if "aste" in c), None) or \
                 next((c for c in cc_cats if "olum" in c), cc_cats[0])
    volum_pos = cc_cats.index(volum_code)
    logger.info("Bruker ContentsCode='%s'", volum_code)

    agg_dims = [d for d in dims if d not in ("ContentsCode", "Tid")]
    if not agg_dims:
        raise ValueError(f"Fant ingen aggregat-dimensjon i {dims}")
    agg_dim = agg_dims[0]
    agg_idx = dims.index(agg_dim)
    agg_cats = list(dimension[agg_dim]["category"]["index"].keys())
    agg_labels = {k: v for k, v in zip(
        agg_cats,
        dimension[agg_dim]["category"].get("label", {}).values()
    )}
    logger.info("Aggregat-dimensjon: '%s' med %d serier", agg_dim, len(agg_cats))
    logger.info("Alle tilgjengelige aggregater i '%s':", agg_dim)
    for _k, _v in agg_labels.items():
        logger.info("  %s: %s", _k, _v)

    # Label-fallback for serier uten hardkodet kode
    _label_fallback: dict[str, list[str]] = {
        "privat_konsum":      ["konsum i husholdninger og ideelle", "konsum i husholdninger",
                               "privat konsum"],
        "bruttoinvesteringer":["bruttoinvestering i alt", "bruttoinvesteringer i alt"],
        "eksport":            ["eksport i alt", "eksport"],
        "import_":            ["import i alt", "import"],
        # BNP Fastlands-Norge — bruker hardkodet kode fra _NR_KODER, label kun som fallback
        "bnp_fastland":       ["bruttonasjonalprodukt fastlands", "bnp fastlands"],
    }

    def finn_kode(kolonne: str) -> str:
        # 1) Direkte kodeoppslag (bekreftet av bruker)
        hardkodet = _NR_KODER.get(kolonne)
        if hardkodet and hardkodet in agg_cats:
            logger.info("  '%s' → '%s' (hardkodet kode)", kolonne, hardkodet)
            return hardkodet
        # 2) Label-basert matching (fallback)
        for kode, label in agg_labels.items():
            for soek in _label_fallback.get(kolonne, []):
                if soek.lower() in label.lower():
                    logger.info("  Matchet '%s' → kode='%s', label='%s'", kolonne, kode, label)
                    return kode
        raise ValueError(
            f"Fant ikke '{kolonne}' (hardkodet kode: {hardkodet!r}, "
            f"label-søk: {_label_fallback.get(kolonne)}). "
            f"Alle tilgjengelige aggregater: {list(agg_labels.items())}"
        )

    n = [sizes[i] for i in range(len(dims))]

    def flat_idx(agg_pos: int, cc_pos: int, t_pos: int) -> int:
        """Beregner flatindeks for 3-dim array med dims-rekkefølge."""
        idx = 0
        pos = [0] * len(dims)
        pos[agg_idx] = agg_pos
        pos[cc_idx]  = cc_pos
        pos[tid_idx] = t_pos
        stride = 1
        for d in reversed(range(len(dims))):
            idx += pos[d] * stride
            stride *= sizes[d]
        return idx

    serier = {}
    for kolonne in _label_fallback:
        kode = finn_kode(kolonne)
        agg_pos = agg_cats.index(kode)
        serie = {}
        for t_pos, tid_kode in enumerate(tid_cats):
            if "K" not in tid_kode:   # hopp over årslige tidskoder (nye i NR23-tabellen)
                continue
            fi = flat_idx(agg_pos, volum_pos, t_pos)
            val = values[fi]
            serie[tid_kode] = float(val) if val is not None else np.nan
        s = pd.Series(serie, name=kolonne)
        s.index = [ssb_kode_til_dato(k) for k in s.index]
        serier[kolonne] = s
        logger.info("  %s ← kode '%s' ('%s')", kolonne, kode, agg_labels[kode])

    df = pd.DataFrame(serier).sort_index()
    logger.info("Nasjonalregnskap (ny struktur) hentet: %d kvartaler (%s–%s)",
                len(df), df.index[0].date(), df.index[-1].date())
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

    query = {"query": [], "response": {"format": "json-stat2"}}
    data = hent_ssb_tabell(table_id, query, bruk_cache=bruk_cache)

    # Finn riktig ContentsCode — SSB har skiftet navn over tid
    cc_cats = list(data.get("dimension", {}).get("ContentsCode", {})
                       .get("category", {}).get("index", {}).keys())
    logger.info("KPI (03013) tilgjengelige ContentsCodes: %s", cc_cats)
    kpi_kode = next(
        (c for c in cc_cats if c.upper() in ("KPI", "KPIALT", "KONSUMPRISER")),
        cc_cats[0] if cc_cats else "KPI",
    )
    logger.info("Bruker ContentsCode='%s' for KPI", kpi_kode)
    s = _parse_json_stat2_maaned(data, kpi_kode)

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
    table_id = "14706"  # erstatter 05327/10235 fra 10. februar 2026

    query = {"query": [], "response": {"format": "json-stat2"}}

    data = hent_ssb_tabell(table_id, query, bruk_cache=bruk_cache)

    cc_cats = list(data.get("dimension", {}).get("ContentsCode", {})
                       .get("category", {}).get("index", {}).keys())
    logger.info("KPI-JAE (14706) tilgjengelige ContentsCodes: %s", cc_cats)
    jae_kode = next(
        (c for c in cc_cats if "JAE" in c.upper() or "JAE" in c),
        cc_cats[0] if cc_cats else "KpiJAE",
    )
    logger.info("Bruker ContentsCode='%s' for KPI-JAE", jae_kode)
    s = _parse_json_stat2_maaned(data, jae_kode)

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

    if contents_code not in contents_cats:
        logger.error(
            "ContentsCode '%s' ikke funnet i månedstabell. Tilgjengelig: %s",
            contents_code, contents_cats,
        )
        raise ValueError(
            f"ContentsCode '{contents_code}' ikke funnet. Tilgjengelig: {contents_cats}"
        )
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
    Henter kvartalsvise lønnsindeksdata fra SSB tabell 11654 (PxWeb v2).

    Tabell 11654 er kvartalsvis erstatning for nedlagt lønnsindeks (09786).
    Bruker GjMdTotalIndeks (indeks for gjennomsnittlig total månedslønn),
    alle næringer (NACE2007=00-99), hele landet (Region=Ialt).
    Basisperiode: 1. kvartal 2024 = 100.

    Parametere
    ----------
    bruk_cache : bool
        Om cache skal brukes.

    Returnerer
    ----------
    pd.Series
        Kvartalsvise lønnsindeksverdier med pd.Timestamp-indeks.
    """
    table_id = "11654"
    cache_fil = _cache_path(table_id)

    if bruk_cache and cache_fil.exists():
        logger.info("Leser SSB-tabell %s fra cache: %s", table_id, cache_fil)
        with open(cache_fil, encoding="utf-8") as f:
            data = json.load(f)
    else:
        url = f"{_SSB_PXWEB_V2}/{table_id}/data"
        params = {
            "lang": "no",
            "valueCodes[NACE2007]": "00-99",        # alle næringer
            "valueCodes[Region]":   "Ialt",          # hele landet
            "valueCodes[ContentsCode]": "GjMdTotalIndeks",  # total lønnsindeks
            "valueCodes[Tid]": "*",                  # alle kvartaler
            "outputFormat": "json-stat2",
        }
        logger.info("Henter SSB-tabell %s (PxWeb v2 GET)", table_id)
        siste_exc: Exception | None = None
        data = None
        for forsok in range(1, 4):
            try:
                resp = requests.get(url, params=params, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                with open(cache_fil, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)
                logger.info("Cachet SSB-tabell %s → %s", table_id, cache_fil)
                break
            except Exception as exc:
                siste_exc = exc
                body = ""
                try:
                    body = resp.text[:300]  # type: ignore[possibly-undefined]
                except Exception:
                    pass
                if forsok < 3:
                    vent = 2 ** forsok
                    logger.warning("SSB 11654 feilet (forsøk %d/3): %s | body: %s. Venter %ds.",
                                   forsok, exc, body, vent)
                    time.sleep(vent)
        if data is None:
            fallback = _find_latest_cache(table_id)
            if fallback:
                with open(fallback, encoding="utf-8") as f:
                    data = json.load(f)
            else:
                raise RuntimeError(
                    f"SSB PxWeb v2 feilet og ingen cache for {table_id}."
                ) from siste_exc

    # Med én verdi per dim utenom Tid er flat-arrayen direkte tidsserien
    dimension = data.get("dimension", {})
    values = data.get("value", [])
    tid_cats = list(dimension["Tid"]["category"]["index"].keys())
    s = pd.Series(
        {k: float(v) if v is not None else np.nan
         for k, v in zip(tid_cats, values)},
        name="lonnsinndeks",
    )
    s = s[[k for k in s.index if "K" in str(k)]]
    s.index = [ssb_kode_til_dato(k) for k in s.index]
    s = s.sort_index()
    logger.info("Lønnsindeks hentet: %d kvartaler (%s–%s)",
                len(s),
                s.index[0].date() if len(s) else "?",
                s.index[-1].date() if len(s) else "?")
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
    query = {"query": [], "response": {"format": "json-stat2"}}

    data = hent_ssb_tabell(table_id, query, bruk_cache=bruk_cache)

    dims = data.get("id", [])
    dimension = data.get("dimension", {})
    values = data.get("value", [])

    if "ContentsCode" in dims:
        contents_cats = list(dimension["ContentsCode"]["category"]["index"].keys())
        logger.info("Boligprisindeks (07241) tilgjengelige koder: %s", contents_cats)
        code = contents_cats[0]
        s = _parse_json_stat2(data, code)
    else:
        tid_cats = list(dimension["Tid"]["category"]["index"].keys())
        s = pd.Series(
            [float(v) if v is not None else np.nan for v in values],
            index=tid_cats,
            name="boligprisindeks",
        )

    s = s[[k for k in s.index if "K" in str(k)]]  # behold kun kvartalskoder
    s.index = [ssb_kode_til_dato(k) for k in s.index]
    s = s.sort_index()
    s.name = "boligprisindeks"
    logger.info("Boligprisindeks hentet: %d kvartaler", len(s))
    return s


def hent_nibor_3m(bruk_cache: bool = True) -> pd.Series:
    """
    Henter 3M NIBOR fra SSB tabell 10701 og beregner kvartalssnitt.

    Tabell 10701 inneholder månedlige NIBOR og foliorente-observasjoner.
    Dimensjonskoder oppdages automatisk via minimal metadata-query.
    Kvartalsverdier beregnes som gjennomsnitt av tre måneder.

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
    table_id = "10701"
    url = f"{_SSB_PXWEB_V2}/{table_id}/data"

    # Steg 1: oppdage dimensjonskoder via minimal query
    try:
        meta_resp = requests.get(
            url,
            params={"lang": "no", "valueCodes[Tid]": "top(1)", "outputFormat": "json-stat2"},
            timeout=30,
        )
        meta_resp.raise_for_status()
        meta = meta_resp.json()
    except Exception as exc:
        logger.warning("SSB 10701 metadata-kall feilet: %s. Prøver uten filter.", exc)
        meta = None

    params: dict[str, str] = {
        "lang": "no",
        "valueCodes[Tid]": "*",
        "outputFormat": "json-stat2",
    }

    if meta is not None:
        dim = meta.get("dimension", {})
        dims = meta.get("id", [])
        logger.info("SSB 10701 dimensjoner: %s", dims)
        for d in dims:
            koder = list(dim.get(d, {}).get("category", {}).get("index", {}).keys())
            labels = dim.get(d, {}).get("category", {}).get("label", {})
            logger.info("  %s: %s", d, {k: labels.get(k, k) for k in koder})

        # Finn rentedimensjonen (alt unntatt Tid)
        rente_dims = [d for d in dims if d != "Tid"]
        for rente_dim in rente_dims:
            nibor_kode = _ssb_oppdag_kode(
                dim, rente_dim, ["nibor", "3 mån", "3m", "3 month"],
            )
            if nibor_kode:
                params[f"valueCodes[{rente_dim}]"] = nibor_kode
                logger.info("SSB 10701: bruker %s=%s for NIBOR 3M", rente_dim, nibor_kode)
                break

    cache_fil = _cache_path(table_id)
    data = None

    if bruk_cache and cache_fil.exists():
        logger.info("Leser SSB-tabell %s fra cache: %s", table_id, cache_fil)
        with open(cache_fil, encoding="utf-8") as f:
            data = json.load(f)
    else:
        siste_exc: Exception | None = None
        for forsok in range(1, 4):
            try:
                resp = requests.get(url, params=params, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                with open(cache_fil, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)
                logger.info("Cachet SSB-tabell %s → %s", table_id, cache_fil)
                break
            except Exception as exc:
                siste_exc = exc
                body = ""
                try:
                    body = resp.text[:300]  # type: ignore[possibly-undefined]
                except Exception:
                    pass
                if forsok < 3:
                    vent = 2 ** forsok
                    logger.warning(
                        "SSB 10701 feilet (forsøk %d/3): %s | body: %s. Venter %ds.",
                        forsok, exc, body, vent,
                    )
                    time.sleep(vent)
        if data is None:
            fallback = _find_latest_cache(table_id)
            if fallback is not None:
                logger.info("SSB 10701: bruker cache-fallback %s", fallback)
                with open(fallback, encoding="utf-8") as f:
                    data = json.load(f)
            else:
                raise RuntimeError(
                    f"SSB PxWeb v2 feilet og ingen cache for {table_id}."
                ) from siste_exc

    # Parse: Tid-dimensjonen, flat array = tidsserie (én rentekode valgt)
    dims_data = data.get("id", [])
    dimension = data.get("dimension", {})
    values = data.get("value", [])
    sizes = data.get("size", [])

    tid_idx = dims_data.index("Tid")
    tid_cats = list(dimension["Tid"]["category"]["index"].keys())
    n_tid = sizes[tid_idx]

    # Antall kombinasjoner langs andre dimensjoner
    n_andre = 1
    for i, sz in enumerate(sizes):
        if i != tid_idx:
            n_andre *= sz

    if n_andre == 1:
        obs_values = values[:n_tid]
    else:
        logger.warning("SSB 10701: %d kombinasjoner — tar første rad.", n_andre)
        obs_values = values[:n_tid]

    s_maaned = pd.Series(
        {k: float(v) if v is not None else np.nan for k, v in zip(tid_cats, obs_values)},
        name="nibor_3m",
    )

    parsed_index = []
    for k in s_maaned.index:
        try:
            parsed_index.append(
                pd.Timestamp(k.replace("M", "-") + "-01") + pd.offsets.MonthEnd(0)
            )
        except Exception:
            parsed_index.append(None)
    s_maaned.index = parsed_index
    s_maaned = s_maaned[pd.notna(s_maaned.index)].sort_index()
    s_maaned.index = pd.DatetimeIndex(s_maaned.index)

    s_kvartal = s_maaned.resample("QE").mean()
    s_kvartal.name = "nibor_3m"
    logger.info("NIBOR 3M (SSB 10701) hentet: %d kvartaler", len(s_kvartal))
    return s_kvartal


def _ssb_oppdag_kode(
    dimension: dict,
    dim_id: str,
    soketermer: list[str],
) -> str | None:
    """
    Finner første dimensjonskode der label eller kode-ID inneholder ett av søkeordene.

    Returnerer kode-ID (str) eller None hvis ingen treff.
    """
    if dim_id not in dimension:
        return None
    cat = dimension[dim_id].get("category", {})
    koder = list(cat.get("index", {}).keys())
    labels = cat.get("label", {})
    for kode in koder:
        tekst = (labels.get(kode, "") + " " + kode).lower()
        if any(term.lower() in tekst for term in soketermer):
            return kode
    return None


def hent_k2_husholdning(bruk_cache: bool = True) -> pd.Series:
    """
    Henter K2 innenlandsk lånegjeld fra SSB tabell 11599 og beregner kvartalssnitt.

    Tabell 11599: innenlandsk lånegjeld, beholdninger og transaksjoner.
    Bruker Publikum som låntakersektor og ujusterte beholdninger (K2-definisjon).
    Månedlige data aggregeres til kvartalsvise gjennomsnitt.

    Parametere
    ----------
    bruk_cache : bool
        Om cache skal brukes.

    Returnerer
    ----------
    pd.Series
        Kvartalsvise K2-verdier (beholdning, NOK + utenl. valuta).
        Indeks: pd.Timestamp (siste dag i kvartal).
    """
    table_id = "11599"
    url = f"{_SSB_PXWEB_V2}/{table_id}/data"

    # Steg 1: hent minimal rad for å oppdage dimensjonskoder
    meta_params = {
        "lang": "no",
        "valueCodes[Tid]": "top(1)",
        "outputFormat": "json-stat2",
    }
    try:
        meta_resp = requests.get(url, params=meta_params, timeout=30)
        meta_resp.raise_for_status()
        meta = meta_resp.json()
    except Exception as exc:
        logger.warning("SSB 11599 metadata-kall feilet: %s. Prøver uten filter.", exc)
        meta = None

    # Bestem dimensjonsfiltre fra metadata
    params: dict[str, str] = {
        "lang": "no",
        "valueCodes[Tid]": "*",
        "outputFormat": "json-stat2",
    }

    if meta is not None:
        dim = meta.get("dimension", {})
        dims = meta.get("id", [])
        logger.info("SSB 11599 dimensjoner: %s", dims)
        for d in dims:
            koder = list(dim.get(d, {}).get("category", {}).get("index", {}).keys())
            labels = dim.get(d, {}).get("category", {}).get("label", {})
            logger.info("  %s: %s", d, {k: labels.get(k, k) for k in koder[:8]})

        # Valuta: "I alt" eller tilsvarende
        valuta_kode = _ssb_oppdag_kode(dim, "Valuta", ["i alt", "total", "alle"])
        if valuta_kode:
            params[f"valueCodes[Valuta]"] = valuta_kode

        # Låntakersektor: "Publikum"
        sektor_kode = _ssb_oppdag_kode(dim, "Låntakersektor", ["publikum", "public"])
        if not sektor_kode:
            sektor_kode = _ssb_oppdag_kode(dim, "Sektor", ["publikum", "public"])
        sektor_dim = "Låntakersektor" if "Låntakersektor" in (meta.get("id") or []) else "Sektor"
        if sektor_kode:
            params[f"valueCodes[{sektor_dim}]"] = sektor_kode

        # ContentsCode: ujusterte beholdninger
        cc_kode = _ssb_oppdag_kode(
            dim, "ContentsCode",
            ["ujustert", "beholdning", "stock", "outstanding"],
        )
        if cc_kode:
            params["valueCodes[ContentsCode]"] = cc_kode

    cache_fil = _cache_path(table_id)
    data = None

    if bruk_cache and cache_fil.exists():
        logger.info("Leser SSB-tabell %s fra cache: %s", table_id, cache_fil)
        with open(cache_fil, encoding="utf-8") as f:
            data = json.load(f)
    else:
        siste_exc: Exception | None = None
        for forsok in range(1, 4):
            try:
                resp = requests.get(url, params=params, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                with open(cache_fil, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)
                logger.info("Cachet SSB-tabell %s → %s", table_id, cache_fil)
                break
            except Exception as exc:
                siste_exc = exc
                body = ""
                try:
                    body = resp.text[:300]  # type: ignore[possibly-undefined]
                except Exception:
                    pass
                if forsok < 3:
                    vent = 2 ** forsok
                    logger.warning(
                        "SSB 11599 feilet (forsøk %d/3): %s | body: %s. Venter %ds.",
                        forsok, exc, body, vent,
                    )
                    time.sleep(vent)
        if data is None:
            fallback = _find_latest_cache(table_id)
            if fallback is not None:
                logger.info("SSB 11599: bruker cache-fallback %s", fallback)
                with open(fallback, encoding="utf-8") as f:
                    data = json.load(f)
            else:
                raise RuntimeError(
                    f"SSB PxWeb v2 feilet og ingen cache for {table_id}."
                ) from siste_exc

    # Parse: finn Tid-dimensjonen og flatt array
    dims_data = data.get("id", [])
    dimension = data.get("dimension", {})
    values = data.get("value", [])
    sizes = data.get("size", [])

    if "Tid" not in dims_data:
        raise ValueError(f"SSB 11599: ingen Tid-dimensjon i responsen. id={dims_data}")

    tid_idx = dims_data.index("Tid")
    tid_cats = list(dimension["Tid"]["category"]["index"].keys())
    n_tid = sizes[tid_idx]

    # Hvis vi har filtrert alle andre dims til 1 verdi, er flat array = tidsserie
    n_andre = 1
    for i, sz in enumerate(sizes):
        if i != tid_idx:
            n_andre *= sz

    if n_andre == 1:
        # Flat array er direkte tidsserien
        obs_values = values[:n_tid]
    else:
        # Ta første kombinasjon (øverste rad) langs alle andre dimensjoner
        logger.warning(
            "SSB 11599: %d kombinasjoner × %d Tid — tar første kombinasjon.",
            n_andre, n_tid,
        )
        obs_values = values[:n_tid]

    s_maaned = pd.Series(
        {k: float(v) if v is not None else np.nan for k, v in zip(tid_cats, obs_values)},
        name="k2_husholdning",
    )

    # Parse månedskoder ('YYYYMXX') til dato
    parsed_index = []
    for k in s_maaned.index:
        try:
            parsed_index.append(
                pd.Timestamp(k.replace("M", "-") + "-01") + pd.offsets.MonthEnd(0)
            )
        except Exception:
            parsed_index.append(None)
    s_maaned.index = parsed_index
    s_maaned = s_maaned[pd.notna(s_maaned.index)].sort_index()
    s_maaned.index = pd.DatetimeIndex(s_maaned.index)

    s_kvartal = s_maaned.resample("QE").mean()
    s_kvartal.name = "k2_husholdning"
    logger.info("K2 (SSB 11599) hentet: %d kvartaler", len(s_kvartal))
    return s_kvartal


