"""
================================================================================
NEMO FASE III — DATAPIPELINE FOR KRYSSJEKKMODELLER
Milepel 1 (datagrunnlag) og Milepel 2 (pipeline)

Henter alle 14 observerte variabler fra Fase II i råformat (nivå),
transformerer til stasjonære serier egnet for BVAR og tilleggsmodeller,
og lagrer til crosscheck_data.csv og crosscheck_meta.json.

Kan kjøres automatisk (cron/task scheduler) ved nytt kvartal.

Kjøring:
    python crosscheck_data.py [--output-dir DIR] [--start-year YYYY]

Avhengigheter:
    pip install requests pandas numpy statsmodels

Datakilder:
    SSB PxWebApi v2       — https://data.ssb.no/api/pxwebapi/v2/tables/{id}/data
    SSB Statistikkbanken  — https://data.ssb.no/api/v0/ (fallback)
    Norges Bank API       — https://data.norges-bank.no/api/
    IMF DataMapper        — https://www.imf.org/external/datamapper/api/v1/
    FRED (St. Louis Fed)  — https://api.stlouisfed.org/fred/ (krever FRED_API_KEY)
================================================================================
"""

import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime, date
from functools import lru_cache
from typing import Optional, List

import numpy as np
import pandas as pd
import requests
from io import StringIO
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("crosscheck_data")

# ── Konfigurasjon ─────────────────────────────────────────────────────────────
START_YEAR    = 2001
START_QUARTER = 1
TIMEOUT       = 45       # sekunder per HTTP-kall
COVID_EXCL    = ("2020Q1", "2021Q4")   # inkluderes i rådata, flagges i meta


def _make_session() -> requests.Session:
    """Opprett requests-session med retry-logikk og User-Agent (fikser FRED 403)."""
    session = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; NEMO-Forecast/3.0; research)",
        "Accept": "application/json, text/csv, */*",
    })
    return session


SESSION = _make_session()


# ── FRED-spesifikk session ────────────────────────────────────────────────────
# FRED svarer ofte tregt fra GitHub-runners. Bruk kort timeout og få retries
# slik at pipelinen raskt faller tilbake til CSV hvis serien ikke er tilgjengelig.
FRED_TIMEOUT = 20


def _make_fred_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=2,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
        "Accept": "text/csv, */*",
    })
    return session


FRED_SESSION = _make_fred_session()
FRED_API_KEY = os.getenv("FRED_API_KEY", "").strip()


# ═══════════════════════════════════════════════════════════════════════════════
# HJELPEFUNKSJONER
# ═══════════════════════════════════════════════════════════════════════════════

def quarter_index(start_year: int = START_YEAR,
                  start_q: int = START_QUARTER) -> pd.PeriodIndex:
    """Generer kvartalsvindeks fra start til i dag."""
    today = date.today()
    end_period = pd.Period(f"{today.year}Q{(today.month - 1) // 3 + 1}")
    start_period = pd.Period(f"{start_year}Q{start_q}")
    return pd.period_range(start_period, end_period, freq="Q")


def to_period_index(series: pd.Series) -> pd.Series:
    """Konverter ulike datoformater til PeriodIndex med frekvens Q."""
    if isinstance(series.index, pd.PeriodIndex):
        return series
    try:
        idx = pd.PeriodIndex(series.index, freq="Q")
    except Exception:
        idx = pd.DatetimeIndex(series.index)
        idx = idx.to_period("Q")
    series.index = idx
    return series


def hp_filter(y: np.ndarray, lam: float = 1600) -> tuple:
    """HP-filter, returnerer (trend, gap)."""
    T = len(y)
    D = np.zeros((T - 2, T))
    for i in range(T - 2):
        D[i, i] = 1; D[i, i + 1] = -2; D[i, i + 2] = 1
    A = np.eye(T) + lam * D.T @ D
    trend = np.linalg.solve(A, y)
    return trend, y - trend


def log_diff_annualised(s: pd.Series, factor: float = 4.0) -> pd.Series:
    """Annualisert kvartalsvekst i prosentpoeng: Δlog(s) × factor × 100."""
    return np.log(s).diff() * factor * 100


def log_diff_q(s: pd.Series) -> pd.Series:
    """Kvartalsvekst (ikke annualisert): Δlog(s) × 100."""
    return np.log(s).diff() * 100


def resample_monthly_to_quarterly(s: pd.Series,
                                   agg: str = "mean") -> pd.Series:
    """Aggreger månedlig serie til kvartal."""
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    return getattr(s.resample("QE"), agg)().to_period("Q")


# ═══════════════════════════════════════════════════════════════════════════════
# DATAKILDER — SSB
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_ssb_jsonstat(data: dict) -> pd.Series:
    """Felles JSON-stat parser for v0 og v2 responser."""
    ids   = data["id"]
    dims  = data["dimension"]
    vals  = data["value"]

    tid_key = next(
        (d for d in ids
         if "tid" in d.lower() or "kvartal" in d.lower() or "year" in d.lower()),
        ids[-1]
    )
    time_cats = list(dims[tid_key]["category"]["label"].values())

    def parse_period(s: str) -> Optional[pd.Period]:
        try:
            if "K" in s:
                y, q = s.split("K")
                return pd.Period(f"{y}Q{q}", freq="Q")
            if "M" in s:
                y, m = s.split("M")
                return pd.Period(f"{y}-{m}", freq="M")
            return pd.Period(s, freq="Q")
        except Exception:
            return None

    periods = [parse_period(t) for t in time_cats]
    valid = [(p, v) for p, v in zip(periods, vals)
             if p is not None and v is not None]
    if not valid:
        return pd.Series(dtype=float)
    idx, clean = zip(*valid)
    return pd.Series(list(clean), index=pd.PeriodIndex(idx),
                     dtype=float).sort_index()


def _period_series_to_quarterly(s: pd.Series) -> pd.Series:
    """Konverter en PeriodIndex-serie (M eller Q) til kvartallig DatetimeIndex-kompatibel."""
    if isinstance(s.index, pd.PeriodIndex):
        freq = s.index.freqstr.upper()
        if "M" in freq:
            # Månedlig → kvartal via DatetimeIndex
            s_dt = s.copy()
            s_dt.index = s.index.to_timestamp()
            quarterly = s_dt.resample("QE").mean()
            counts = s_dt.resample("QE").count()
            # Forkast partielle kvartaler (< 3 månedsverdier) for å unngå
            # at siste ufullstendige kvartal gir urimelige veksttall ved transformasjon.
            quarterly = quarterly[counts >= 3]
            return quarterly.to_period("Q")
        # Allerede kvartallig
        s.index = s.index.asfreq("Q")
        return s
    return s


@lru_cache(maxsize=16)
def _ssb_meta(table_id: str) -> list:
    """Hent SSB v0 tabellmetadata (bufret). Returnerer liste av {code, text, values, valueTexts}."""
    url = f"https://data.ssb.no/api/v0/no/table/{table_id}"
    try:
        r = SESSION.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        d = r.json()
        result = []
        for var in d.get("variables", []):
            result.append({
                "code":       var.get("code", ""),
                "text":       var.get("text", ""),
                "values":     var.get("values", []),
                "valueTexts": var.get("valueTexts", var.get("values", [])),
            })
        log.debug(f"SSB {table_id} meta: {[v['code'] for v in result]}")
        return result
    except Exception as e:
        log.warning(f"SSB meta {table_id}: {e}")
        return []


def _ssb_find_value(options: List[str], labels: List[str],
                    keywords: List[str]) -> Optional[str]:
    """Finn første verdikode som matcher et nøkkelord (case-insensitiv) i kode eller label."""
    for kw in keywords:
        kl = kw.lower()
        for code, label in zip(options, labels):
            if kl == code.lower() or kl in code.lower() or kl in label.lower():
                return code
    return None


def _ssb_smart_query(table_id: str,
                     contents_keywords: List[str],
                     sector_keywords: Optional[List[str]] = None) -> pd.Series:
    """
    Bygg SSB v0-spørring basert på tabellmetadata.
    Velger ContentsCode via innholds-nøkkelord og evt. sektordimensjon via sektor-nøkkelord.
    For ukjente dimensjoner brukes første gyldige verdi.
    """
    meta = _ssb_meta(table_id)
    if not meta:
        return pd.Series(dtype=float)

    query_parts = []
    for var in meta:
        vc   = var["code"]
        vt   = var["text"].lower()
        vals = var["values"]
        lbls = var["valueTexts"]
        if not vals:
            continue

        if "tid" in vc.lower() or "tid" in vt or "quarter" in vt:
            query_parts.append({
                "code": vc,
                "selection": {"filter": "all", "values": ["*"]},
            })
        elif "contents" in vc.lower() or vc.lower() == "contentsCode".lower():
            code = _ssb_find_value(vals, lbls, contents_keywords)
            if code is None:
                log.warning(f"SSB {table_id}: ingen ContentsCode for {contents_keywords}; "
                            f"tilgjengelig: {list(zip(vals[:6], lbls[:6]))}")
                return pd.Series(dtype=float)
            query_parts.append({
                "code": vc,
                "selection": {"filter": "item", "values": [code]},
            })
        elif sector_keywords:
            # Prøv sektor-nøkkelord først (f.eks. 'husholdninger'), deretter
            # generiske aggregater for dimensjoner som ikke er sektor
            # (f.eks. Valuta = 'I alt', Region = 'Hele landet').
            code = _ssb_find_value(vals, lbls, sector_keywords)
            if not code:
                code = _ssb_find_value(
                    vals, lbls,
                    ["i alt", "alle", "total", "totalt", "hele", "00", "0000"],
                )
            if code:
                label = dict(zip(vals, lbls)).get(code, "")
                log.info(f"  SSB {table_id}/{vc}: valgte {code!r} ({label!r})")
                query_parts.append({
                    "code": vc,
                    "selection": {"filter": "item", "values": [code]},
                })
            else:
                log.warning(
                    f"SSB {table_id}/{vc}: ingen match for {sector_keywords} "
                    f"eller aggregat; tilgjengelig: "
                    f"{list(zip(vals[:8], lbls[:8]))}"
                )
                return pd.Series(dtype=float)
        else:
            query_parts.append({
                "code": vc,
                "selection": {"filter": "item", "values": [vals[0]]},
            })

    if not query_parts:
        return pd.Series(dtype=float)

    query = {"query": query_parts, "response": {"format": "json-stat2"}}
    url = f"https://data.ssb.no/api/v0/no/table/{table_id}"
    try:
        resp = SESSION.post(url, json=query, timeout=TIMEOUT)
        resp.raise_for_status()
        return _parse_ssb_jsonstat(resp.json())
    except Exception as e:
        log.warning(f"SSB {table_id}: {e}")
        return pd.Series(dtype=float)


# ContentsCode for 09190: bruk faste priser sesongjustert (nivå i mill. kr, 2023-priser)
# slik at vår egen log_diff_q gir korrekt kvartalsvekst.
_NR_CONTENTS = ["FastePriserSesJust", "Faste priser, sesongjustert", "Faste"]


def fetch_bnp_fastland() -> pd.Series:
    """BNP Fastlands-Norge, faste priser sesongjustert — SSB 09190."""
    s = _ssb_smart_query(
        "09190",
        contents_keywords=_NR_CONTENTS,
        sector_keywords=[
            "nr23_9fn", "9fn",
            "bruttonasjonalprodukt fastlands-norge",
            "bnp fastlands-norge",
            "fastlands-norge",
        ],
    )
    s = _period_series_to_quarterly(s)
    log.info(f"  BNP fastland:    {len(s)} kvartaler (SSB 09190)")
    return s


def fetch_privat_konsum() -> pd.Series:
    """Privat konsum i husholdninger, faste priser sesongjustert — SSB 09190."""
    s = _ssb_smart_query(
        "09190",
        contents_keywords=_NR_CONTENTS,
        sector_keywords=[
            "pk.",                        # «pk.»-prefiks i alle privat-konsum-koder
            "konsum i husholdninger",
            "privat konsum",
            "husholdningers konsum",
        ],
    )
    s = _period_series_to_quarterly(s)
    log.info(f"  Privat konsum:   {len(s)} kvartaler (SSB 09190)")
    return s


def fetch_investering() -> pd.Series:
    """Bruttoinvestering i fast realkapital, faste priser sesongjustert — SSB 09190."""
    s = _ssb_smart_query(
        "09190",
        contents_keywords=_NR_CONTENTS,
        sector_keywords=[
            "nr23_7", "bruttoinvestering i fast realkapital",
            "bruttoinvestering", "fast realkapital", "investering",
        ],
    )
    s = _period_series_to_quarterly(s)
    log.info(f"  Investering:     {len(s)} kvartaler (SSB 09190)")
    return s


def fetch_eksport() -> pd.Series:
    """Eksport i alt, faste priser sesongjustert — SSB 09190."""
    s = _ssb_smart_query(
        "09190",
        contents_keywords=_NR_CONTENTS,
        sector_keywords=[
            "nr23_11", "eksport i alt", "eksport",
        ],
    )
    s = _period_series_to_quarterly(s)
    log.info(f"  Eksport:         {len(s)} kvartaler (SSB 09190)")
    return s


def fetch_import() -> pd.Series:
    """Import i alt, faste priser sesongjustert — SSB 09190."""
    s = _ssb_smart_query(
        "09190",
        contents_keywords=_NR_CONTENTS,
        sector_keywords=[
            "nr23_14", "import i alt", "import",
        ],
    )
    s = _period_series_to_quarterly(s)
    log.info(f"  Import:          {len(s)} kvartaler (SSB 09190)")
    return s


def fetch_kpi() -> pd.Series:
    """KPI alle varer, månedlig → kvartalsgjennomsnitt — SSB 03013."""
    s = _ssb_smart_query(
        "03013",
        contents_keywords=["KpiIndMnd", "indeks", "kpi", "totalindeks"],
        sector_keywords=["TOTAL", "total", "alle"],
    )
    s = _period_series_to_quarterly(s)
    log.info(f"  KPI:             {len(s)} kvartaler (SSB 03013)")
    return s


def fetch_lonnsindeks() -> pd.Series:
    """Lønn per normalårsverk — SSB 09786."""
    s = _ssb_smart_query(
        "09786",
        contents_keywords=["Fortjeneste", "lønn", "lonn", "wage", "earnings"],
    )
    s = _period_series_to_quarterly(s)
    log.info(f"  Lønnsindeks:     {len(s)} kvartaler (SSB 09786)")
    return s


def fetch_boligpris() -> pd.Series:
    """Boligprisindeks, kvartalsvis — prøver 11136 → 07221 → 07241(KvPris)."""
    candidates = [
        ("11136", ["BoligprIS", "prisindeks", "indeks"],
                  ["00", "i alt", "alle", "totalt"]),
        ("07221", ["BoligprIS", "prisindeks", "indeks"],
                  ["0000", "hele landet", "hele landet i alt", "landet", "i alt", "totalt"]),
        ("07241", ["KvPris", "kvadratmeter", "pris"],
                  ["00", "i alt", "alle", "boliger"]),
    ]
    for table_id, contents_kw, sector_kw in candidates:
        s = _ssb_smart_query(table_id, contents_kw, sector_kw)
        if not s.empty:
            s = _period_series_to_quarterly(s)
            log.info(f"  Boligpris:       {len(s)} kvartaler (SSB {table_id})")
            return s
    log.warning("  Boligpris:       0 kvartaler (alle tabeller feilet)")
    return pd.Series(dtype=float)


# ═══════════════════════════════════════════════════════════════════════════════
# DATAKILDER — NORGES BANK
# ═══════════════════════════════════════════════════════════════════════════════

def nb_api(series_key: str, label: str) -> pd.Series:
    """Generell henter fra Norges Bank SDMX-JSON API."""
    url = f"https://data.norges-bank.no/api/data/{series_key}"
    params = {
        "startPeriod": f"{START_YEAR}-01-01",
        "format": "sdmx-json",
        "locale": "no",
    }
    try:
        resp = SESSION.get(url, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.warning(f"Norges Bank {series_key}: {e}")
        return pd.Series(dtype=float)

    try:
        obs  = data["data"]["dataSets"][0]["series"]
        time = data["data"]["structure"]["dimensions"]["observation"][0]
        times = [p["id"] for p in time["values"]]
        # Bruk første serie
        ser_data = list(obs.values())[0]["observations"]
        result = {times[int(k)]: v[0] for k, v in ser_data.items()
                  if v[0] is not None}
        s = pd.Series(result, dtype=float).sort_index()
        # Dagsdata → kvartalsgjennomsnitt; forkast partielle kvartaler
        # (mindre enn ~60 handelsdager) for å unngå at et pågående kvartal
        # gir utslag i log-differanser nedstrøms.
        if len(s) > 200:
            s.index = pd.to_datetime(s.index)
            grp = s.resample("QE")
            quarterly = grp.mean()
            counts = grp.count()
            quarterly = quarterly[counts >= 60]
            s = quarterly.to_period("Q")
        else:
            s = to_period_index(s)
        log.info(f"  {label:<20} {len(s)} kvartaler (Norges Bank)")
        return s
    except Exception as e:
        log.warning(f"  Parsing {label}: {e}")
        return pd.Series(dtype=float)


def fetch_styringsrente() -> pd.Series:
    # Norges Bank SDMX: Interest Rates dataflow, styringsrente (Key Policy Rate)
    return nb_api("IR/B.KPRA.SD.R", "Styringsrente:")


@lru_cache(maxsize=1)
def _nb_dataflows() -> list:
    """Hent liste over tilgjengelige Norges Bank-dataflows for diagnostikk."""
    try:
        r = SESSION.get("https://data.norges-bank.no/api/dataflow",
                        params={"format": "sdmx-json"}, timeout=TIMEOUT)
        r.raise_for_status()
        d = r.json()
        flows = d.get("data", {}).get("dataflows", []) or d.get("dataflows", [])
        ids = []
        for f in flows:
            fid = f.get("id", "")
            name = f.get("name", f.get("names", {}).get("no", ""))
            ids.append((fid, name))
        return ids
    except Exception as e:
        log.warning(f"NB dataflow probe: {e}")
        return []


def _nb_try_paths(candidates: List[str], label: str) -> pd.Series:
    """Prøv en rekke SDMX-stier til første 200 OK."""
    for path in candidates:
        s = nb_api(path, label)
        if not s.empty:
            return s
    # Logg tilgjengelige dataflows ved totalt feil — hjelp fremtidig diagnostikk
    flows = _nb_dataflows()
    if flows:
        log.info(f"  NB tilgjengelige dataflows ({len(flows)}): "
                 f"{', '.join(fid for fid, _ in flows[:20])}")
    return pd.Series(dtype=float)


def _nb_discover_key(dataflow: str, match_terms: List[str]) -> Optional[str]:
    """
    Query et NB-dataflow for å finne en series-key som matcher match_terms
    (søk i dimensjonsverdier og labels). Returnerer full SDMX-sti.
    """
    try:
        url = f"https://data.norges-bank.no/api/data/{dataflow}"
        r = SESSION.get(url,
                        params={"format": "sdmx-json", "lastNObservations": 1},
                        timeout=TIMEOUT)
        r.raise_for_status()
        d = r.json()
        dims = d["data"]["structure"]["dimensions"]["series"]
        series = d["data"]["dataSets"][0]["series"]
        best = None
        for key_indices in series.keys():
            parts = [int(x) for x in key_indices.split(":")]
            vid_parts, labels = [], []
            for dim, i in zip(dims, parts):
                val = dim["values"][i]
                vid_parts.append(val["id"])
                labels.append(val.get("name", ""))
            combined = (" ".join(vid_parts) + " " + " ".join(labels)).upper()
            if all(t.upper() in combined for t in match_terms):
                key_str = f"{dataflow}/{'.'.join(vid_parts)}"
                log.info(f"  NB {dataflow} oppdaget {match_terms}: {key_str}")
                return key_str
        log.info(f"  NB {dataflow}: fant ingen match for {match_terms} "
                 f"blant {len(series)} serier")
    except Exception as e:
        log.warning(f"NB discover {dataflow} {match_terms}: {e}")
    return None


def fetch_nibor_3m() -> pd.Series:
    """
    Kortsiktig pengemarkedsrente — NOWA (Norwegian Overnight Weighted Average).
    NOWA ble flyttet fra IR- til SHORT_RATES-dataflowen (NB dokumentasjon).
    """
    # Kjent offisiell nøkkel (verifisert i NBs dokumentasjon).
    candidates = ["SHORT_RATES/B.NOWA.ON.R"]
    # Dynamisk discovery som fallback — søk i flere dataflows hvis direkte
    # nøkkel skulle endre seg igjen.
    for df_name in ["SHORT_RATES", "IR", "MONEY_MARKET", "FINANCIAL_INDICATORS"]:
        key = _nb_discover_key(df_name, ["NOWA"])
        if key and key not in candidates:
            candidates.append(key)
    return _nb_try_paths(candidates, "NOWA (proxy 3M):")


def fetch_nok_eur() -> pd.Series:
    return nb_api("EXR/B.EUR.NOK.SP", "NOK/EUR:")


def fetch_kredittvekst() -> pd.Series:
    """
    K2 kredittindikator, husholdninger — publiseres av SSB.
    Primærtabell 11599: "Innenlandsk lånegjeld … etter måned, valuta og
    låntakersektor". ContentsCode = 'Tolvmånedersvekst. Prosent',
    sektor-dimensjon 'låntakersektor', verdi 'Husholdninger mv.'.
    """
    candidates = [
        ("11599",
         ["Tolvmånedersvekst", "tolvmnedersvekst", "tolvmndsvekst",
          "12-månedersvekst", "12 mnd", "årsvekst", "vekst"],
         ["husholdninger mv", "husholdninger", "hushold", "hhold"]),
    ]
    for table_id, contents_kw, sector_kw in candidates:
        s = _ssb_smart_query(table_id, contents_kw, sector_kw)
        if not s.empty:
            s = _period_series_to_quarterly(s)
            log.info(f"  K2 husholdninger: {len(s)} kvartaler (SSB {table_id})")
            return s
    log.warning("  K2 husholdninger: 0 kvartaler (alle tabeller feilet)")
    return pd.Series(dtype=float)


# ═══════════════════════════════════════════════════════════════════════════════
# DATAKILDER — FRED (via CSV-nedlasting, ingen API-nøkkel)
# ═══════════════════════════════════════════════════════════════════════════════

def _fred_fetch_json(series_id: str) -> pd.Series:
    """Hent serie via autentisert FRED JSON-API (krever FRED_API_KEY)."""
    url = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json"
    )
    resp = FRED_SESSION.get(url, timeout=FRED_TIMEOUT)
    resp.raise_for_status()
    obs = resp.json().get("observations", [])
    data = {o["date"]: (float(o["value"]) if o["value"] not in (".", "", None) else np.nan)
            for o in obs}
    s = pd.Series(data, dtype=float).dropna()
    s.index = pd.to_datetime(s.index)
    return s


def _fred_fetch_csv(series_id: str) -> pd.Series:
    """Fallback: hent serie fra FRED via CSV-endepunkt (uten nøkkel)."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    resp = FRED_SESSION.get(url, timeout=FRED_TIMEOUT)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text), parse_dates=["DATE"], index_col="DATE")
    s = df.iloc[:, 0].dropna()
    s.index = pd.DatetimeIndex(s.index)
    return s


def fred_csv(series_id: str, label: str) -> pd.Series:
    """Hent serie fra FRED (JSON-API hvis nøkkel, ellers CSV-scraping)."""
    try:
        if FRED_API_KEY:
            s = _fred_fetch_json(series_id)
            source = "FRED-API"
        else:
            s = _fred_fetch_csv(series_id)
            source = "FRED-CSV"
        if len(s) > 300:
            # Daglig serie: forkast partielle kvartaler (<60 handelsdager)
            grp = s.resample("QE")
            quarterly = grp.mean()
            counts = grp.count()
            quarterly = quarterly[counts >= 60]
            s = quarterly.to_period("Q")
        else:
            # Kvartallig/månedlig serie: bruk siste verdi
            s = s.resample("QE").last().to_period("Q")
        log.info(f"  {label:<20} {len(s)} kvartaler ({source})")
        return s
    except Exception as e:
        log.warning(f"  FRED {series_id}: {e}")
        return pd.Series(dtype=float)


def fetch_oljepris() -> pd.Series:
    """Oljepris Brent USD/fat — FRED DCOILBRENTEU."""
    return fred_csv("DCOILBRENTEU", "Oljepris Brent:")


def fetch_handelspartner_bnp() -> pd.Series:
    """
    Handelsandels-vektet BNP-gap for Norges viktigste handelspartnere.
    Vekter (tilnærmet fra Norges Banks varehandelsstatistikk):
        Tyskland 20%, Sverige 16%, UK 15%, USA 9%, Frankrike 8%,
        resten fordelt på Kina, Nederland, Danmark mv.
    HP-filtrering med lambda=1600.
    """
    partner_series = {
        "CLVMNACSCAB1GQDE": 0.20,   # Tyskland (Eurostat)
        "CLVMNACSCAB1GQSE": 0.16,   # Sverige (Eurostat)
        "CLVMNACSCAB1GQUK": 0.15,   # UK real GDP (Eurostat)
        "GDPC1":            0.09,   # USA real GDP (2017-dollar)
        "CLVMNACSCAB1GQFR": 0.08,   # Frankrike (Eurostat)
        "CLVMNACSCAB1GQNL": 0.06,   # Nederland (Eurostat)
        "CLVMNACSCAB1GQDK": 0.06,   # Danmark (Eurostat, korrekt 2-bokstavs)
    }
    remaining_weight = 1.0 - sum(partner_series.values())

    combined = None
    tot_w = 0.0
    for fid, w in partner_series.items():
        s = fred_csv(fid, f"Handelspartner ({fid[:6]}):")
        if s.empty:
            continue
        s = s.dropna()
        if len(s) < 20:
            continue
        log_s = np.log(s.values.astype(float))
        _, gap = hp_filter(log_s)
        gap_s = pd.Series(gap * 100, index=s.index)
        if combined is None:
            combined = gap_s * w
        else:
            combined = combined.add(gap_s * w, fill_value=0)
        tot_w += w

    if combined is None or tot_w == 0:
        return pd.Series(dtype=float)

    combined = combined / tot_w  # Renormaliser til tilgjengelige serier
    log.info(f"  {'Handelspartner-BNP:':<20} {len(combined)} kvartaler (FRED, {len(partner_series)} land)")
    return combined


# ═══════════════════════════════════════════════════════════════════════════════
# FALLBACK — bruk eksisterende datafil ved API-feil
# ═══════════════════════════════════════════════════════════════════════════════

def merge_with_fallback(df: pd.DataFrame, fallback_path: str) -> pd.DataFrame:
    """
    Fyll manglende obs-kolonner fra fallback-CSV (f.eks. nemo_data_faktisk_v2.csv)
    ved API-feil eller tomme serier. Sikrer at modellene alltid har data å kjøre på.
    """
    if not fallback_path or not os.path.exists(fallback_path):
        log.info("  Ingen fallback-fil angitt eller funnet.")
        return df
    try:
        fb = pd.read_csv(fallback_path, index_col=0)
        try:
            fb.index = pd.PeriodIndex(fb.index, freq="Q")
        except Exception:
            fb.index = pd.PeriodIndex(pd.to_datetime(fb.index), freq="Q")
    except Exception as e:
        log.warning(f"  Kunne ikke lese fallback '{fallback_path}': {e}")
        return df

    obs_cols = [c for c in fb.columns
                if not c.endswith("_level") and c != "covid_flag"]
    filled = []
    for col in obs_cols:
        if col not in df.columns or len(df[col].dropna()) < 10:
            df[col] = fb[col].reindex(df.index)
            filled.append(col)
    if filled:
        log.warning(f"  FALLBACK brukt for {len(filled)} serier: {', '.join(filled)}")
    else:
        log.info("  Alle serier hentet live — ingen fallback nødvendig.")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFORMASJONER — FASE III FORMAT
# ═══════════════════════════════════════════════════════════════════════════════

def transform_all(raw: dict) -> pd.DataFrame:
    """
    Transformer råserier til stasjonære serier egnet for BVAR.

    Konvensjoner (konsistente med nemo_data_faktisk_v2.csv):
        dy_obs    : Δlog(BNP) × 100
        dc_obs    : Δlog(konsum) × 100
        dinv_obs  : Δlog(investering) × 100
        dx_obs    : Δlog(eksport) × 100
        dm_obs    : Δlog(import) × 100
        pi_obs    : Δlog(KPI) × 4 × 100  (annualisert)
        dw_obs    : Δlog(lønn) × 4 × 100
        i_R_obs   : styringsrente / 4 (kvartalsnivå, i pst.)
        i_3m_obs  : NIBOR 3M / 4
        ds_obs    : Δlog(NOK/EUR) × 100
        dpO_obs   : Δlog(oljepris) × 100
        dyS_obs   : handelspartner-BNP gap
        dh_obs    : Δlog(boligpris) × 100
        db_obs    : kredittvekst (K2 HH, allerede vekst)

    I tillegg lagres nivåseriene med suffiks _level for VECM/kointegrasjonsmodeller.
    """
    out = {}
    errors = []

    def try_transform(name: str, fn, fallback=None):
        try:
            result = fn()
            if result is not None and not (hasattr(result, 'empty') and result.empty):
                out[name] = result
        except Exception as e:
            errors.append(f"{name}: {e}")
            if fallback is not None:
                out[name] = fallback

    # ── Volymserier (log-differanser) ────────────────────────────────────────
    for obs_name, raw_key in [
        ("dy_obs",   "bnp"),
        ("dc_obs",   "konsum"),
        ("dinv_obs", "investering"),
        ("dx_obs",   "eksport"),
        ("dm_obs",   "import"),
    ]:
        try_transform(
            obs_name,
            lambda k=raw_key: log_diff_q(raw[k]) if k in raw else None
        )
        try_transform(
            obs_name + "_level",
            lambda k=raw_key: np.log(raw[k]) if k in raw else None
        )

    # ── KPI og lønn (annualisert inflasjon/vekst) ────────────────────────────
    for obs_name, raw_key, factor in [
        ("pi_obs", "kpi",  4.0),
        ("dw_obs", "lonn", 4.0),
        ("dh_obs", "boligpris", 1.0),  # kvartalsvekst holder for bolig
    ]:
        try_transform(
            obs_name,
            lambda k=raw_key, f=factor:
                log_diff_annualised(raw[k], factor=f) if k in raw else None
        )
        try_transform(
            obs_name + "_level",
            lambda k=raw_key: np.log(raw[k]) if k in raw else None
        )

    # ── Renter (konverter % per år → % per kvartal) ──────────────────────────
    for obs_name, raw_key in [("i_R_obs", "styringsrente"), ("i_3m_obs", "nibor")]:
        try_transform(
            obs_name,
            lambda k=raw_key: raw[k] / 4.0 if k in raw else None
        )
        try_transform(
            obs_name + "_level",
            lambda k=raw_key: raw[k] if k in raw else None
        )

    # ── Valutakurs og oljepris (log-differanser) ─────────────────────────────
    for obs_name, raw_key in [("ds_obs", "nok_eur"), ("dpO_obs", "oljepris")]:
        try_transform(
            obs_name,
            lambda k=raw_key: log_diff_q(raw[k]) if k in raw else None
        )
        try_transform(
            obs_name + "_level",
            lambda k=raw_key: np.log(raw[k]) if k in raw else None
        )

    # ── Handelspartner-BNP (allerede gap) ───────────────────────────────────
    try_transform("dyS_obs", lambda: raw.get("handelspartner"))
    try_transform("dyS_obs_level", lambda: raw.get("handelspartner"))

    # ── Kredittvekst (allerede vekstrate) ───────────────────────────────────
    try_transform("db_obs", lambda: raw.get("kreditt"))
    try_transform(
        "db_obs_level",
        lambda: np.log(1 + raw["kreditt"] / 100) if "kreditt" in raw else None
    )

    if errors:
        log.warning(f"Transformasjonsfeil: {errors}")

    # Bygg felles DataFrame
    df = pd.DataFrame(out)
    return df


def align_to_common_index(df: pd.DataFrame,
                           start_year: int = START_YEAR) -> pd.DataFrame:
    """
    Juster alle serier til felles kvartalsvindeks.
    Fyller manglende verdier med NaN (ikke interpolering — modellene håndterer dette).
    """
    full_idx = quarter_index(start_year)
    df = df.reindex(full_idx)
    return df


def flag_covid(df: pd.DataFrame) -> pd.Series:
    """Returner boolean maske for COVID-periode."""
    start = pd.Period(COVID_EXCL[0], freq="Q")
    end   = pd.Period(COVID_EXCL[1], freq="Q")
    return pd.Series(
        [(p >= start and p <= end) for p in df.index],
        index=df.index,
        name="covid_flag",
        dtype=bool,
    )


def demean(df: pd.DataFrame,
           exclude_covid: bool = True) -> tuple:
    """
    Fjern gjennomsnitt fra alle serier (kun differanser-/vekstsserier, ikke nivå).
    Returnerer (demeaned_df, means_dict).
    """
    non_level = [c for c in df.columns if not c.endswith("_level")]
    covid_mask = flag_covid(df)
    means = {}
    df_out = df.copy()
    for col in non_level:
        s = df[col].copy()
        if exclude_covid:
            s_fit = s[~covid_mask]
        else:
            s_fit = s
        mu = s_fit.dropna().mean()
        df_out[col] = s - mu
        means[col] = float(mu)
    return df_out, means


# ═══════════════════════════════════════════════════════════════════════════════
# HOVED-PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(output_dir: str = ".", fallback_path: str = None) -> tuple:
    """
    Kjør komplett datapipeline.

    Returns:
        (df_raw, df_transformed, df_demeaned, meta)
    """
    os.makedirs(output_dir, exist_ok=True)
    log.info("=" * 65)
    log.info("  NEMO FASE III — DATAPIPELINE")
    log.info(f"  Kjøretidspunkt: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 65)

    # ── Steg 1: Hent rådata ─────────────────────────────────────────────────
    log.info("\n[1/4] Henter rådata fra kildeAPI-er...")
    raw = {}
    fetchers = {
        "bnp":            fetch_bnp_fastland,
        "konsum":         fetch_privat_konsum,
        "investering":    fetch_investering,
        "eksport":        fetch_eksport,
        "import":         fetch_import,
        "kpi":            fetch_kpi,
        "lonn":           fetch_lonnsindeks,
        "styringsrente":  fetch_styringsrente,
        "nibor":          fetch_nibor_3m,
        "nok_eur":        fetch_nok_eur,
        "kreditt":        fetch_kredittvekst,
        "oljepris":       fetch_oljepris,
        "boligpris":      fetch_boligpris,
        "handelspartner": fetch_handelspartner_bnp,
    }

    fetch_status = {}
    for key, fn in fetchers.items():
        try:
            s = fn()
            if s is not None and not s.empty:
                raw[key] = to_period_index(s)
                fetch_status[key] = {"ok": True, "n": len(s)}
            else:
                fetch_status[key] = {"ok": False, "n": 0, "reason": "tom serie"}
        except Exception as e:
            log.error(f"  Feil ved innhenting av {key}: {e}")
            fetch_status[key] = {"ok": False, "n": 0, "reason": str(e)}

    ok_count = sum(1 for v in fetch_status.values() if v["ok"])
    log.info(f"  Hentet {ok_count}/{len(fetchers)} serier OK.")
    if ok_count < 10:
        log.warning("  Advarsel: Færre enn 10 serier hentet. Sjekk nettverkstilgang og API-endepunkter.")

    # ── Steg 2: Transformer ─────────────────────────────────────────────────
    log.info("\n[2/4] Transformerer til stasjonære serier...")
    df_transformed = transform_all(raw)

    # ── Steg 3: Juster til felles indeks og demean ──────────────────────────
    log.info("\n[3/4] Justerer til felles kvartalsvindeks og demeanor...")
    df_aligned = align_to_common_index(df_transformed)
    df_demeaned, means = demean(df_aligned)

    covid_col = flag_covid(df_demeaned)
    df_demeaned["covid_flag"] = covid_col.astype(int)
    df_aligned["covid_flag"]  = covid_col.astype(int)

    # ── Steg 3b: Fyll inn fra fallback ved API-feil ─────────────────────────
    if fallback_path:
        log.info(f"\n[3b/4] Sjekker fallback mot: {fallback_path}")
        df_demeaned = merge_with_fallback(df_demeaned, fallback_path)

    # ── Steg 4: Lagre ───────────────────────────────────────────────────────
    log.info("\n[4/4] Lagrer output...")

    # Konverter PeriodIndex til string for CSV-kompatibilitet
    def period_to_str(df):
        df = df.copy()
        df.index = df.index.astype(str)
        return df

    csv_path = os.path.join(output_dir, "crosscheck_data.csv")
    period_to_str(df_demeaned).to_csv(csv_path)
    log.info(f"  Lagret: {csv_path}  ({df_demeaned.shape[0]} rader × {df_demeaned.shape[1]} kolonner)")

    raw_csv = os.path.join(output_dir, "crosscheck_data_raw.csv")
    # Sett alle råserier i én DataFrame
    raw_frames = {k: v for k, v in raw.items() if not v.empty}
    if raw_frames:
        raw_df = pd.DataFrame(raw_frames)
        raw_df.index = raw_df.index.astype(str)
        raw_df.to_csv(raw_csv)
        log.info(f"  Lagret: {raw_csv}")

    # Metadata
    now_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    meta = {
        "pipeline_version": "fase3_v1",
        "run_timestamp":    now_str,
        "start_period":     str(df_demeaned.index[0]) if len(df_demeaned) else "–",
        "end_period":       str(df_demeaned.index[-1]) if len(df_demeaned) else "–",
        "n_quarters":       int(len(df_demeaned)),
        "n_variables":      int(df_demeaned.shape[1]),
        "columns":          list(df_demeaned.columns),
        "obs_series": [c for c in df_demeaned.columns
                       if not c.endswith("_level") and c != "covid_flag"],
        "level_series": [c for c in df_demeaned.columns
                         if c.endswith("_level")],
        "demean_values":    means,
        "covid_exclusion":  list(COVID_EXCL),
        "fetch_status":     fetch_status,
        "sources": {
            "bnp_konsum_inv_eks_imp": "SSB Statistikkbanken, tabell 09190",
            "kpi":    "SSB Statistikkbanken, tabell 03013 (månedlig → kvartal)",
            "lonn":   "SSB Statistikkbanken, tabell 09786",
            "bolig":  "SSB Statistikkbanken, tabell 07241",
            "renter": "Norges Bank API (SDMX-JSON)",
            "nok_eur":"Norges Bank API (SDMX-JSON)",
            "kreditt":"Norges Bank API, K2 husholdninger",
            "olje":   "FRED, DCOILBRENTEU (Brent spot USD/fat)",
            "partner":"FRED, volumindekser 7 handelspartnere, HP-gap λ=1600",
        },
        "transformations": {
            "dy,dc,dinv,dx,dm": "Δlog(X) × 100 (kvartalsvekst pst.)",
            "pi_obs":           "Δlog(KPI) × 4 × 100 (annualisert)",
            "dw_obs":           "Δlog(lønn) × 4 × 100 (annualisert)",
            "dh_obs":           "Δlog(boligpris) × 100 (kvartalsvekst)",
            "i_R_obs, i_3m_obs":"rente / 4 (kvartalsnivå, pst.)",
            "ds_obs, dpO_obs":  "Δlog(X) × 100",
            "dyS_obs":          "HP-gap, vektet snitt handelspartnere",
            "db_obs":           "K2 kredittvekst HH (råvekstrate)",
            "*_level":          "log-nivå av tilsvarende volumserie",
            "covid_flag":       "1 = 2020K1–2021K4, ellers 0",
        },
    }
    meta_path = os.path.join(output_dir, "crosscheck_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    log.info(f"  Lagret: {meta_path}")

    log.info("\n  Pipeline ferdig.")
    log.info(f"  Periode: {meta['start_period']} – {meta['end_period']}")
    log.info(f"  Variabler OK: {ok_count}/{len(fetchers)}")
    return df_aligned, df_demeaned, meta


# ═══════════════════════════════════════════════════════════════════════════════
# KJØRING
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NEMO Fase III — Datapipeline for kryssjekkmodeller"
    )
    parser.add_argument(
        "--output-dir", default=".",
        help="Mappe for output-filer (default: arbeidsmappe)"
    )
    parser.add_argument(
        "--start-year", type=int, default=START_YEAR,
        help=f"Start-år for dataserie (default: {START_YEAR})"
    )
    parser.add_argument(
        "--fallback", default=None,
        help="Sti til fallback-CSV (f.eks. nemo_data_faktisk_v2.csv) ved API-feil"
    )
    args = parser.parse_args()
    START_YEAR = args.start_year

    _, df_out, meta = run_pipeline(
        output_dir=args.output_dir,
        fallback_path=args.fallback,
    )

    # Kort oppsummering i terminal
    print("\n" + "=" * 55)
    print("  OPPSUMMERING")
    print("=" * 55)
    print(f"  Periode:      {meta['start_period']} – {meta['end_period']}")
    print(f"  Kvartaler:    {meta['n_quarters']}")
    obs_ok = sum(1 for v in meta["fetch_status"].values() if v["ok"])
    print(f"  Serier hentet: {obs_ok}/{len(meta['fetch_status'])}")
    missing = [k for k, v in meta["fetch_status"].items() if not v["ok"]]
    if missing:
        print(f"  Mangler:      {', '.join(missing)}")
    print(f"  Output:       crosscheck_data.csv, crosscheck_meta.json")
    print("=" * 55)
