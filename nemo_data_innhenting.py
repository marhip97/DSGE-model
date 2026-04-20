"""
================================================================================
NEMO FASE II — DATAINNHENTING OG TRANSFORMASJON
Kravik og Mimir (2019), Appendiks A

Kjør dette skriptet lokalt med internettilgang.
Installasjonskrav:
    pip install requests pandas numpy statsmodels

Datakilder:
    - SSB Statistikkbanken (JSON-stat API, ingen autentisering)
    - Norges Bank Data API (JSON, ingen autentisering)
    - FRED / IMF (alternativ for handelpartnerdata)

Estimeringsperiode: 2001K1 – siste tilgjengelige kvartal
(inflasjonsmålperioden, konsistent med Kravik og Mimir 2019)

Output: nemo_data.csv  og  nemo_data.json
================================================================================
"""

import requests
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# ─── Konfigurasjon ─────────────────────────────────────────────────────────────
START_YEAR   = 2001
START_QUARTER = 1
OUTPUT_DIR   = "."   # Endre til ønsket mappe

print("=" * 65)
print("  NEMO FASE II — DATAINNHENTING")
print(f"  Estimeringsperiode: {START_YEAR}K{START_QUARTER} – siste kvartal")
print("=" * 65)


# ─── Hjelpefunksjoner ─────────────────────────────────────────────────────────

def hp_filter(y: np.ndarray, lam: float = 1600):
    """HP-filter, returnerer (trend, syklus)."""
    T = len(y)
    D = np.zeros((T - 2, T))
    for i in range(T - 2):
        D[i, i] = 1; D[i, i+1] = -2; D[i, i+2] = 1
    A = np.eye(T) + lam * D.T @ D
    trend = np.linalg.solve(A, y)
    return trend, y - trend


def log_diff(series: pd.Series) -> pd.Series:
    """Kvartalsvise log-differanser."""
    return np.log(series).diff()


def quarter_to_date(year: int, q: int) -> str:
    return f"{year}Q{q}"


def date_to_quarter(dt_str: str):
    """Konverterer '2001Q1' til (2001, 1)."""
    y, q = dt_str.split("Q")
    return int(y), int(q)


def filter_from(df: pd.DataFrame, start_year: int, start_q: int) -> pd.DataFrame:
    """Behold bare observasjoner fra og med startdato."""
    mask = [(y > start_year or (y == start_year and q >= start_q))
            for y, q in [date_to_quarter(d) for d in df.index]]
    return df[mask]


def ssb_json_stat(table_id: str, query: dict) -> pd.DataFrame:
    """
    Henter data fra SSB Statistikkbanken via JSON-stat API.

    Parametere
    ----------
    table_id : str   f.eks. "09189"
    query    : dict  SSB API-spørring (se eksempler under)

    Returnerer
    ----------
    pd.DataFrame med kvartalsverdier
    """
    url = f"https://data.ssb.no/api/v0/no/table/{table_id}"
    resp = requests.post(url, json=query, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # Parse JSON-stat
    dims  = data["dimension"]
    vals  = data["value"]
    sizes = data["size"]
    ids   = data["id"]

    # Finn tidsdimensjonen
    time_dim = None
    for d in ids:
        if "Tid" in dims[d]["label"] or d.lower() in ("tid","kvartal","year","quarter"):
            time_dim = d
            break
    if time_dim is None:
        time_dim = ids[-1]

    time_cats = list(dims[time_dim]["category"]["label"].values())

    # Anta én-dimensjonal output (én variabel per kall)
    series = pd.Series(vals, index=time_cats)
    return series


def norges_bank_api(series_key: str, freq: str = "Q") -> pd.Series:
    """
    Henter tidsseriedata fra Norges Banks Data API.
    https://data.norges-bank.no/api/

    Eksempel series_key: "NIBOR/3M", "POLICY_RATE", "EXCHANGE_RATES/B.NOK.EUR.SP"
    """
    url = f"https://data.norges-bank.no/api/data/{series_key}"
    params = {
        "startPeriod": f"{START_YEAR}-01-01",
        "format": "sdmx-json",
        "locale": "no",
    }
    if freq:
        params["frequency"] = freq

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # Parse SDMX-JSON
    obs  = data["data"]["dataSets"][0]["series"]
    dims = data["data"]["structure"]["dimensions"]["series"]
    time = data["data"]["structure"]["dimensions"]["observation"][0]

    times = [p["id"] for p in time["values"]]
    result = {}
    for key, ser in obs.items():
        for t_idx, val_list in ser["observations"].items():
            t = times[int(t_idx)]
            result[t] = val_list[0] if val_list[0] is not None else np.nan

    series = pd.Series(result).sort_index()
    series.index = [t[:7].replace("-", "Q").replace("Q0", "Q")
                    if "Q" not in t else t for t in series.index]
    return series


# ═══════════════════════════════════════════════════════════════════════════════
# BLOKK 1: BNP OG NASJONALREGNSKAPSSTØRRELSER (SSB)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[1/7] BNP og nasjonalregnskap (SSB)")

# Tabell 09189: BNP fastland, privat konsum, offentlig konsum, investering,
#               eksport og import — kvartalsvise volumindekser (2020=100)
QUERY_NR = {
    "query": [
        {
            "code": "ContentsCode",
            "selection": {
                "filter": "item",
                "values": [
                    "BNPfastland",   # BNP Fastlands-Norge
                    "PK",            # Privat konsum
                    "OK",            # Offentlig konsum
                    "BINV",          # Brutto realinvesteringer
                    "EKSPORT",       # Eksport varer og tjenester
                    "IMPORT",        # Import varer og tjenester
                ]
            }
        },
        {
            "code": "Tid",
            "selection": {
                "filter": "item",
                "values": [f"{y}K{q}" for y in range(START_YEAR, 2025)
                           for q in range(1, 5)]
            }
        }
    ],
    "response": {"format": "json-stat2"}
}

# Merk: Kjør dette lokalt — koden under illustrerer kallet
print("  → SSB Tabell 09189 (Nasjonalregnskap, kvartalsvise volumindekser)")
print("  URL: https://data.ssb.no/api/v0/no/table/09189")
print("  Variabler: BNP fastland, privat konsum, off. konsum, investering, eksport, import")

# Syntetiske data for testing (erstattes med faktisk API-kall)
np.random.seed(42)
T_periods = (2024 - START_YEAR) * 4
quarters = [f"{START_YEAR + t//4}Q{t%4+1}" for t in range(T_periods)]

# Lag realistiske norske makrodata (fra HP-filtrerte trender + syklus)
trend_Y = np.linspace(100, 140, T_periods)
cycle_Y = 1.5 * np.sin(np.linspace(0, 4*np.pi, T_periods)) + \
          0.5 * np.random.randn(T_periods)
Y_raw   = trend_Y + cycle_Y

nr_data = pd.DataFrame({
    "bnp_fastland": Y_raw,
    "privat_konsum": 0.5 * Y_raw + np.random.randn(T_periods) * 0.8,
    "off_konsum":    0.25 * Y_raw + np.random.randn(T_periods) * 0.3,
    "investering":   0.20 * Y_raw + np.random.randn(T_periods) * 2.0,
    "eksport":       0.23 * Y_raw + np.random.randn(T_periods) * 1.5,
    "import_":       0.34 * Y_raw + np.random.randn(T_periods) * 1.8,
}, index=quarters)

print(f"  Hentet {len(nr_data)} kvartaler (syntetisk — erstatt med API-kall)")


# ═══════════════════════════════════════════════════════════════════════════════
# BLOKK 2: PRISER OG INFLASJON (SSB)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[2/7] Priser og inflasjon (SSB)")

# KPI (tabell 03013) og KPI-JAE (tabell 10235)
print("  → SSB Tabell 03013 (KPI totalindeks, månedlig → kvartalssnitt)")
print("  → SSB Tabell 10235 (KPI-JAE, justert for avgifter og energi)")
print("  → SSB Tabell 10235 (Importprisindeks)")

# Syntetisk
pi_underlying = np.cumsum(
    0.005 + 0.002 * np.sin(np.linspace(0, 6*np.pi, T_periods)) +
    0.001 * np.random.randn(T_periods)
)
KPI_index = 100 * np.exp(pi_underlying)

pris_data = pd.DataFrame({
    "KPI":       KPI_index,
    "KPI_JAE":   KPI_index * (1 + 0.001 * np.random.randn(T_periods)),
    "importpris": KPI_index * 0.95 * (1 + 0.01 * np.random.randn(T_periods)),
}, index=quarters)

print(f"  Hentet {len(pris_data)} kvartaler (syntetisk)")


# ═══════════════════════════════════════════════════════════════════════════════
# BLOKK 3: ARBEIDSMARKED (SSB)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[3/7] Arbeidsmarked (SSB)")

# Sysselsetting: AKU (tabell 05111)
# Lønnsindeks: KvartalsvIs lønnsstatistikk (tabell 09786)
print("  → SSB Tabell 05111 (AKU: sysselsatte, kvartalsvise)")
print("  → SSB Tabell 09786 (Lønnsindeks, kvartalsvise)")

sysseltrend = np.linspace(2400, 2800, T_periods)  # tusen sysselsatte
arbeid_data = pd.DataFrame({
    "sysselsetting": sysseltrend + 20 * np.sin(np.linspace(0, 4*np.pi, T_periods)) + 5 * np.random.randn(T_periods),
    "lonnindeks":    100 * np.exp(np.cumsum(0.007 + 0.001 * np.random.randn(T_periods))),
}, index=quarters)

print(f"  Hentet {len(arbeid_data)} kvartaler (syntetisk)")


# ═══════════════════════════════════════════════════════════════════════════════
# BLOKK 4: RENTER (NORGES BANK)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[4/7] Renter (Norges Bank)")

# Styringsrente: https://data.norges-bank.no/api/data/POLICY_RATE
# 3M NIBOR:      https://data.norges-bank.no/api/data/NIBOR/3M
print("  → Norges Bank API: Styringsrente (POLICY_RATE)")
print("  URL: https://data.norges-bank.no/api/data/POLICY_RATE?startPeriod=2001-01-01&format=sdmx-json")
print("  → Norges Bank API: 3M NIBOR")
print("  URL: https://data.norges-bank.no/api/data/NIBOR/3M?startPeriod=2001-01-01&format=sdmx-json")

# Syntetisk rente (reflekterer faktisk norsk rentehistorie)
rate_path = np.array([
    # 2001-2005: fallende fra 7% til 2%
    *np.linspace(0.07, 0.02, 20),
    # 2005-2008: stigende til 5.75%
    *np.linspace(0.02, 0.0575, 12),
    # 2008-2009: krise, kuttet til 1.25%
    *np.linspace(0.0575, 0.0125, 6),
    # 2010-2011: opp til 2.25%
    *np.linspace(0.0125, 0.0225, 6),
    # 2012-2019: ned til 0.5%
    *np.linspace(0.0225, 0.005, 28),
    # 2020: COVID, kutt til 0%
    *np.linspace(0.005, 0.0, 4),
    # 2021: lav
    *np.linspace(0.0, 0.005, 4),
    # 2022-2023: kraftig heving
    *np.linspace(0.005, 0.045, 8),
])
rate_path = rate_path[:T_periods]
if len(rate_path) < T_periods:
    rate_path = np.concatenate([rate_path,
        np.full(T_periods - len(rate_path), rate_path[-1])])

rente_data = pd.DataFrame({
    "styringsrente": rate_path,
    "nibor_3m":      rate_path + 0.003 + 0.001 * np.random.randn(T_periods),
}, index=quarters)

print(f"  Hentet {len(rente_data)} kvartaler (syntetisk)")


# ═══════════════════════════════════════════════════════════════════════════════
# BLOKK 5: VALUTAKURS OG OLJEPRIS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[5/7] Valutakurs og oljepris")

# NOK/EUR: Norges Bank
# URL: https://data.norges-bank.no/api/data/EXR/B.EUR.NOK.SP
print("  → Norges Bank API: NOK/EUR (EXR/B.EUR.NOK.SP)")
print("  URL: https://data.norges-bank.no/api/data/EXR/B.EUR.NOK.SP")
# Oljepris Brent (USD): IMF Primary Commodity Prices
print("  → IMF API: Brent crude (POILBRE_USD)")
print("  URL: https://www.imf.org/external/datamapper/api/v1/POILBRE")

nok_eur_levels = np.cumsum([0] + list(0.002 * np.random.randn(T_periods - 1))) + 8.3
oil_path = np.array([
    *np.linspace(25, 145, 28),    # 2001-2007: opp til ~145
    *np.linspace(145, 40, 4),     # 2008: krise
    *np.linspace(40, 120, 10),    # 2009-2011: opp igjen
    *np.linspace(120, 50, 6),     # 2014-2015: nedtur
    *np.linspace(50, 75, 12),     # 2016-2019: moderat
    *np.linspace(75, 25, 4),      # 2020: COVID
    *np.linspace(25, 110, 6),     # 2021-2022: opp
    *np.linspace(110, 80, 6),     # 2023-2024: ned
])[:T_periods]
if len(oil_path) < T_periods:
    oil_path = np.concatenate([oil_path, np.full(T_periods - len(oil_path), oil_path[-1])])

valuta_data = pd.DataFrame({
    "nok_eur": nok_eur_levels[:T_periods],
    "brent_usd": oil_path,
}, index=quarters)

print(f"  Hentet {len(valuta_data)} kvartaler (syntetisk)")


# ═══════════════════════════════════════════════════════════════════════════════
# BLOKK 6: HANDELSPARTNERNES BNP (IMF / OECD)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[6/7] Handelspartnernes BNP (IMF)")

# IMF World Economic Outlook Database
# URL: https://www.imf.org/external/datamapper/api/v1/NGDP_RPCH
# Alternativt: ECB Statistical Data Warehouse for EZ
print("  → IMF WEO: BNP handelspartnere (NGDP_RPCH for EU-land)")
print("  URL: https://www.imf.org/external/datamapper/api/v1/NGDP_RPCH/@EU")
print("  NB: IMF WEO er årsdata — interpoler til kvartal med Chow-Lin")

# Syntetisk
global_trend = np.linspace(100, 145, T_periods)
global_cycle = (2.0 * np.sin(np.linspace(0, 5*np.pi, T_periods)) +
                0.5 * np.random.randn(T_periods))
global_cycle[28:32] -= 5.0   # Finanskrise 2008-2009
global_cycle[76:80] -= 4.0   # COVID 2020

foreign_data = pd.DataFrame({
    "bnp_handelspartnere": global_trend + global_cycle,
}, index=quarters)

print(f"  Hentet {len(foreign_data)} kvartaler (syntetisk)")


# ═══════════════════════════════════════════════════════════════════════════════
# BLOKK 7: BOLIGMARKED OG KREDITT (SSB / NORGES BANK)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[7/7] Boligmarked og kreditt")

# Boligprisindeks: SSB tabell 07241
# Husholdningenes kreditt (K2): Norges Bank
print("  → SSB Tabell 07241 (Boligprisindeks, kvartalsvise)")
print("  URL: https://data.ssb.no/api/v0/no/table/07241")
print("  → Norges Bank: K2 Husholdningenes kredittvekst")
print("  URL: https://data.norges-bank.no/api/data/CREDIT_INDICATOR/K2.HH")

bolig_trend = 100 * np.exp(np.cumsum(0.012 + 0.002 * np.random.randn(T_periods)))
bolig_trend[76:80] /= 1.05  # COVID-dip
kreditt_trend = 100 * np.exp(np.cumsum(0.008 + 0.001 * np.random.randn(T_periods)))

bolig_data = pd.DataFrame({
    "boligprisindeks": bolig_trend,
    "k2_husholdning":  kreditt_trend,
}, index=quarters)

print(f"  Hentet {len(bolig_data)} kvartaler (syntetisk)")


# ═══════════════════════════════════════════════════════════════════════════════
# BLOKK 8: TRANSFORMASJONER (Kravik & Mimir 2019, Appendiks A)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[Transformasjoner] Kravik og Mimir (2019), Appendiks A")

# Samle alle serier
raw = pd.concat([nr_data, pris_data, arbeid_data, rente_data,
                 valuta_data, foreign_data, bolig_data], axis=1)
raw = raw.dropna()

# HP-filter λ=1600 for realøkonomiske størrelser
def hp_cycle(series):
    arr = np.log(series.values.astype(float))
    arr = arr[~np.isnan(arr)]
    _, cyc = hp_filter(arr, lam=1600)
    return cyc

obs = pd.DataFrame(index=raw.index)

# 1. dy_obs: BNP-vekst (log-diff)
obs["dy_obs"]     = log_diff(raw["bnp_fastland"])

# 2. dc_obs: Konsumvekst (log-diff)
obs["dc_obs"]     = log_diff(raw["privat_konsum"])

# 3. dinv_obs: Investeringsvekst (log-diff)
obs["dinv_obs"]   = log_diff(raw["investering"])

# 4. dx_obs: Eksportvekst (log-diff)
obs["dx_obs"]     = log_diff(raw["eksport"])

# 5. dm_obs: Importvekst (log-diff)
obs["dm_obs"]     = log_diff(raw["import_"])

# 6. pi_obs: KPI-inflasjon annualisert (log-diff × 4)
obs["pi_obs"]     = log_diff(raw["KPI"]) * 4

# 7. pi_core_obs: Kjerneinflasjon annualisert
obs["pi_core_obs"] = log_diff(raw["KPI_JAE"]) * 4

# 8. dw_obs: Lønnsvekst (log-diff)
obs["dw_obs"]     = log_diff(raw["lonnindeks"])

# 9. dl_obs: Sysselsettingsvekst (log-diff)
obs["dl_obs"]     = log_diff(raw["sysselsetting"])

# 10. i_R_obs: Styringsrente (kvartalsverdi, ikke annualisert)
obs["i_R_obs"]    = raw["styringsrente"] / 4   # kvartalsvise renter

# 11. i_3m_obs: 3M NIBOR (kvartalsverdi)
obs["i_3m_obs"]   = raw["nibor_3m"] / 4

# 12. ds_obs: Valutakursendring (log-diff NOK/EUR)
obs["ds_obs"]     = log_diff(raw["nok_eur"])

# 13. dpO_obs: Oljeprisvekst (log-diff, real USD)
#   Deflater med US CPI (forenklet: bruk råpris her)
obs["dpO_obs"]    = log_diff(raw["brent_usd"])

# 14. dyS_obs: Handelspartner-BNP (HP-gap)
yS_log = np.log(raw["bnp_handelspartnere"].values.astype(float))
_, cyc_yS = hp_filter(yS_log, lam=1600)
obs["dyS_obs"]    = pd.Series(cyc_yS, index=raw.index)

# 15. dh_obs: Boligprisvekst (log-diff)
obs["dh_obs"]     = log_diff(raw["boligprisindeks"])

# 16. db_obs: Kredittvekst (log-diff)
obs["db_obs"]     = log_diff(raw["k2_husholdning"])

# Fjern NaN fra differansiering
obs = obs.dropna()

# Demean (stasjonaritetsbetingelse for MH)
obs_demeaned = obs - obs.mean()

print(f"\n  Observasjonsmatrise: {obs_demeaned.shape[0]} kvartaler × {obs_demeaned.shape[1]} variabler")
print(f"  Periode: {obs_demeaned.index[0]} – {obs_demeaned.index[-1]}")


# ═══════════════════════════════════════════════════════════════════════════════
# BLOKK 9: DIAGNOSTIKK
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[Diagnostikk] Deskriptiv statistikk")
print(f"\n  {'Variabel':<18} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print(f"  {'─'*52}")
for col in obs_demeaned.columns:
    s = obs_demeaned[col]
    print(f"  {col:<18} {s.mean():>8.4f} {s.std():>8.4f} {s.min():>8.4f} {s.max():>8.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# BLOKK 10: LAGRE OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

csv_path  = os.path.join(OUTPUT_DIR, "nemo_data.csv")
json_path = os.path.join(OUTPUT_DIR, "nemo_data.json")

obs_demeaned.to_csv(csv_path)
obs_demeaned.to_json(json_path, orient="index", indent=2)

print(f"\n  Lagret: {csv_path}")
print(f"  Lagret: {json_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# BRUKERVEILEDNING FOR FAKTISK DATAINNHENTING
# ═══════════════════════════════════════════════════════════════════════════════

GUIDE = """
================================================================================
INSTRUKSJONER FOR FAKTISK DATAINNHENTING
================================================================================

For å hente faktiske data: erstatt de syntetiske blokkene over med disse kallene.

─── SSB Statistikkbanken ────────────────────────────────────────────────────────

BNP og nasjonalregnskap (kvartalsvise volumindekser):
  Tabell: 09189
  URL:    https://data.ssb.no/api/v0/no/table/09189
  Format: POST med JSON-spørring (se QUERY_NR ovenfor)
  Variabler å velge: BNPfastland, PK, OK, BINV, EKSPORT, IMPORT

KPI:
  Tabell: 03013 (månedlig) → gjennomsnitt per kvartal
  URL:    https://data.ssb.no/api/v0/no/table/03013

KPI-JAE:
  Tabell: 10235 (månedlig)
  URL:    https://data.ssb.no/api/v0/no/table/10235

Importprisindeks:
  Tabell: 08946
  URL:    https://data.ssb.no/api/v0/no/table/08946

Sysselsetting (AKU):
  Tabell: 05111 (kvartalsvise)
  URL:    https://data.ssb.no/api/v0/no/table/05111

Lønnsindeks:
  Tabell: 09786 (kvartalsvise)
  URL:    https://data.ssb.no/api/v0/no/table/09786

Boligprisindeks:
  Tabell: 07241 (kvartalsvise)
  URL:    https://data.ssb.no/api/v0/no/table/07241

─── Norges Bank API ─────────────────────────────────────────────────────────────

Styringsrente (dagsdata → kvartalssnitt):
  URL: https://data.norges-bank.no/api/data/POLICY_RATE
      ?startPeriod=2001-01-01&format=sdmx-json&locale=no

3M NIBOR:
  URL: https://data.norges-bank.no/api/data/NIBOR/3M
      ?startPeriod=2001-01-01&format=sdmx-json

NOK/EUR valutakurs (spot, dagsdata → kvartalssnitt):
  URL: https://data.norges-bank.no/api/data/EXR/B.EUR.NOK.SP
      ?startPeriod=2001-01-01&format=sdmx-json

K2 Husholdningenes kredittvekst:
  URL: https://data.norges-bank.no/api/data/CREDIT_INDICATOR/K2.HH
      ?startPeriod=2001-01-01&format=sdmx-json

─── IMF / Verdensbanken ─────────────────────────────────────────────────────────

Oljepris Brent USD (månedlig → kvartal):
  URL: https://www.imf.org/external/datamapper/api/v1/POILBRE

Handelspartnernes BNP (årsdata → Chow-Lin til kvartal):
  Norges viktigste handelspartnere (BNP-vekter, ca.):
    EU (60%), USA (10%), UK (10%), resten (20%)
  Hent per land fra IMF WEO:
    https://www.imf.org/external/datamapper/api/v1/NGDP_RPCH/DEU/GBR/USA/SWE

  Alternativt: ECB Area Wide Model data (kvartal, eurosonens BNP)
    https://sdw-wsrest.ecb.europa.eu/service/data/MNA/Q.Y.I8.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.LR.N

─── Transformasjoner (Kravik og Mimir 2019, Appendiks A) ────────────────────────

Alle realøkonomiske variable: log-differanser (kvartalsvekst)
Renter: kvartalsverdier (ikke annualisert i selve datavektoren)
Inflasjon: log-differanse × 4 (annualisert kvartalsvekst)
Sykliske variable (alternativt): HP-gap med λ=1600

Demeaning: trekk fra sampelgjennomsnittet per variabel
    (krever stasjonære observasjoner for likelihood-beregning)

─── Observasjoner for Bayesiansk estimering ─────────────────────────────────────

Brukt i Kravik og Mimir (2019) (seksjon 3.1):
  - Periode: 2001K1 – 2017K4 (estimering), 2018K1 – 2019K1 (evaluering)
  - 13 observerte variabler (se Appendiks A)
  - Målingsfeil på alle variable unntatt renter (∼10% av variansen)

Anbefalte ekstra perioder for dette prosjektet:
  - Ta med til og med siste kvartal (2024K4 eller oppdatert)
  - Identifiser strukturelle brudd: 2008K3 (Lehman), 2020K1 (COVID)
    → Vurder dummyvariable eller kortere estimeringsperiode
================================================================================
"""
print(GUIDE)
print("=" * 65)
print("  Kjøring fullført.")
print(f"  Data lagret: nemo_data.csv, nemo_data.json")
print("=" * 65)
