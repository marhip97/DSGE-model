"""
NEMO datapipeline — Fase 1.

Automatisk innhenting og transformasjon av 14 observerte variabler
til Kalman-filterets observasjonsvektor Y.

Transformasjoner følger Kravik og Mimir (2019), Appendiks A.

Kjør som modul:
    python -m nemo.data.pipeline [--no-cache]

Output:
    data/processed/nemo_data.csv       — demeaned observasjoner
    data/processed/nemo_demean.json    — sampelgjennomsnitt per variabel
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from nemo.data.ssb import (
    hent_nasjonalregnskap,
    hent_kpi,
    hent_kpi_jae,
    hent_lonnsinndeks,
    hent_boligprisindeks,
)
from nemo.data.norges_bank import (
    hent_styringsrente,
    hent_nibor_3m,
    hent_valutakurs_nok_eur,
    hent_k2_husholdning,
)
from nemo.data.fred import (
    hent_brent_oljepris,
    hent_euro_bnp,
)

logger = logging.getLogger(__name__)

# ─── Konstantar ────────────────────────────────────────────────────────────────

OBSERVASJONSVARIABLER = [
    "dy_obs",    # BNP fastland, log-diff
    "dc_obs",    # Privat konsum, log-diff
    "dinv_obs",  # Bruttoinvesteringer, log-diff
    "dx_obs",    # Eksport, log-diff
    "dm_obs",    # Import, log-diff
    "pi_obs",    # KPI-inflasjon, annualisert log-diff × 4
    "dw_obs",    # Lønnsindeks, log-diff
    "i_R_obs",   # Styringsrente / 400 (kvartalsverdier)
    "i_3m_obs",  # NIBOR 3M / 400
    "ds_obs",    # NOK/EUR valutakurs, log-diff
    "dpO_obs",   # Brent oljepris, log-diff
    "dyS_obs",   # Handelspartner-BNP, HP-gap
    "dh_obs",    # Boligprisindeks, log-diff
    "db_obs",    # K2 husholdning, log-diff
]

ESTIMERINGSSTART = pd.Timestamp("2001-01-01")
ESTIMERINGSSLUTT = pd.Timestamp("2026-01-01")


def _finn_repo_root() -> Path:
    """Finner repo-rotkatalog ved å lete etter pyproject.toml."""
    kandidat = Path(__file__).resolve()
    for _ in range(10):
        if (kandidat / "pyproject.toml").exists():
            return kandidat
        kandidat = kandidat.parent
    raise FileNotFoundError("Fant ikke pyproject.toml — kan ikke bestemme repo-rot.")


def hp_filter(y: np.ndarray, lam: float = 1600) -> tuple[np.ndarray, np.ndarray]:
    """
    Hodrick-Prescott-filter for separering av trend og syklus.

    Løser: min_{g} sum_t (y_t - g_t)^2 + lam * sum_t (Δ²g_t)^2

    Parametere
    ----------
    y : np.ndarray
        Tidsseriedata (1D-array).
    lam : float
        Glattingsparameter. Standard for kvartalsdata: 1600 (Hodrick & Prescott 1997).

    Returnerer
    ----------
    tuple[np.ndarray, np.ndarray]
        (trend, syklus) — begge med samme lengde som y.
    """
    T = len(y)
    if T < 4:
        raise ValueError(f"HP-filter krever minst 4 observasjoner, fikk {T}.")

    # Bygg andregangs differansematrise D (T-2 × T)
    D = np.zeros((T - 2, T))
    for i in range(T - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0

    A = np.eye(T) + lam * D.T @ D

    with np.errstate(all="warn"):
        trend = np.linalg.solve(A, y)

    syklus = y - trend
    return trend, syklus


def log_diff(series: pd.Series) -> pd.Series:
    """
    Beregner kvartalsvise log-differanser av en tidsserie.

    Parametere
    ----------
    series : pd.Series
        Nivåverdier (må være positive).

    Returnerer
    ----------
    pd.Series
        Log-differanser: ln(x_t) - ln(x_{t-1}).
    """
    with np.errstate(divide="warn", invalid="warn"):
        log_s = np.log(series.astype(float))
    return log_s.diff()


def _filtrer_periode(df: pd.DataFrame) -> pd.DataFrame:
    """Behold kun observasjoner innenfor estimeringsperioden."""
    mask = (df.index >= ESTIMERINGSSTART) & (df.index <= ESTIMERINGSSLUTT)
    return df.loc[mask]


def transformer_til_obs(
    nr: pd.DataFrame,
    kpi: pd.Series,
    lonn: pd.Series,
    bolig: pd.Series,
    styringsrente: pd.Series,
    nibor: pd.Series,
    nok_eur: pd.Series,
    k2: pd.Series,
    brent: pd.Series,
    euro_bnp: pd.Series,
    kpi_jae: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Transformerer rådata til de 14 observerte NEMO-variablene.

    Transformasjoner følger Kravik og Mimir (2019), Appendiks A.

    Parametere
    ----------
    nr : pd.DataFrame
        Nasjonalregnskap (bnp_fastland, privat_konsum, bruttoinvesteringer,
        eksport, import_).
    kpi : pd.Series
        KPI-indeks (kvartalssnitt).
    kpi_jae : pd.Series or None
        KPI-JAE-indeks (kvartalssnitt). Hvis gitt, legges pi_core_obs til.
    lonn : pd.Series
        Lønnsindeks (kvartal).
    bolig : pd.Series
        Boligprisindeks (kvartal).
    styringsrente : pd.Series
        Styringsrente (annualisert %, kvartalsgj.snitt).
    nibor : pd.Series
        NIBOR 3M (annualisert %, kvartalsgj.snitt).
    nok_eur : pd.Series
        NOK/EUR spotkurs (kvartalssnitt).
    k2 : pd.Series
        K2 husholdningskreditt (nivå, kvartalssnitt).
    brent : pd.Series
        Brent oljepris (USD per fat, kvartalssnitt).
    euro_bnp : pd.Series
        Euro-sone BNP (volum, kvartal).

    Returnerer
    ----------
    pd.DataFrame
        DataFrame med 14 NEMO-observasjonsvariabler og kvartalsvise datoer.
    """
    obs = pd.DataFrame(index=nr.index, dtype=float)

    # 1. dy_obs: BNP fastland, log-differanse (kvartalsvekst)
    obs["dy_obs"] = log_diff(nr["bnp_fastland"])

    # 2. dc_obs: Privat konsum, log-differanse
    obs["dc_obs"] = log_diff(nr["privat_konsum"])

    # 3. dinv_obs: Bruttoinvesteringer, log-differanse
    obs["dinv_obs"] = log_diff(nr["bruttoinvesteringer"])

    # 4. dx_obs: Eksport, log-differanse
    obs["dx_obs"] = log_diff(nr["eksport"])

    # 5. dm_obs: Import, log-differanse
    obs["dm_obs"] = log_diff(nr["import_"])

    # 6. pi_obs: KPI-inflasjon, annualisert (log-diff × 4)
    kpi_aligned = kpi.reindex(nr.index)
    obs["pi_obs"] = log_diff(kpi_aligned) * 4
    if kpi_jae is not None:
        kpi_jae_aligned = kpi_jae.reindex(nr.index)
        obs["pi_core_obs"] = log_diff(kpi_jae_aligned) * 4

    # 7. dw_obs: Lønnsvekst, log-differanse
    lonn_aligned = lonn.reindex(nr.index)
    obs["dw_obs"] = log_diff(lonn_aligned)

    # 8. i_R_obs: Styringsrente / 400 (kvartalsverdier fra annualisert %)
    sr_aligned = styringsrente.reindex(nr.index)
    obs["i_R_obs"] = sr_aligned / 400.0

    # 9. i_3m_obs: NIBOR 3M / 400
    nibor_aligned = nibor.reindex(nr.index)
    obs["i_3m_obs"] = nibor_aligned / 400.0

    # 10. ds_obs: Valutakursendring, log-differanse NOK/EUR
    nok_aligned = nok_eur.reindex(nr.index)
    obs["ds_obs"] = log_diff(nok_aligned)

    # 11. dpO_obs: Brent oljeprisvekst, log-differanse
    brent_aligned = brent.reindex(nr.index)
    obs["dpO_obs"] = log_diff(brent_aligned)

    # 12. dyS_obs: Handelspartner-BNP, HP-gap av log-nivå
    euro_aligned = euro_bnp.reindex(nr.index)
    euro_valid = euro_aligned.dropna()
    if len(euro_valid) >= 4:
        with np.errstate(divide="warn", invalid="warn"):
            log_euro = np.log(euro_valid.astype(float).values)
        _, syklus = hp_filter(log_euro, lam=1600)
        dyS = pd.Series(syklus, index=euro_valid.index)
        obs["dyS_obs"] = dyS.reindex(nr.index)
    else:
        logger.warning("For få handelspartner-BNP-observasjoner til HP-filter.")
        obs["dyS_obs"] = np.nan

    # 13. dh_obs: Boligprisvekst, log-differanse
    bolig_aligned = bolig.reindex(nr.index)
    obs["dh_obs"] = log_diff(bolig_aligned)

    # 14. db_obs: K2 husholdning, log-differanse
    k2_aligned = k2.reindex(nr.index)
    obs["db_obs"] = log_diff(k2_aligned)

    return obs[OBSERVASJONSVARIABLER]


def demean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Demeaner alle kolonner og returnerer gjennomsnittene.

    Parametere
    ----------
    df : pd.DataFrame
        Observasjonsmatrise med eventuelle NaN-verdier.

    Returnerer
    ----------
    tuple[pd.DataFrame, dict]
        (demeaned_df, means) der means er en dict {kolonne: gjennomsnitt}.
    """
    means = df.mean().to_dict()
    df_demeaned = df - df.mean()
    return df_demeaned, means


def kjor_pipeline(bruk_cache: bool = True, inkluder_kpi_jae: bool = False) -> pd.DataFrame:
    """
    Kjører den fullstendige datapipelinen og lagrer resultater.

    1. Henter rådata fra SSB, Norges Bank og FRED.
    2. Transformerer til 14 NEMO-observasjonsvariabler.
       Hvis inkluder_kpi_jae=True: henter også SSB 10235 og legger til pi_core_obs.
    3. Demeaner.
    4. Lagrer data/processed/nemo_data.csv og nemo_demean.json.

    Parametere
    ----------
    bruk_cache : bool
        Om eksisterende cache skal brukes (True = raskere, anbefalt).
        False tvinger nye API-kall.

    Returnerer
    ----------
    pd.DataFrame
        Demeaned observasjonsmatrise klar for Kalman-filter.
    """
    logger.info("=== NEMO datapipeline starter ===")

    # ── Steg 1: Hent rådata ─────────────────────────────────────────────────
    logger.info("[1/3] Henter rådata...")

    logger.info("  SSB: Nasjonalregnskap (09189)")
    nr = hent_nasjonalregnskap(bruk_cache=bruk_cache)

    logger.info("  SSB: KPI (03013)")
    kpi = hent_kpi(bruk_cache=bruk_cache)

    logger.info("  SSB: Lønnsindeks (09786)")
    lonn = hent_lonnsinndeks(bruk_cache=bruk_cache)

    logger.info("  SSB: Boligprisindeks (07241)")
    bolig = hent_boligprisindeks(bruk_cache=bruk_cache)

    logger.info("  NB: Styringsrente (POLICY_RATE)")
    styringsrente = hent_styringsrente(bruk_cache=bruk_cache)

    logger.info("  NB: NIBOR 3M")
    nibor = hent_nibor_3m(bruk_cache=bruk_cache)

    logger.info("  NB: NOK/EUR valutakurs")
    nok_eur = hent_valutakurs_nok_eur(bruk_cache=bruk_cache)

    logger.info("  NB: K2 husholdning")
    k2 = hent_k2_husholdning(bruk_cache=bruk_cache)

    logger.info("  FRED: Brent oljepris (%s)", "DCOILBRENTEU")
    brent = hent_brent_oljepris(bruk_cache=bruk_cache)

    logger.info("  FRED: Euro-sone BNP (handelspartnere)")
    euro_bnp = hent_euro_bnp(bruk_cache=bruk_cache)

    kpi_jae = None
    if inkluder_kpi_jae:
        logger.info("  SSB: KPI-JAE (10235)")
        kpi_jae = hent_kpi_jae(bruk_cache=bruk_cache)

    # ── Steg 2: Transformer ──────────────────────────────────────────────────
    logger.info("[2/3] Transformerer til observasjonsvariabler...")

    obs_raw = transformer_til_obs(
        nr=nr,
        kpi=kpi,
        lonn=lonn,
        bolig=bolig,
        styringsrente=styringsrente,
        nibor=nibor,
        nok_eur=nok_eur,
        k2=k2,
        brent=brent,
        euro_bnp=euro_bnp,
        kpi_jae=kpi_jae,
    )

    # Filtrer til estimeringsperiode
    obs_raw = _filtrer_periode(obs_raw)

    # Fjern rader med for mange NaN (første observasjon mangler alltid pga. diff)
    obs_raw = obs_raw.dropna(how="all")

    # Logg NaN-teller per kolonne
    nan_sum = obs_raw.isna().sum()
    if nan_sum.any():
        logger.warning(
            "NaN-verdier i obs-matrise:\n%s",
            nan_sum[nan_sum > 0].to_string()
        )

    # ── Steg 3: Demean og lagre ──────────────────────────────────────────────
    logger.info("[3/3] Demeaner og lagrer...")

    obs_demeaned, means = demean(obs_raw)

    # Lagre
    repo_root = _finn_repo_root()
    processed_dir = repo_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    if inkluder_kpi_jae:
        csv_path  = processed_dir / "nemo_data_kpi_jae.csv"
        json_path = processed_dir / "nemo_demean_kpi_jae.json"
    else:
        csv_path  = processed_dir / "nemo_data.csv"
        json_path = processed_dir / "nemo_demean.json"

    obs_demeaned.to_csv(csv_path, index_label="kvartal")
    logger.info("Lagret: %s (%d obs × %d var)", csv_path, *obs_demeaned.shape)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(means, f, indent=2, ensure_ascii=False)
    logger.info("Lagret: %s", json_path)

    logger.info("=== Pipeline fullført ===")
    logger.info(
        "Periode: %s — %s (%d kvartaler)",
        obs_demeaned.index[0].date(),
        obs_demeaned.index[-1].date(),
        len(obs_demeaned),
    )

    return obs_demeaned


def _konfigurer_logging(nivaa: str = "INFO") -> None:
    """Setter opp konsolllogging for pipeline-kjøring."""
    logging.basicConfig(
        level=getattr(logging, nivaa.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def main() -> None:
    """Inngangspunkt for `python -m nemo.data.pipeline`."""
    parser = argparse.ArgumentParser(
        description="NEMO datapipeline — henter og transformerer makrodata."
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Tving nye API-kall (ignorer eksisterende cache).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Loggingsnivå (standard: INFO).",
    )
    parser.add_argument(
        "--kpi-jae",
        action="store_true",
        help="Hent også KPI-JAE (SSB 10235) og lag nemo_data_kpi_jae.csv.",
    )
    args = parser.parse_args()

    _konfigurer_logging(args.log_level)
    bruk_cache = not args.no_cache

    try:
        df = kjor_pipeline(bruk_cache=bruk_cache, inkluder_kpi_jae=args.kpi_jae)
        print(f"\nFerdig! {len(df)} kvartaler × {len(df.columns)} variabler.")
        print(f"Periode: {df.index[0].date()} — {df.index[-1].date()}")
        print("\nDeskriptiv statistikk:")
        print(df.describe().to_string())
    except Exception as exc:
        logger.error("Pipeline feilet: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
