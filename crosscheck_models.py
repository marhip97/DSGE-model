"""
================================================================================
NEMO FASE III — MODELLBIBLIOTEK FOR KRYSSJEKKMODELLER
Milepel 3

Inneholder tre reduserte BVAR-er med Minnesota prior, univariate AR-modeller
for eksogene drivere, og månedlige ARIMA-modeller for KPI og styringsrente
(nowcast-supplement per alternativ 1).

Arkitektur:
    BVAR-1  Realøkonomi        : BNP, konsum, investering, eksport, import
    BVAR-2  Priser og lønn     : KPI, lønn, oljepris, handelspartner-BNP
    BVAR-3  Finansielle/bolig  : styringsrente, NIBOR, NOK/EUR, boligpris, kreditt

    AR      Oljepris, handelspartner-BNP (eksogene drivere)
    ARIMA   KPI månedlig, styringsrente månedlig (nowcast)

Gibbs-sampler: 2 000 draws, 1 000 burn-in (Minnesota semi-analytisk posterior).

Bruk:
    from crosscheck_models import ModelLibrary
    lib = ModelLibrary(data_path="crosscheck_data.csv")
    lib.fit_all()
    results = lib.forecast_all(h=12)
    lib.save(results, "crosscheck_results.json")

    Eller fra kommandolinje:
    python crosscheck_models.py --data crosscheck_data.csv --horizon 12

Avhengigheter:
    pip install pandas numpy scipy statsmodels
================================================================================
"""

import argparse
import json
import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.stats import invwishart, matrix_normal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("crosscheck_models")

# ── Konstantar ────────────────────────────────────────────────────────────────
COVID_START = "2020Q1"
COVID_END   = "2021Q4"

# Variabelgrupper
BVAR1_VARS  = ["dy_obs", "dc_obs", "dinv_obs", "dx_obs", "dm_obs"]
BVAR2_VARS  = ["pi_obs", "dw_obs", "dpO_obs", "dyS_obs"]
BVAR3_VARS  = ["i_R_obs", "i_3m_obs", "ds_obs", "dh_obs", "db_obs"]
AR_VARS     = ["dpO_obs", "dyS_obs"]
ALL_OBS     = BVAR1_VARS + BVAR2_VARS + BVAR3_VARS

# Månedlige kolonnenavn (fra crosscheck_data_raw.csv)
MONTHLY_KPI_COL   = "kpi"
MONTHLY_RATE_COL  = "styringsrente"


# ═══════════════════════════════════════════════════════════════════════════════
# HJELPEFUNKSJONER
# ═══════════════════════════════════════════════════════════════════════════════

def _lag_matrix(Y: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bygg lagget regressormatrise for VAR(p).

    Parameters
    ----------
    Y : (T × n) observasjonsmatrise
    p : lagorden

    Returns
    -------
    X : (T-p × n*p+1) regressormatrise [konstant, y_{t-1}, ..., y_{t-p}]
    Y_dep : (T-p × n) avhengig variabel
    """
    T, n = Y.shape
    X_list = [np.ones((T - p, 1))]
    for lag in range(1, p + 1):
        X_list.append(Y[p - lag: T - lag, :])
    X = np.hstack(X_list)
    Y_dep = Y[p:, :]
    return X, Y_dep


def _select_lag_bic(Y: np.ndarray,
                    p_max: int = 4,
                    exclude_mask: Optional[np.ndarray] = None) -> int:
    """
    Velg lagorden via BIC for VAR. Returnerer p i [1, p_max].
    exclude_mask: bool-array lengde T, True = ekskluder (COVID).
    """
    T, n = Y.shape
    bic_vals = []
    for p in range(1, p_max + 1):
        X, Yd = _lag_matrix(Y, p)
        T_eff = len(Yd)
        if exclude_mask is not None:
            mask = exclude_mask[p:]
            X   = X[~mask]
            Yd  = Yd[~mask]
            T_eff = len(Yd)
        if T_eff < n * p + 2:
            bic_vals.append(np.inf)
            continue
        try:
            B_ols = np.linalg.lstsq(X, Yd, rcond=None)[0]
            resid = Yd - X @ B_ols
            Sigma = resid.T @ resid / T_eff
            sign, logdet = np.linalg.slogdet(Sigma)
            k = n * (n * p + 1)
            bic = logdet + k * np.log(T_eff) / T_eff
            bic_vals.append(float(bic))
        except Exception:
            bic_vals.append(np.inf)
    best_p = int(np.argmin(bic_vals)) + 1
    return best_p


def _ols_var(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """OLS-estimat av VAR. Returnerer (B_ols, Sigma_ols)."""
    T = len(Y)
    B = np.linalg.lstsq(X, Y, rcond=None)[0]
    resid = Y - X @ B
    Sigma = resid.T @ resid / T
    return B, Sigma


# ═══════════════════════════════════════════════════════════════════════════════
# BVAR MED MINNESOTA PRIOR
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MinnesotaPrior:
    """
    Minnesota prior for BVAR.

    Parametere (se Litterman 1986, Kadiyala & Karlsson 1997):
        lambda1 : Overall tightness — kontrollerer hvor mye posterior
                  trekkes mot prior. Lavere = mer regularisering.
                  Typisk 0.1–0.2 for makrodata.
        lambda2 : Krysslagsdempning — reduserer vekt på andre variablers
                  lag relativt til egne lag.
        lambda3 : Lag decay — høyere = raskere avtagende vekt på fjerne lag.
        mu5     : Sum-of-coefficients prior — trekker mot unit root / random walk.
                  Settes til None for å deaktivere.
    """
    lambda1: float = 0.15
    lambda2: float = 0.50
    lambda3: float = 1.00
    mu5:     Optional[float] = 1.0


class BVARMinnesota:
    """
    Bayesiansk VAR med Minnesota prior og Gibbs-sampler.

    Modell: Y_t = c + B_1 Y_{t-1} + ... + B_p Y_{t-p} + e_t
            e_t ~ N(0, Sigma)

    Posterior via Normal-Inverse-Wishart Gibbs-sampler
    (Kadiyala & Karlsson 1997, Koop & Korobilis 2010).
    """

    def __init__(self,
                 name: str,
                 variables: List[str],
                 p: int = 4,
                 prior: Optional[MinnesotaPrior] = None,
                 n_draws: int = 2000,
                 n_burnin: int = 1000):
        self.name      = name
        self.variables = variables
        self.n         = len(variables)
        self.p         = p
        self.prior     = prior or MinnesotaPrior()
        self.n_draws   = n_draws
        self.n_burnin  = n_burnin

        # Satt etter fit()
        self.B_post_mean: Optional[np.ndarray] = None
        self.Sigma_post_mean: Optional[np.ndarray] = None
        self.draws_B:     Optional[np.ndarray] = None   # (n_draws × k × n)
        self.draws_Sigma: Optional[np.ndarray] = None   # (n_draws × n × n)
        self.Y_fit:       Optional[np.ndarray] = None
        self.fitted:      bool = False

    def _build_prior_moments(self,
                              sigma_ols: np.ndarray
                              ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bygg Minnesota prior-gjennomsnitt og presisjon for vec(B).

        Prior-gjennomsnitt:
            - Egne lag-1: 1 (random walk prior for ikke-stasjonære)
              men vi bruker stasjonære differanser → sett til 0
            - Alle andre koeffisienter: 0

        Prior-presisjon (diagonal):
            Var(B_{ij,l}) = (lambda1 * lambda2)^2 / l^lambda3
                            × sigma_i / sigma_j   for i ≠ j
            Var(B_{ii,l}) = lambda1^2 / l^lambda3  for eget lag
            Var(konstant) : stor (diffus)
        """
        pr  = self.prior
        n, p = self.n, self.p
        k   = n * p + 1   # antall regressorer per ligning

        # Antall koeffisienter totalt
        B_prior_mean = np.zeros((k, n))
        # Alle serier er stasjonære differanser → prior mean = 0

        # Bygger diagonal presisjon Omega_0^{-1}
        # (varianser for hver koeffisient)
        variances = np.zeros(k)
        # Konstant: diffus
        variances[0] = 1e6

        for l in range(1, p + 1):
            for j in range(n):
                idx = 1 + (l - 1) * n + j
                if idx < k:
                    # Eget lag
                    var_ii = (pr.lambda1 ** 2) / (l ** pr.lambda3)
                    # Kryssvar — bruker sigma-ratio for skalering
                    var_ij = ((pr.lambda1 * pr.lambda2) ** 2 / (l ** pr.lambda3)
                              * sigma_ols[j, j])
                    variances[idx] = var_ij   # lagres per regressor (j)

        # Presisjonmatrise (diagonal)
        Omega_0_inv = np.diag(1.0 / np.maximum(variances, 1e-12))
        return B_prior_mean, Omega_0_inv

    def fit(self,
            data: pd.DataFrame,
            covid_mask: Optional[np.ndarray] = None) -> "BVARMinnesota":
        """
        Estimer BVAR via Gibbs-sampler.

        Parameters
        ----------
        data       : DataFrame med self.variables som kolonner
        covid_mask : bool-array (T,), True = ekskluder fra estimering
        """
        _sub = data[self.variables]
        _valid = _sub.notna().all(axis=1)
        Y_full = _sub[_valid].values.astype(float)
        T_full = len(Y_full)
        # Justerer covid_mask til gyldige rader (trailing NaN dropper siste kvartal)
        if covid_mask is not None:
            covid_mask = covid_mask[_valid.values]

        # Velg lagorden via BIC hvis ikke angitt eksplisitt (p=4 er default)
        X_full, Y_dep_full = _lag_matrix(Y_full, self.p)
        T_eff_full = len(Y_dep_full)

        # Bruk COVID-maske
        if covid_mask is not None:
            mask = covid_mask[self.p:]
            X   = X_full[~mask]
            Y_d = Y_dep_full[~mask]
        else:
            X, Y_d = X_full, Y_dep_full

        T_eff = len(Y_d)
        k = X.shape[1]

        log.info(f"  {self.name}: n={self.n}, p={self.p}, T={T_eff} (COVID-ekskl.)")

        # OLS som startpunkt og for prior-kalibrering
        B_ols, Sigma_ols = _ols_var(X, Y_d)

        # Prior-momenter
        B_prior, Omega_0_inv = self._build_prior_moments(Sigma_ols)

        # ── Gibbs-sampler ─────────────────────────────────────────────────────
        # Initialisering
        Sigma = Sigma_ols.copy()
        Sigma = (Sigma + Sigma.T) / 2 + np.eye(self.n) * 1e-6

        draws_B     = np.zeros((self.n_draws, k, self.n))
        draws_Sigma = np.zeros((self.n_draws, self.n, self.n))

        total = self.n_draws + self.n_burnin
        B_cur = B_ols.copy()

        for draw in range(total):
            # ── Steg 1: Sample B | Sigma ───────────────────────────────────
            Sigma_inv = np.linalg.inv(Sigma + np.eye(self.n) * 1e-8)
            # Posterior presisjon: Omega_n^{-1} = Omega_0^{-1} + X'(Sigma^{-1} ⊗ I)X
            # For diagonal Omega_0^{-1} og full Sigma: kolonne-vis
            B_post_cols = []
            for eq in range(self.n):
                s_ii = float(Sigma_inv[eq, eq])
                Omega_n_inv = Omega_0_inv + s_ii * (X.T @ X)
                Omega_n     = np.linalg.inv(Omega_n_inv + np.eye(k) * 1e-10)
                b_prior_eq  = B_prior[:, eq]
                b_post_mean = Omega_n @ (
                    Omega_0_inv @ b_prior_eq
                    + s_ii * (X.T @ Y_d[:, eq])
                )
                # Sample fra N(b_post_mean, Omega_n)
                try:
                    L = np.linalg.cholesky(Omega_n + np.eye(k) * 1e-10)
                    b_draw = b_post_mean + L @ np.random.randn(k)
                except np.linalg.LinAlgError:
                    b_draw = b_post_mean
                B_post_cols.append(b_draw)
            B_cur = np.column_stack(B_post_cols)

            # ── Steg 2: Sample Sigma | B ────────────────────────────────────
            resid  = Y_d - X @ B_cur
            S_post = resid.T @ resid + np.eye(self.n) * 1e-6
            S_post = (S_post + S_post.T) / 2
            df_post = T_eff + self.n + 1
            try:
                Sigma = invwishart.rvs(df=df_post, scale=S_post)
                Sigma = (Sigma + Sigma.T) / 2
            except Exception:
                Sigma = S_post / df_post

            # ── Lagre post burn-in ──────────────────────────────────────────
            if draw >= self.n_burnin:
                idx = draw - self.n_burnin
                draws_B[idx]     = B_cur
                draws_Sigma[idx] = Sigma

        self.draws_B     = draws_B
        self.draws_Sigma = draws_Sigma
        self.B_post_mean     = draws_B.mean(axis=0)
        self.Sigma_post_mean = draws_Sigma.mean(axis=0)
        self.Y_fit           = Y_full
        self.X_full          = X_full
        self.Y_dep_full      = Y_dep_full
        self.fitted          = True
        log.info(f"  {self.name}: Gibbs ferdig ({self.n_draws} draws).")
        return self

    def forecast(self,
                 h: int = 12,
                 quantiles: Tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)
                 ) -> Dict:
        """
        Produser h-kvartals prognose med usikkerhetsintervaller.

        Fremgangsmåte: For hvert posterior draw simuleres en bane h steg frem.
        Dette gir automatisk korrekte prediksjonsintervaller som inkorporerer
        både parameterusikkerhet (fra draws_B) og sjokk-usikkerhet (fra draws_Sigma).

        Returns
        -------
        dict med nøkler per variabel:
            mean   : (h,) posterior prediksjonsgjennomsnitt
            bands  : {str(q): (h,)} kvantiler
        """
        if not self.fitted:
            raise RuntimeError(f"{self.name}: kall fit() først.")

        n, p = self.n, self.p
        # Siste p observasjoner som startpunkt
        last_obs = self.Y_fit[-p:].copy()  # (p × n)

        all_paths = np.zeros((self.n_draws, h, n))

        for d in range(self.n_draws):
            B_d     = self.draws_B[d]       # (k × n)
            Sigma_d = self.draws_Sigma[d]   # (n × n)
            try:
                chol_S = np.linalg.cholesky(Sigma_d + np.eye(n) * 1e-8)
            except np.linalg.LinAlgError:
                chol_S = np.diag(np.sqrt(np.maximum(np.diag(Sigma_d), 1e-8)))

            # Bygg initial lagbuffer
            lag_buf = last_obs.copy()  # nyeste siste

            for t in range(h):
                # Regressor: [1, y_{t-1}, ..., y_{t-p}]
                x_t = np.concatenate([[1.0],
                                       lag_buf[::-1].flatten()])
                # Forventning
                y_hat = B_d.T @ x_t
                # Trekk sjokk
                eps   = chol_S @ np.random.randn(n)
                y_new = y_hat + eps
                all_paths[d, t, :] = y_new

                # Oppdater lagbuffer
                lag_buf = np.roll(lag_buf, 1, axis=0)
                lag_buf[0] = y_new

        # Aggreger over draws
        results = {}
        for vi, var in enumerate(self.variables):
            paths_v = all_paths[:, :, vi]   # (n_draws × h)
            results[var] = {
                "mean": paths_v.mean(axis=0).tolist(),
                "bands": {
                    str(int(q * 100)): np.quantile(paths_v, q, axis=0).tolist()
                    for q in quantiles
                },
            }
        return results

    def fitted_values(self) -> pd.DataFrame:
        """Returnerer tilpassede verdier (posterior mean B)."""
        if not self.fitted:
            raise RuntimeError(f"{self.name}: kall fit() først.")
        fitted = self.X_full @ self.B_post_mean
        idx = self.Y_dep_full   # bare for å få riktig lengde
        return pd.DataFrame(fitted, columns=self.variables)


# ═══════════════════════════════════════════════════════════════════════════════
# UNIVARIAT AR-MODELL (eksogene drivere)
# ═══════════════════════════════════════════════════════════════════════════════

class ARModel:
    """
    Univariat AR(p) med OLS-estimering og analytiske prediksjonsintervaller.
    Brukes for oljepris og handelspartner-BNP.
    """

    def __init__(self,
                 name: str,
                 variable: str,
                 p_max: int = 4):
        self.name     = name
        self.variable = variable
        self.p_max    = p_max
        self.p:       int = p_max
        self.coef:    Optional[np.ndarray] = None
        self.sigma2:  float = 1.0
        self.fitted:  bool = False

    def fit(self,
            data: pd.DataFrame,
            covid_mask: Optional[np.ndarray] = None) -> "ARModel":
        valid  = data[self.variable].notna().values
        y_full = data[self.variable].dropna().values.astype(float)

        # Trim covid_mask to match y_full (removes rows dropped by dropna)
        if covid_mask is not None and len(covid_mask) == len(valid):
            cm = covid_mask[valid]
        else:
            cm = covid_mask

        # BIC-lagvalg
        best_p, best_bic = 1, np.inf
        for p in range(1, self.p_max + 1):
            X, Y_d = _lag_matrix(y_full.reshape(-1, 1), p)
            if cm is not None:
                m = cm[p:]
                X_f = X[~m]; Y_f = Y_d[~m]
            else:
                X_f, Y_f = X, Y_d
            if len(X_f) < p + 2:
                continue
            b = np.linalg.lstsq(X_f, Y_f, rcond=None)[0]
            resid = Y_f - X_f @ b
            s2 = float(np.squeeze(resid.T @ resid)) / len(resid)
            k = p + 1
            bic = np.log(s2) + k * np.log(len(X_f)) / len(X_f)
            if bic < best_bic:
                best_bic = bic
                best_p   = p

        self.p = best_p
        X, Y_d = _lag_matrix(y_full.reshape(-1, 1), self.p)
        if cm is not None:
            m = cm[self.p:]
            X_f = X[~m]; Y_f = Y_d[~m]
        else:
            X_f, Y_f = X, Y_d
        self.coef = np.linalg.lstsq(X_f, Y_f, rcond=None)[0].flatten()
        resid     = Y_f - X_f @ self.coef.reshape(-1, 1)
        self.sigma2   = float(np.squeeze(resid.T @ resid)) / len(resid)
        self.y_full   = y_full
        self.fitted   = True
        log.info(f"  {self.name} ({self.variable}): AR({self.p}), "
                 f"sigma={np.sqrt(self.sigma2):.4f}")
        return self

    def forecast(self,
                 h: int = 12,
                 n_sim: int = 2000
                 ) -> Dict:
        """
        Monte Carlo-simulering av h-stegs prediksjon.
        Returnerer mean og 5/25/50/75/95-prosentiler.
        """
        if not self.fitted:
            raise RuntimeError(f"{self.name}: kall fit() først.")
        p = self.p
        last_y = self.y_full[-p:].copy()
        paths  = np.zeros((n_sim, h))

        for s in range(n_sim):
            buf = last_y.copy()
            for t in range(h):
                x_t  = np.concatenate([[1.0], buf[::-1]])
                y_hat = float(self.coef @ x_t)
                eps   = np.random.randn() * np.sqrt(self.sigma2)
                y_new = y_hat + eps
                paths[s, t] = y_new
                buf = np.roll(buf, 1)
                buf[0] = y_new

        quantiles = (0.05, 0.25, 0.50, 0.75, 0.95)
        return {
            self.variable: {
                "mean": paths.mean(axis=0).tolist(),
                "bands": {
                    str(int(q * 100)): np.quantile(paths, q, axis=0).tolist()
                    for q in quantiles
                },
            }
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MÅNEDLIG ARIMA (nowcast-supplement for KPI og styringsrente)
# ═══════════════════════════════════════════════════════════════════════════════

class MonthlyARIMA:
    """
    Månedlig ARIMA-modell for nowcast av inneværende kvartal.

    Bruker statsmodels SARIMAX med automatisk ordensvalg via AIC
    (prøver ARIMA(p,1,q) for p,q i {0,1,2}).

    Nowcast-logikk:
        Gitt at vi er i måned m av inneværende kvartal og har observert
        1 eller 2 måneder, beregnes forventet kvartalsgjennomsnitt ved å
        fremskrive de gjenværende månedene og ta snitt.
    """

    def __init__(self,
                 name: str,
                 variable_monthly: str,
                 variable_quarterly: str):
        self.name               = name
        self.variable_monthly   = variable_monthly    # kolonnenavn i rådata
        self.variable_quarterly = variable_quarterly  # kolonnenavn i crosscheck_data
        self.model_fit          = None
        self.fitted             = False
        # KPI-nowcast må konvertere fra indeksnivå til annualisert Δlog
        # (samme enhet som pi_obs i crosscheck_data.csv: Δlog × 4 × 100).
        # Rente-nowcast beholder nivå direkte (% p.a. matcher i_R_obs-enheten).
        self.transform_to_pi      = (variable_quarterly == "pi_obs")
        self.q_full_levels: Optional[pd.Series] = None
        self.last_full_quarter_level: Optional[float] = None

    def fit(self, monthly_series: pd.Series) -> "MonthlyARIMA":
        """Estimer ARIMA på månedlig serie."""
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ImportError:
            log.warning(f"  {self.name}: statsmodels ikke installert — hopper over.")
            return self

        s = monthly_series.dropna().astype(float)
        if len(s) < 24:
            log.warning(f"  {self.name}: for kort serie ({len(s)} mnd)")
            return self

        best_aic = np.inf
        best_fit = None
        # Automatisk ordensvalg: ARIMA(p,1,q)
        for p in range(3):
            for q in range(3):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m = SARIMAX(s, order=(p, 1, q),
                                    trend="c",
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
                        fit = m.fit(disp=False, maxiter=200)
                    if fit.aic < best_aic:
                        best_aic = fit.aic
                        best_fit = fit
                        best_order = (p, 1, q)
                except Exception:
                    continue

        if best_fit is None:
            log.warning(f"  {self.name}: ARIMA-estimering feilet.")
            return self

        self.model_fit = best_fit
        self.fitted    = True
        self.last_obs  = s

        # Lagre fullstendige kvartalsnivåer (bare kvartal med 3 observerte
        # måneder) for nowcast-transformasjon. Partielle kvartaler ekskluderes
        # så referansenivået aldri peker på et halvferdig kvartal.
        try:
            s_dt = s.copy()
            s_dt.index = pd.to_datetime(s.index)
            q_agg  = s_dt.resample("QE").agg(["mean", "count"])
            q_full = q_agg.loc[q_agg["count"] == 3, "mean"]
            q_full.index = q_full.index.to_period("Q")
            self.q_full_levels = q_full
            self.last_full_quarter_level = (
                float(q_full.iloc[-1]) if len(q_full) else None
            )
        except Exception as e:
            log.warning(f"  {self.name}: kunne ikke beregne kvartalsnivåer: {e}")
            self.q_full_levels = None
            self.last_full_quarter_level = None

        log.info(f"  {self.name}: ARIMA{best_order}, AIC={best_aic:.1f}, "
                 f"T={len(s)} mnd")
        return self

    def _level_to_pi(self, level_q: float, prev_level: Optional[float]) -> float:
        """Δlog(KPI_q / KPI_{q-1}) × 4 × 100 — samme transform som pi_obs."""
        if (prev_level is None or prev_level <= 0
                or level_q is None or level_q <= 0
                or not np.isfinite(prev_level) or not np.isfinite(level_q)):
            return float("nan")
        return float(np.log(level_q / prev_level) * 4.0 * 100.0)

    def _prev_quarter_level(self) -> Optional[float]:
        """Siste fullstendige kvartalsnivå FØR inneværende kvartal."""
        if self.q_full_levels is None or len(self.q_full_levels) == 0:
            return None
        current_q = pd.Timestamp(self.last_obs.index[-1]).to_period("Q")
        # Finn siste fullstendige kvartal strengt før current_q
        mask = self.q_full_levels.index < current_q
        prev = self.q_full_levels[mask]
        if len(prev) > 0:
            return float(prev.iloc[-1])
        # Fallback: hvis current_q selv er fullstendig, bruk nest siste
        if len(self.q_full_levels) >= 2:
            return float(self.q_full_levels.iloc[-2])
        return None

    def nowcast_current_quarter(self,
                                 months_observed: int = 1) -> Dict:
        """
        Nowcast for inneværende kvartal.

        Parameters
        ----------
        months_observed : antall måneder observert i inneværende kvartal (1 eller 2)

        Returns
        -------
        dict med 'nowcast', 'ci_50_lo', 'ci_50_hi', 'ci_90_lo', 'ci_90_hi'
        """
        if not self.fitted:
            return {}

        def _pack(n, lo50, hi50, lo90, hi90) -> Dict:
            """Konverter nivå → pi_obs-enhet hvis transform_to_pi, ellers behold nivå."""
            if self.transform_to_pi:
                prev = self._prev_quarter_level()
                return {
                    "nowcast":  self._level_to_pi(n,    prev),
                    "ci_50_lo": self._level_to_pi(lo50, prev),
                    "ci_50_hi": self._level_to_pi(hi50, prev),
                    "ci_90_lo": self._level_to_pi(lo90, prev),
                    "ci_90_hi": self._level_to_pi(hi90, prev),
                    "months_observed": months_observed,
                }
            return {
                "nowcast":  float(n),
                "ci_50_lo": float(lo50), "ci_50_hi": float(hi50),
                "ci_90_lo": float(lo90), "ci_90_hi": float(hi90),
                "months_observed": months_observed,
            }

        months_remaining = 3 - months_observed
        if months_remaining == 0:
            # Alle måneder observert — bruk observert kvartalssnitt
            q_val = float(self.last_obs.iloc[-3:].mean())
            return _pack(q_val, q_val, q_val, q_val, q_val)
        try:
            fc = self.model_fit.get_forecast(months_remaining)
            fc_mean  = fc.predicted_mean.values
            fc_ci50  = fc.conf_int(alpha=0.50).values
            fc_ci90  = fc.conf_int(alpha=0.10).values
            obs_vals = self.last_obs.iloc[-months_observed:].values

            # Kvartalssnitt: snitt av observerte + fremskrivning
            all_vals = np.concatenate([obs_vals, fc_mean])
            ncast    = float(all_vals.mean())

            # Konfidensintervaller via delta-metode (forenkling)
            all_lo50 = np.concatenate([obs_vals, fc_ci50[:, 0]])
            all_hi50 = np.concatenate([obs_vals, fc_ci50[:, 1]])
            all_lo90 = np.concatenate([obs_vals, fc_ci90[:, 0]])
            all_hi90 = np.concatenate([obs_vals, fc_ci90[:, 1]])

            return _pack(
                ncast,
                float(all_lo50.mean()), float(all_hi50.mean()),
                float(all_lo90.mean()), float(all_hi90.mean()),
            )
        except Exception as e:
            log.warning(f"  {self.name} nowcast: {e}")
            return {}

    def forecast_quarterly(self,
                            h_quarters: int = 12) -> Dict:
        """
        Produser kvartalsvise prognoser ved å aggregere månedlige fremskrivninger.
        Komplementerer BVAR for de neste h_quarters kvartalene.
        """
        if not self.fitted:
            return {}
        h_months = h_quarters * 3
        try:
            fc  = self.model_fit.get_forecast(h_months)
            mn  = fc.predicted_mean.values
            ci50 = fc.conf_int(alpha=0.50).values
            ci90 = fc.conf_int(alpha=0.10).values

            # Aggreger måneder til kvartal (gjennomsnitt per 3 måneder)
            def agg_to_q(arr):
                n = len(arr) // 3
                return [float(arr[i*3:(i+1)*3].mean()) for i in range(n)]

            mean_levels = agg_to_q(mn)
            lo50_levels = agg_to_q(ci50[:, 0])
            hi50_levels = agg_to_q(ci50[:, 1])
            lo90_levels = agg_to_q(ci90[:, 0])
            hi90_levels = agg_to_q(ci90[:, 1])

            if self.transform_to_pi:
                # Kjed log-diff mot forrige kvartals nivå. Alle CI-grener deler
                # samme sentrale prev-bane for å unngå at kvantil-bånd drifter
                # urealistisk mye over horisonten.
                anchor = self._prev_quarter_level()

                def chain(levels: List[float]) -> List[float]:
                    out, prev = [], anchor
                    for lev in levels:
                        out.append(self._level_to_pi(lev, prev))
                        if lev is not None and lev > 0 and np.isfinite(lev):
                            prev = lev
                    return out

                return {
                    self.variable_quarterly: {
                        "mean":      chain(mean_levels),
                        "ci_50_lo":  chain(lo50_levels),
                        "ci_50_hi":  chain(hi50_levels),
                        "ci_90_lo":  chain(lo90_levels),
                        "ci_90_hi":  chain(hi90_levels),
                    }
                }

            return {
                self.variable_quarterly: {
                    "mean":      mean_levels,
                    "ci_50_lo":  lo50_levels,
                    "ci_50_hi":  hi50_levels,
                    "ci_90_lo":  lo90_levels,
                    "ci_90_hi":  hi90_levels,
                }
            }
        except Exception as e:
            log.warning(f"  {self.name} kvartalsprognose: {e}")
            return {}


# ═══════════════════════════════════════════════════════════════════════════════
# MODELLBIBLIOTEK — KOORDINERER ALLE MODELLER
# ═══════════════════════════════════════════════════════════════════════════════

class ModelLibrary:
    """
    Koordinerer estimering og fremskrivning for alle kryssjekkmodeller.

    Bruk:
        lib = ModelLibrary("crosscheck_data.csv")
        lib.fit_all()
        results = lib.forecast_all(h=12)
        lib.save(results, "crosscheck_results.json")
    """

    def __init__(self,
                 data_path: str,
                 raw_data_path: Optional[str] = None,
                 p: int = 4,
                 n_draws: int = 2000,
                 n_burnin: int = 1000):
        self.data_path     = data_path
        self.raw_data_path = raw_data_path
        self.p             = p
        self.n_draws       = n_draws
        self.n_burnin      = n_burnin

        # Last data
        self.data: pd.DataFrame = self._load_data(data_path)
        self.raw_data: Optional[pd.DataFrame] = (
            self._load_raw(raw_data_path) if raw_data_path else None
        )
        self.covid_mask: np.ndarray = self._build_covid_mask()

        # Definer modeller
        prior = MinnesotaPrior(lambda1=0.15, lambda2=0.50, lambda3=1.0)

        self.bvar1 = BVARMinnesota(
            name="BVAR-1 Realøkonomi",
            variables=BVAR1_VARS,
            p=p, prior=prior,
            n_draws=n_draws, n_burnin=n_burnin,
        )
        self.bvar2 = BVARMinnesota(
            name="BVAR-2 Priser/lønn",
            variables=BVAR2_VARS,
            p=p, prior=prior,
            n_draws=n_draws, n_burnin=n_burnin,
        )
        self.bvar3 = BVARMinnesota(
            name="BVAR-3 Finansielle/bolig",
            variables=BVAR3_VARS,
            p=p, prior=prior,
            n_draws=n_draws, n_burnin=n_burnin,
        )
        self.ar_olje = ARModel(
            name="AR Oljepris",
            variable="dpO_obs",
            p_max=4,
        )
        self.ar_partner = ARModel(
            name="AR Handelspartner",
            variable="dyS_obs",
            p_max=4,
        )
        self.arima_kpi = MonthlyARIMA(
            name="ARIMA KPI månedlig",
            variable_monthly=MONTHLY_KPI_COL,
            variable_quarterly="pi_obs",
        )
        self.arima_rente = MonthlyARIMA(
            name="ARIMA Rente månedlig",
            variable_monthly=MONTHLY_RATE_COL,
            variable_quarterly="i_R_obs",
        )

    def _load_data(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, index_col=0)
        # Konverter strengindeks til PeriodIndex
        try:
            df.index = pd.PeriodIndex(df.index, freq="Q")
        except Exception:
            pass
        # Behold bare OBS-kolonner (ikke _level, ikke covid_flag)
        obs_cols = [c for c in ALL_OBS if c in df.columns]
        missing  = [c for c in ALL_OBS if c not in df.columns]
        if missing:
            log.warning(f"  Kolonner mangler i data: {missing}")
        return df[obs_cols + (["covid_flag"] if "covid_flag" in df.columns else [])]

    def _load_raw(self, path: str) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            return df
        except Exception as e:
            log.warning(f"  Rådata ikke lastet: {e}")
            return None

    def _build_covid_mask(self) -> np.ndarray:
        if "covid_flag" in self.data.columns:
            return self.data["covid_flag"].fillna(0).astype(bool).values
        # Bygg fra datoer
        start = pd.Period(COVID_START, freq="Q")
        end   = pd.Period(COVID_END,   freq="Q")
        mask  = np.array(
            [(p >= start and p <= end) for p in self.data.index],
            dtype=bool,
        )
        return mask

    def _select_lag_bvar(self, variables: List[str]) -> int:
        """Velg lagorden for en BVAR via BIC."""
        cols = [c for c in variables if c in self.data.columns]
        if len(cols) < len(variables):
            return self.p
        Y = self.data[cols].dropna().values
        p_opt = _select_lag_bic(Y, p_max=self.p,
                                 exclude_mask=self.covid_mask[:len(Y)])
        return p_opt

    def fit_all(self) -> "ModelLibrary":
        """Estimer alle modeller."""
        log.info("\n" + "=" * 55)
        log.info("  ESTIMERER KRYSSJEKKMODELLER")
        log.info("=" * 55)

        # Bruk BIC for å velge lagorden per BVAR
        log.info("\n[1/5] BVAR-1 Realøkonomi...")
        p1 = _select_lag_bic(
            self.data[[c for c in BVAR1_VARS if c in self.data.columns]].values,
            p_max=self.p, exclude_mask=self.covid_mask,
        )
        log.info(f"  Lagorden BIC: p={p1}")
        self.bvar1.p = p1
        self.bvar1.fit(self.data, covid_mask=self.covid_mask)

        log.info("\n[2/5] BVAR-2 Priser/lønn...")
        p2 = _select_lag_bic(
            self.data[[c for c in BVAR2_VARS if c in self.data.columns]].values,
            p_max=self.p, exclude_mask=self.covid_mask,
        )
        log.info(f"  Lagorden BIC: p={p2}")
        self.bvar2.p = p2
        self.bvar2.fit(self.data, covid_mask=self.covid_mask)

        log.info("\n[3/5] BVAR-3 Finansielle/bolig...")
        p3 = _select_lag_bic(
            self.data[[c for c in BVAR3_VARS if c in self.data.columns]].values,
            p_max=self.p, exclude_mask=self.covid_mask,
        )
        log.info(f"  Lagorden BIC: p={p3}")
        self.bvar3.p = p3
        self.bvar3.fit(self.data, covid_mask=self.covid_mask)

        log.info("\n[4/5] AR-modeller (eksogene drivere)...")
        self.ar_olje.fit(self.data, covid_mask=self.covid_mask)
        self.ar_partner.fit(self.data, covid_mask=self.covid_mask)

        log.info("\n[5/5] Månedlige ARIMA-modeller (nowcast)...")
        if self.raw_data is not None:
            if MONTHLY_KPI_COL in self.raw_data.columns:
                self.arima_kpi.fit(self.raw_data[MONTHLY_KPI_COL].dropna())
            else:
                log.warning(f"  KPI månedlig ikke i rådata.")
            if MONTHLY_RATE_COL in self.raw_data.columns:
                self.arima_rente.fit(self.raw_data[MONTHLY_RATE_COL].dropna())
            else:
                log.warning(f"  Styringsrente månedlig ikke i rådata.")
        else:
            log.warning("  Ingen rådata angitt — ARIMA-modeller hoppes over.")

        log.info("\n  Estimering ferdig.")
        return self

    def forecast_all(self,
                     h: int = 12,
                     months_observed_current_q: int = 1) -> Dict:
        """
        Fremskrivning h kvartaler for alle modeller.

        Parameters
        ----------
        h                          : prognosehorisonten i kvartaler
        months_observed_current_q  : måneder observert i inneværende kvartal
                                     (for nowcast-justeringen)
        """
        log.info(f"\n  Beregner fremskrivninger h={h} kvartaler...")
        results = {"meta": {
            "h": h,
            "n_draws": self.n_draws,
            "n_burnin": self.n_burnin,
            "end_period": str(self.data.index[-1]),
        }}

        # BVAR-prognoser
        results["bvar1"] = self.bvar1.forecast(h=h) if self.bvar1.fitted else {}
        results["bvar2"] = self.bvar2.forecast(h=h) if self.bvar2.fitted else {}
        results["bvar3"] = self.bvar3.forecast(h=h) if self.bvar3.fitted else {}

        # AR-prognoser
        results["ar_olje"]    = self.ar_olje.forecast(h=h)    if self.ar_olje.fitted else {}
        results["ar_partner"] = self.ar_partner.forecast(h=h) if self.ar_partner.fitted else {}

        # Månedlig nowcast
        results["nowcast"] = {}
        if self.arima_kpi.fitted:
            results["nowcast"]["pi_obs"] = self.arima_kpi.nowcast_current_quarter(
                months_observed=months_observed_current_q
            )
            results["nowcast"]["pi_obs_quarterly"] = self.arima_kpi.forecast_quarterly(
                h_quarters=h
            ).get("pi_obs", {})

        if self.arima_rente.fitted:
            results["nowcast"]["i_R_obs"] = self.arima_rente.nowcast_current_quarter(
                months_observed=months_observed_current_q
            )

        # Tilpassede verdier (for historisk sammenligning mot DSGE)
        results["fitted"] = {}
        for model, name in [
            (self.bvar1, "bvar1"),
            (self.bvar2, "bvar2"),
            (self.bvar3, "bvar3"),
        ]:
            if model.fitted:
                fv = model.fitted_values()
                results["fitted"][name] = {
                    col: fv[col].tolist() for col in fv.columns
                }

        log.info("  Fremskrivning ferdig.")
        return results

    def save(self, results: Dict, path: str) -> None:
        """Lagre resultater til JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False,
                      default=lambda x: float(x) if isinstance(x, np.floating) else x)
        log.info(f"  Lagret: {path}")

    def model_summary(self) -> str:
        """Skriv ut oppsummering av estimerte modeller."""
        lines = ["\n  MODELLOVERSIKT", "  " + "─" * 50]
        for model in [self.bvar1, self.bvar2, self.bvar3]:
            status = "estimert" if model.fitted else "ikke estimert"
            lines.append(f"  {model.name:<35} p={model.p}  [{status}]")
            if model.fitted:
                lines.append(f"    Variabler: {', '.join(model.variables)}")
        for model in [self.ar_olje, self.ar_partner]:
            status = "estimert" if model.fitted else "ikke estimert"
            lines.append(f"  {model.name:<35} p={model.p}  [{status}]")
        for model in [self.arima_kpi, self.arima_rente]:
            status = "estimert" if model.fitted else "ikke estimert"
            lines.append(f"  {model.name:<35} [{status}]")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# KJØRING
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NEMO Fase III — Modellbibliotek for kryssjekkmodeller"
    )
    parser.add_argument("--data",       default="crosscheck_data.csv")
    parser.add_argument("--raw-data",   default="crosscheck_data_raw.csv")
    parser.add_argument("--horizon",    type=int, default=12)
    parser.add_argument("--draws",      type=int, default=2000)
    parser.add_argument("--burnin",     type=int, default=1000)
    parser.add_argument("--lag",        type=int, default=4)
    parser.add_argument("--output",     default="crosscheck_results.json")
    parser.add_argument("--months-obs", type=int, default=1,
                        help="Måneder observert i inneværende kvartal (nowcast)")
    args = parser.parse_args()

    lib = ModelLibrary(
        data_path=args.data,
        raw_data_path=args.raw_data if os.path.exists(args.raw_data) else None,
        p=args.lag,
        n_draws=args.draws,
        n_burnin=args.burnin,
    )
    lib.fit_all()
    print(lib.model_summary())

    results = lib.forecast_all(
        h=args.horizon,
        months_observed_current_q=args.months_obs,
    )
    lib.save(results, args.output)

    print(f"\n  Output: {args.output}")
    print(f"  Periode:   {results['meta']['end_period']} + {args.horizon} kvartaler")
