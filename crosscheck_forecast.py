"""
================================================================================
NEMO FASE III — FORECAST-AGGREGERING OG ENSEMBLE
Milepål 4 (forecast og usikkerhet) og Milepål 5 (ensemble)

Tar output fra crosscheck_models.py og produserer:
  - Ensemble-prognose per variabel (vektet kombinasjon av modeller)
  - 50 % og 90 % prediksjonsintervaller
  - Nowcast-justert startverdi for KPI og styringsrente
  - Historisk tilpasning (fitted values) for alle variabler
  - Ferdig JSON-struktur klar for dashboard og DSGE-sammenligning

Ensemble-metodikk:
  Lik vekting som default (equal-weighted combination).
  Støtter også RMSE-invers vekting basert på historisk treningsperiode.
  Litteratur: Timmermann (2006), Genre et al. (2013).

Bruk:
    python crosscheck_forecast.py \\
        --results crosscheck_results.json \\
        --data    crosscheck_data.csv \\
        --output  crosscheck_ensemble.json

    Eller fra Python:
        from crosscheck_forecast import EnsembleForecaster
        ens = EnsembleForecaster("crosscheck_results.json", "crosscheck_data.csv")
        output = ens.run(h=12)
        ens.save(output, "crosscheck_ensemble.json")

Avhengigheter:
    pip install pandas numpy scipy
================================================================================
"""

import argparse
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("crosscheck_forecast")

# ── Variabelkart ─────────────────────────────────────────────────────────────────────────────
VAR_TO_MODELS = {
    "dy_obs":   ["bvar1"],
    "dc_obs":   ["bvar1"],
    "dinv_obs": ["bvar1"],
    "dx_obs":   ["bvar1"],
    "dm_obs":   ["bvar1"],
    "pi_obs":   ["bvar2"],
    "dw_obs":   ["bvar2"],
    "dpO_obs":  ["bvar2", "ar_olje"],
    "dyS_obs":  ["bvar2", "ar_partner"],
    "i_R_obs":  ["bvar3"],
    "i_3m_obs": ["bvar3"],
    "ds_obs":   ["bvar3"],
    "dh_obs":   ["bvar3"],
    "db_obs":   ["bvar3"],
}

VAR_LABELS = {
    "dy_obs":   "BNP fastland",
    "dc_obs":   "Privat konsum",
    "dinv_obs": "Investering",
    "dx_obs":   "Eksport",
    "dm_obs":   "Import",
    "pi_obs":   "KPI-inflasjon",
    "dw_obs":   "Lønnsvekst",
    "dpO_obs":  "Oljepris",
    "dyS_obs":  "Handelspartner-BNP",
    "i_R_obs":  "Styringsrente",
    "i_3m_obs": "NIBOR 3M",
    "ds_obs":   "NOK/EUR",
    "dh_obs":   "Boligpris",
    "db_obs":   "Kredittvekst",
}

VAR_UNITS = {
    "dy_obs":   "pst. kv.vekst",
    "dc_obs":   "pst. kv.vekst",
    "dinv_obs": "pst. kv.vekst",
    "dx_obs":   "pst. kv.vekst",
    "dm_obs":   "pst. kv.vekst",
    "pi_obs":   "pst. årsrate",
    "dw_obs":   "pst. årsrate",
    "dpO_obs":  "pst. kv.vekst",
    "dyS_obs":  "gap, pst.",
    "i_R_obs":  "pst. p.a. / 4",
    "i_3m_obs": "pst. p.a. / 4",
    "ds_obs":   "pst. kv.vekst",
    "dh_obs":   "pst. kv.vekst",
    "db_obs":   "pst. vekst",
}

COVID_START = "2020Q1"
COVID_END   = "2021Q4"


# ═══════════════════════════════════════════════════════════════════════════════
HJELPEFUNKSJONER
═══════════════════════════════════════════════════════════════════════════════

def _normal_quantile(mean: np.ndarray,
                     std: np.ndarray,
                     q: float) -> np.ndarray:
    from scipy.stats import norm
    return mean + norm.ppf(q) * std


def _combine_forecast_moments(means: List[np.ndarray],
                               variances: List[np.ndarray],
                               weights: np.ndarray
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Kombiner prognosemomenter fra flere modeller.

    Law of total variance:
        E[y] = Σ_m w_m * μ_m
        Var[y] = Σ_m w_m * (σ_m² + μ_m²) - E[y]²
               = Σ_m w_m * σ_m²  +  Σ_m w_m * (μ_m - E[y])²

    Andre ledd er modell-uenighets-variansen (mellom-modell spredning).
    Dette gir bredere intervaller når modellene er uenige.
    """
    w = weights / weights.sum()
    combined_mean = sum(w[i] * means[i] for i in range(len(means)))
    within_var    = sum(w[i] * variances[i] for i in range(len(means)))
    between_var   = sum(
        w[i] * (means[i] - combined_mean) ** 2
        for i in range(len(means))
    )
    combined_var = within_var + between_var
    return combined_mean, combined_var


def _extract_model_forecast(results: Dict,
                              model_key: str,
                              variable: str,
                              h: int
                              ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    model_fc = results.get(model_key, {})
    var_fc   = model_fc.get(variable)
    if var_fc is None:
        return None

    mean = np.array(var_fc["mean"][:h], dtype=float)
    bands = var_fc.get("bands", {})

    if "5" in bands and "95" in bands:
        lo = np.array(bands["5"][:h], dtype=float)
        hi = np.array(bands["95"][:h], dtype=float)
        std = (hi - lo) / (2 * 1.6449)
    elif "25" in bands and "75" in bands:
        lo = np.array(bands["25"][:h], dtype=float)
        hi = np.array(bands["75"][:h], dtype=float)
        std = (hi - lo) / (2 * 0.6745)
    else:
        std = np.ones(h) * 0.5

    variance = np.maximum(std ** 2, 1e-10)
    return mean, variance


def _rmse_weights(fitted: Dict,
                  data: pd.DataFrame,
                  variables: List[str],
                  model_keys: List[str],
                  covid_mask: np.ndarray,
                  p: int = 4
                  ) -> Dict[str, np.ndarray]:
    """
    Beregn RMSE-invers vekter for ensemble per variabel.
    Fallback til lik vekting hvis fitted values mangler eller
    RMSE-differansen er liten (< 5 % relativ forskjell).
    """
    weights = {}
    for var in variables:
        model_list = VAR_TO_MODELS.get(var, [])
        available  = [m for m in model_list if m in model_keys]
        if len(available) <= 1:
            weights[var] = np.ones(len(available))
            continue

        rmse_vals = []
        obs = data[var].values.astype(float) if var in data.columns else None

        for m_key in available:
            fv = fitted.get(m_key, {}).get(var)
            if fv is None or obs is None:
                rmse_vals.append(np.inf)
                continue
            fv_arr = np.array(fv, dtype=float)
            n_fit  = len(fv_arr)
            obs_tr = obs[-n_fit:]
            mask   = covid_mask[-n_fit:]
            n      = min(len(obs_tr), len(fv_arr), len(mask))
            obs_tr = obs_tr[:n]; mask = mask[:n]; fv_arr = fv_arr[:n]
            valid  = ~mask & ~np.isnan(obs_tr) & ~np.isnan(fv_arr)
            if valid.sum() < 4:
                rmse_vals.append(np.inf)
                continue
            rmse_vals.append(float(np.sqrt(np.mean((obs_tr[valid] - fv_arr[valid]) ** 2))))

        rmse_arr = np.array(rmse_vals)
        if np.any(np.isinf(rmse_arr)) or rmse_arr.min() < 1e-12:
            weights[var] = np.ones(len(available))
        else:
            rel_diff = (rmse_arr.max() - rmse_arr.min()) / rmse_arr.mean()
            if rel_diff < 0.05:
                weights[var] = np.ones(len(available))
            else:
                inv_rmse = 1.0 / rmse_arr
                weights[var] = inv_rmse / inv_rmse.sum()

    return weights


# ═══════════════════════════════════════════════════════════════════════════════
NOWCAST-JUSTERING
═══════════════════════════════════════════════════════════════════════════════

def apply_nowcast(ensemble_mean: np.ndarray,
                  ensemble_var: np.ndarray,
                  nowcast: Dict,
                  variable: str,
                  h: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Juster første horisont (h=1) med nowcast-estimat.
    Blander BVAR-prognose og nowcast med vekt alpha = months_observed / 3.
    """
    ncast_info = nowcast.get(variable, {})
    if not ncast_info or "nowcast" not in ncast_info:
        return ensemble_mean, ensemble_var

    months_obs = ncast_info.get("months_observed", 1)
    alpha = months_obs / 3.0
    ncast_val = float(ncast_info["nowcast"])

    ci_lo = ncast_info.get("ci_90_lo", ncast_val - 0.5)
    ci_hi = ncast_info.get("ci_90_hi", ncast_val + 0.5)
    ncast_std = max((ci_hi - ci_lo) / (2 * 1.6449), 1e-4)
    ncast_var  = ncast_std ** 2

    adj_mean = ensemble_mean.copy()
    adj_var  = ensemble_var.copy()
    adj_mean[0] = (1 - alpha) * ensemble_mean[0] + alpha * ncast_val
    adj_var[0]  = (1 - alpha) ** 2 * ensemble_var[0] + alpha ** 2 * ncast_var

    return adj_mean, adj_var


# ═══════════════════════════════════════════════════════════════════════════════
HISTORISK TILPASNING
═══════════════════════════════════════════════════════════════════════════════

def build_historical_fit(fitted: Dict,
                          data: pd.DataFrame,
                          variables: List[str],
                          covid_mask: np.ndarray,
                          p: int = 4) -> Dict:
    """
    Bygg historisk tilpasning per variabel.
    For variabler med flere modeller: vektet snitt av fitted values.
    """
    out = {}
    for var in variables:
        model_list = VAR_TO_MODELS.get(var, [])
        obs = data[var].values.astype(float) if var in data.columns else None
        if obs is None:
            continue

        fv_list = []
        for m_key in model_list:
            fv = fitted.get(m_key, {}).get(var)
            if fv is not None:
                fv_list.append(np.array(fv, dtype=float))

        if not fv_list:
            continue

        fv_mean = np.mean(fv_list, axis=0)
        n_fit   = len(fv_mean)

        # Aligner fra slutten for konsistens med variabel lagorden
        obs_aligned   = obs[-n_fit:]
        covid_aligned = covid_mask[-n_fit:]

        resid = obs_aligned - fv_mean
        resid_no_covid = resid.copy()
        resid_no_covid[covid_aligned] = np.nan

        valid = ~covid_aligned & ~np.isnan(obs_aligned) & ~np.isnan(fv_mean)
        rmse  = float(np.sqrt(np.mean(resid[valid] ** 2))) if valid.sum() > 0 else np.nan

        out[var] = {
            "observed":      [float(x) if not np.isnan(x) else None
                              for x in obs_aligned],
            "fitted":        [float(x) if not np.isnan(x) else None
                              for x in fv_mean],
            "residual":      [float(x) if not np.isnan(x) else None
                              for x in resid_no_covid],
            "rmse":          round(rmse, 5) if not np.isnan(rmse) else None,
            "covid_periods": covid_aligned.tolist(),
            "periods":       [str(idx) for idx in data.index[-n_fit:]],
        }
    return out


# ═══════════════════════════════════════════════════════════════════════════════
ENSEMBLE FORECASTER
═══════════════════════════════════════════════════════════════════════════════

class EnsembleForecaster:
    """
    Koordinerer ensemble-aggregering av alle kryssjekkmodellers prognoser.

    Workflow:
        1. Last modellresultater fra crosscheck_results.json
        2. Beregn RMSE-invers vekter per variabel
        3. Kombiner prognosemomenter (mean + total variance)
        4. Juster h=1 med nowcast for KPI og styringsrente
        5. Produser 50 % og 90 % intervaller
        6. Bygg historisk tilpasning
        7. Generer ferdig output-struktur
    """

    def __init__(self,
                 results_path: str,
                 data_path: str,
                 weighting: str = "rmse",
                 p: int = 4):
        self.weighting = weighting
        self.p         = p

        log.info("Laster modellresultater og data...")
        with open(results_path, encoding="utf-8") as f:
            self.results = json.load(f)

        self.data = pd.read_csv(data_path, index_col=0)
        try:
            self.data.index = pd.PeriodIndex(self.data.index, freq="Q")
        except Exception:
            pass

        if "covid_flag" in self.data.columns:
            self.covid_mask = self.data["covid_flag"].fillna(0).astype(bool).values
        else:
            start = pd.Period(COVID_START, freq="Q")
            end   = pd.Period(COVID_END,   freq="Q")
            self.covid_mask = np.array(
                [(p >= start and p <= end) for p in self.data.index], dtype=bool
            )

        self.variables  = list(VAR_TO_MODELS.keys())
        self.h_meta     = int(self.results.get("meta", {}).get("h", 12))
        self.end_period = self.results.get("meta", {}).get("end_period", "")
        log.info(f"  Slutt-periode: {self.end_period}, h={self.h_meta}")

    def _get_forecast_periods(self, h: int) -> List[str]:
        if not self.end_period:
            return [f"h+{i+1}" for i in range(h)]
        try:
            last = pd.Period(self.end_period, freq="Q")
            return [str(last + i + 1) for i in range(h)]
        except Exception:
            return [f"h+{i+1}" for i in range(h)]

    def run(self, h: Optional[int] = None) -> Dict:
        """
        Kjør komplett ensemble-aggregering.

        Returns
        -------
        Dict med struktur:
            forecasts   : per variabel — mean, ci_50, ci_90, periods
            fitted      : historisk tilpasning per variabel
            weights     : ensemble-vekter per variabel
            nowcast     : nowcast-info for KPI og rente
            meta        : kjøremetadata
        """
        h = h or self.h_meta
        log.info(f"\n  Ensemble-aggregering: h={h} kvartaler, vekting={self.weighting}")

        fitted_raw  = self.results.get("fitted", {})
        nowcast_raw = self.results.get("nowcast", {})
        periods     = self._get_forecast_periods(h)

        all_model_keys = ["bvar1", "bvar2", "bvar3", "ar_olje", "ar_partner"]
        available_keys = [k for k in all_model_keys if k in self.results]

        if self.weighting == "rmse":
            weights_by_var = _rmse_weights(
                fitted_raw, self.data, self.variables,
                available_keys, self.covid_mask, self.p
            )
        else:
            weights_by_var = {
                var: np.ones(len([m for m in VAR_TO_MODELS.get(var, [])
                                  if m in available_keys]))
                for var in self.variables
            }

        forecasts   = {}
        weights_out = {}

        for var in self.variables:
            model_list = [m for m in VAR_TO_MODELS.get(var, [])
                          if m in available_keys]
            if not model_list:
                log.warning(f"  {var}: ingen tilgjengelige modeller")
                continue

            moments = []
            for m_key in model_list:
                res = _extract_model_forecast(self.results, m_key, var, h)
                if res is not None:
                    moments.append(res)

            if not moments:
                log.warning(f"  {var}: ingen forecast-data funnet")
                continue

            w_arr = weights_by_var.get(var, np.ones(len(moments)))
            w_arr = w_arr[:len(moments)]
            w_arr = w_arr / w_arr.sum()

            means_list = [m[0] for m in moments]
            vars_list  = [m[1] for m in moments]

            ens_mean, ens_var = _combine_forecast_moments(
                means_list, vars_list, w_arr
            )

            if var in ("pi_obs", "i_R_obs"):
                ens_mean, ens_var = apply_nowcast(
                    ens_mean, ens_var, nowcast_raw, var, h
                )

            ens_std = np.sqrt(np.maximum(ens_var, 1e-10))

            ci_50_lo = _normal_quantile(ens_mean, ens_std, 0.25)
            ci_50_hi = _normal_quantile(ens_mean, ens_std, 0.75)
            ci_90_lo = _normal_quantile(ens_mean, ens_std, 0.05)
            ci_90_hi = _normal_quantile(ens_mean, ens_std, 0.95)

            forecasts[var] = {
                "label":   VAR_LABELS.get(var, var),
                "unit":    VAR_UNITS.get(var, ""),
                "periods": periods[:h],
                "mean":    ens_mean.tolist(),
                "ci_50":   {"lo": ci_50_lo.tolist(), "hi": ci_50_hi.tolist()},
                "ci_90":   {"lo": ci_90_lo.tolist(), "hi": ci_90_hi.tolist()},
                "std":     ens_std.tolist(),
                "models_used": model_list,
            }

            model_detail = {}
            for i, m_key in enumerate(model_list):
                if i < len(moments):
                    model_detail[m_key] = {
                        "mean":   moments[i][0].tolist(),
                        "weight": float(w_arr[i]),
                    }
            forecasts[var]["model_detail"] = model_detail
            weights_out[var] = {m: float(w_arr[i])
                                for i, m in enumerate(model_list)}

        log.info(f"  Aggregert {len(forecasts)}/{len(self.variables)} variabler.")

        log.info("  Bygger historisk tilpasning...")
        historical = build_historical_fit(
            fitted_raw, self.data, self.variables,
            self.covid_mask, self.p
        )

        eval_summary = self._evaluation_summary(historical)

        output = {
            "meta": {
                "end_period":       self.end_period,
                "h":                h,
                "forecast_periods": periods,
                "weighting":        self.weighting,
                "variables":        self.variables,
                "n_variables":      len(forecasts),
                "nowcast_applied":  list(nowcast_raw.keys()),
            },
            "forecasts":  forecasts,
            "historical": historical,
            "weights":    weights_out,
            "nowcast":    nowcast_raw,
            "evaluation": eval_summary,
        }

        log.info("  Ensemble ferdig.")
        return output

    def _evaluation_summary(self, historical: Dict) -> Dict:
        summary = {}
        for var, hdata in historical.items():
            obs   = np.array([x for x in hdata["observed"] if x is not None],
                             dtype=float)
            fv    = np.array([x for x in hdata["fitted"]   if x is not None],
                             dtype=float)
            covid = np.array(hdata["covid_periods"])
            n     = min(len(obs), len(fv), len(covid))
            obs, fv, covid = obs[:n], fv[:n], covid[:n]

            valid = ~covid & ~np.isnan(obs) & ~np.isnan(fv)
            if valid.sum() < 4:
                continue
            resid  = obs[valid] - fv[valid]
            rmse   = float(np.sqrt(np.mean(resid ** 2)))
            mae    = float(np.mean(np.abs(resid)))
            ss_tot = float(np.sum((obs[valid] - obs[valid].mean()) ** 2))
            ss_res = float(np.sum(resid ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan

            summary[var] = {
                "label": VAR_LABELS.get(var, var),
                "rmse":  round(rmse, 5),
                "mae":   round(mae, 5),
                "r2":    round(r2, 4) if not np.isnan(r2) else None,
                "n_obs": int(valid.sum()),
            }
        return summary

    def save(self, output: Dict, path: str) -> None:
        def _serialize(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, bool):
                return obj
            raise TypeError(f"Ikke serialiserbar: {type(obj)}")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False,
                      default=_serialize)
        log.info(f"  Lagret: {path}")

    def print_summary(self, output: Dict) -> None:
        print("\n" + "=" * 65)
        print("  ENSEMBLE FORECAST — OPPSUMMERING")
        print("=" * 65)
        print(f"  Slutt-periode  : {output['meta']['end_period']}")
        print(f"  Horisont       : {output['meta']['h']} kvartaler")
        print(f"  Vekting        : {output['meta']['weighting']}")
        print(f"  Variabler      : {output['meta']['n_variables']}")

        print("\n  PASNING (RMSE, ex. COVID):")
        print(f"  {'Variabel':<22} {'RMSE':>8}  {'MAE':>8}  {'R\u00b2':>7}  {'N':>5}")
        print("  " + "─" * 55)
        for var, ev in sorted(output["evaluation"].items(),
                               key=lambda x: x[1].get("rmse", 99)):
            r2_str = f"{ev['r2']:.3f}" if ev["r2"] is not None else "  n/a"
            print(f"  {ev['label']:<22} {ev['rmse']:>8.4f}  "
                  f"{ev['mae']:>8.4f}  {r2_str:>7}  {ev['n_obs']:>5}")

        print("\n  PROGNOSE — FØRSTE OG FJERDE KVARTAL:")
        print(f"  {'Variabel':<22} {'h=1 (mean)':>12}  "
              f"{'CI90 lo':>10}  {'CI90 hi':>10}  {'h=4 (mean)':>12}")
        print("  " + "─" * 72)
        for var, fc in output["forecasts"].items():
            if not fc["mean"]:
                continue
            h1   = fc["mean"][0]
            lo1  = fc["ci_90"]["lo"][0]
            hi1  = fc["ci_90"]["hi"][0]
            h4   = fc["mean"][3] if len(fc["mean"]) >= 4 else float("nan")
            unit = fc["unit"]
            print(f"  {fc['label']:<22} {h1:>+12.3f}  "
                  f"{lo1:>+10.3f}  {hi1:>+10.3f}  {h4:>+12.3f}  {unit}")
        print("=" * 65)


# ═══════════════════════════════════════════════════════════════════════════════
KJØRING
═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NEMO Fase III — Ensemble forecast-aggregering"
    )
    parser.add_argument("--results",   default="crosscheck_results.json")
    parser.add_argument("--data",      default="crosscheck_data.csv")
    parser.add_argument("--output",    default="crosscheck_ensemble.json")
    parser.add_argument("--horizon",   type=int, default=12)
    parser.add_argument("--weighting", default="rmse",
                        choices=["equal", "rmse"])
    parser.add_argument("--lag",       type=int, default=4)
    args = parser.parse_args()

    ens = EnsembleForecaster(
        results_path=args.results,
        data_path=args.data,
        weighting=args.weighting,
        p=args.lag,
    )
    output = ens.run(h=args.horizon)
    ens.print_summary(output)
    ens.save(output, args.output)
    print(f"\n  Output: {args.output}")
