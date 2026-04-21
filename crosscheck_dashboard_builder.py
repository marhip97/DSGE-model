"""
================================================================================
NEMO — DASHBOARD BUILDER
Milepel 8 (DSGE-integrasjon) + samlet dashboard

Les analyse_resultater.json (Fase II) og crosscheck_ensemble.json (Fase III),
slå dem sammen til nemo_dashboard_data.json, og injiser i nemo_dashboard.html
slik at dashboardet fungerer uten webserver (file://-protokoll).

Bruk:
    python crosscheck_dashboard_builder.py \
        --dsge    analyse_resultater.json \
        --ck      crosscheck_ensemble.json \
        --template nemo_dashboard.html \
        --output  nemo_dashboard_built.html

    Eller bare slå sammen til JSON uten HTML-injeksjon:
    python crosscheck_dashboard_builder.py \
        --dsge    analyse_resultater.json \
        --ck      crosscheck_ensemble.json \
        --json-only

Avhengigheter: ingen (kun stdlib)
================================================================================
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, Optional

try:
    import requests
except ImportError:
    requests = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("dashboard_builder")


# ═══════════════════════════════════════════════════════════════════════════════
# FASE II — KONVERTERING AV DSGE-OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

# Mapping fra Fase II variabelnavn til Fase III obs-navn
DSGE_TO_OBS = {
    "y":    "dy_obs",
    "pi":   "pi_obs",
    "i":    "i_R_obs",
    "rer":  "ds_obs",
    "bolig":"dh_obs",
}

def load_dsge(path: str) -> Optional[Dict]:
    """Last og valider Fase II analyse_resultater.json."""
    if not os.path.exists(path):
        log.warning(f"  DSGE-fil ikke funnet: {path}")
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    required = ["irf", "forecast", "meta"]
    missing  = [k for k in required if k not in data]
    if missing:
        log.warning(f"  DSGE-data mangler nøkler: {missing}")
    log.info(f"  Fase II lastet: {path}")
    return data


def extract_dsge_forecast(dsge: Dict) -> Dict:
    """
    Trekk ut DSGE-fremskrivning i format som dashboardet forventer.

    Returnerer dict på formen:
        {obs_navn: {mean: [...], std: [...], periods: [...]}}
    """
    fc_raw = dsge.get("forecast", {})
    meta   = dsge.get("meta", {})

    # Conditional forecast (sjokk-betinget) foretrekkes over baseline
    cond = fc_raw.get("conditional", [])   # (h × NZ) som liste av lister
    base = fc_raw.get("baseline", [])
    std_p = fc_raw.get("std_P", {})

    # Forsøk å lese end_period fra meta
    end_period = meta.get("last_obs_date", "")
    if end_period:
        # Konverter "2025-09-30" → "2025Q3"
        try:
            from datetime import datetime as dt
            d = dt.strptime(end_period[:10], "%Y-%m-%d")
            q = (d.month - 1) // 3 + 1
            end_period = f"{d.year}Q{q}"
        except Exception:
            end_period = end_period[:7].replace("-", "Q")

    # Generer periode-labels fremover fra end_period
    def next_quarters(start_str: str, n: int):
        try:
            y, q = int(start_str[:4]), int(start_str[5])
            periods = []
            for _ in range(n):
                q += 1
                if q > 4:
                    q = 1; y += 1
                periods.append(f"{y}Q{q}")
            return periods
        except Exception:
            return [f"h+{i+1}" for i in range(n)]

    # DSGE-variabelindekser (fra equations.py i Fase II)
    # Disse er definerte konstanter i equations.py:
    # Y=0, PI=1, I_R=2, RER=3, Q_H=4
    VAR_IDX = {"y": 0, "pi": 1, "i": 2, "rer": 3, "bolig": 4}

    h = len(cond) if cond else len(base)
    periods = next_quarters(end_period, h)

    result = {}
    for dsge_name, obs_name in DSGE_TO_OBS.items():
        idx = VAR_IDX.get(dsge_name)
        if idx is None:
            continue

        # Hent mean fra conditional forecast
        if cond and len(cond) > 0:
            if isinstance(cond[0], list):
                mean = [float(cond[t][idx]) for t in range(h)
                        if t < len(cond) and idx < len(cond[t])]
            else:
                mean = [float(cond[t]) for t in range(h) if t < len(cond)]
        elif base:
            if isinstance(base[0], list):
                mean = [float(base[t][idx]) for t in range(h)
                        if t < len(base) and idx < len(base[t])]
            else:
                mean = []
        else:
            mean = []

        # Standardavvik fra std_P (per variabel)
        std_key_map = {"y":"y", "pi":"pi", "i":"i", "rer":"rer", "bolig":"bolig"}
        std_key = std_key_map.get(dsge_name, dsge_name)
        std = std_p.get(std_key, [0.3] * h)
        std = [float(s) for s in std[:h]]
        # Fyll opp hvis std er kortere enn h
        while len(std) < len(mean):
            std.append(std[-1] if std else 0.3)

        if not mean:
            continue

        result[obs_name] = {
            "mean":    mean[:h],
            "std":     std[:h],
            "periods": periods[:len(mean)],
            "source":  "dsge_fase2_conditional",
        }

    log.info(f"  DSGE-fremskrivning: {len(result)} variabler, h={h}")
    return result


def extract_dsge_meta(dsge: Dict) -> Dict:
    """Trekk ut nøkkelmetadata fra Fase II for dashboardet."""
    meta = dsge.get("meta", {})
    return {
        "eig_max":       meta.get("eig_max"),
        "n_states":      meta.get("n_states"),
        "n_shocks":      meta.get("n_shocks"),
        "last_obs_date": meta.get("last_obs_date"),
        "psrf_max":      meta.get("psrf_max"),
        "ess_min":       meta.get("ess_min"),
        "acc_rate":      meta.get("acc_rate"),
        "n_draws":       meta.get("n_draws"),
        "T_pre":         meta.get("T_pre"),
        "T_post":        meta.get("T_post"),
        "psi_R":         meta.get("psi_R"),
        "psi_P1":        meta.get("psi_P1"),
        "psi_Y":         meta.get("psi_Y"),
        "ss_i_pct":      meta.get("ss_i_pct"),
        "ss_pi_pct":     meta.get("ss_pi_pct"),
        "ss_y_pct":      meta.get("ss_y_pct"),
        "actual_rate_pct": meta.get("actual_rate_pct"),
        "actual_i_pct":  meta.get("actual_i_pct"),
        "actual_pi_pct": meta.get("actual_pi_pct"),
        "actual_y_pct":  meta.get("actual_y_pct"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# IRF-TRANSFORMASJON  (norske nøkler → korte JS-nøkler)
# ═══════════════════════════════════════════════════════════════════════════════

_SHOCK_TO_ID: Dict[str, str] = {
    "Pengepol.":    "monetary",
    "Risikopremie": "risk",
    "Prismarkup":   "cost",
    "TFP":          "tech",
    "Oljepris":     "oil",
    "Ettersp.":     "demand",
    "Bolig":        "housing",
    "Konsum":       "consumption",
}
_VAR_TO_ID: Dict[str, str] = {
    "BNP-gap":       "y",
    "KPI-inflasjon": "pi",
    "Styringsrente": "i",
    "RER-gap":       "rer",
    "Boligpris-gap": "q_H",
    "Konsum-gap":    "c_NW",
}


def _add_irf_est(dsge: Dict) -> None:
    """Legg til irf_est/irf_cal/irf_ci med korte nøkler som dashboardet forventer."""
    raw = dsge.get("irf", {})
    irf_est: Dict[str, Any] = {}
    for shock_label, vars_dict in raw.items():
        sid = _SHOCK_TO_ID.get(shock_label)
        if sid is None:
            continue
        irf_est[sid] = {
            _VAR_TO_ID[v]: vals
            for v, vals in vars_dict.items()
            if v in _VAR_TO_ID
        }
    dsge["irf_est"] = irf_est
    dsge.setdefault("irf_cal", {})
    dsge.setdefault("irf_ci",  {})
    log.info(f"  IRF transformert: {len(irf_est)} sjokk → irf_est")


def _add_posterior(dsge: Dict, chain_path: str = "chain_v3_v2_posterior.json") -> None:
    """Les MCMC-kjedesammendrag og legg til posterior-array for Estimering-tabellen."""
    if not os.path.exists(chain_path):
        log.warning(f"  chain-fil ikke funnet: {chain_path}")
        return
    with open(chain_path) as f:
        chain = json.load(f)
    summ = chain.get("summary", {})
    posterior = [
        {
            "name":  name,
            "km":    vals.get("km"),
            "mean":  vals.get("mean"),
            "std":   vals.get("std"),
            "p5":    vals.get("p5"),
            "p95":   vals.get("p95"),
            "psrf":  vals.get("psrf"),
            "ess":   vals.get("ess"),
            "fixed": False,
        }
        for name, vals in summ.items()
    ]
    dsge["posterior"] = posterior
    meta = dsge.setdefault("meta", {})
    psrf_vals = [p["psrf"] for p in posterior if p["psrf"] is not None]
    ess_vals  = [p["ess"]  for p in posterior if p["ess"]  is not None]
    if psrf_vals:
        meta["psrf_max"] = round(max(psrf_vals), 4)
    if ess_vals:
        meta["ess_min"] = round(min(ess_vals), 1)
    log.info(f"  Posterior: {len(posterior)} parametere (psrf_max={meta.get('psrf_max')}, "
             f"ess_min={meta.get('ess_min')})")


def _add_last_level(ck_data: Dict) -> None:
    """Sett last_level i forecasts fra siste historiske obs — tetter gap i Kryssjekk-grafene."""
    historical = ck_data.get("historical", {})
    for vname, fc in ck_data.get("forecasts", {}).items():
        if fc.get("last_level") is not None:
            continue
        hist = historical.get(vname, {})
        obs = [v for v in hist.get("observed", []) if v is not None]
        if obs:
            fc["last_level"] = obs[-1]


# ═══════════════════════════════════════════════════════════════════════════════
# SAMMENSLÅING
# ═══════════════════════════════════════════════════════════════════════════════

def merge(dsge_path: str, ck_path: str) -> Dict:
    """
    Slå sammen Fase II og Fase III output til ett dashboard-JSON-objekt.

    Struktur:
        {
          "meta":     {build_time, dsge_ok, ck_ok},
          "dsge":     {irf, fevd, hist_decomp, forecast, aktive_sjokk,
                       forecast_level, hist_level, posterior, meta, ...},
          "dsge_fc":  {obs_navn: {mean, std, periods}},   ← kryssjekk-format
          "crosscheck":{forecasts, historical, evaluation, weights, nowcast, meta}
        }
    """
    log.info("\n" + "=" * 55)
    log.info("  NEMO DASHBOARD BUILDER")
    log.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 55)

    # ── Last Fase II ──────────────────────────────────────────────────────
    dsge_data = load_dsge(dsge_path) if dsge_path else None
    dsge_ok   = dsge_data is not None
    if dsge_ok:
        _add_irf_est(dsge_data)
        _add_posterior(dsge_data)

    # ── Last Fase III ─────────────────────────────────────────────────────
    ck_data = None
    ck_ok   = False
    if ck_path and os.path.exists(ck_path):
        with open(ck_path, encoding="utf-8") as f:
            ck_data = json.load(f)
        ck_ok = True
        _add_last_level(ck_data)
        log.info(f"  Fase III lastet: {ck_path}")
    else:
        log.warning(f"  Fase III-fil ikke funnet: {ck_path}")

    # ── Trekk ut DSGE-fremskrivning i kryssjekkformat ─────────────────────
    dsge_fc = {}
    if dsge_ok:
        dsge_fc = extract_dsge_forecast(dsge_data)

    # ── Bygg output ───────────────────────────────────────────────────────
    output = {
        "meta": {
            "build_time": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "dsge_ok":    dsge_ok,
            "ck_ok":      ck_ok,
            "dsge_path":  dsge_path,
            "ck_path":    ck_path,
        },
        "dsge":      dsge_data or {},
        "dsge_fc":   dsge_fc,
        "crosscheck": ck_data  or {},
    }

    log.info(f"\n  Resultat:")
    log.info(f"    DSGE (Fase II):   {'OK' if dsge_ok else 'MANGLER'}")
    log.info(f"    Kryssjekk (Fase III): {'OK' if ck_ok else 'MANGLER'}")
    log.info(f"    DSGE-FC variabler: {len(dsge_fc)}")
    return output


# ═══════════════════════════════════════════════════════════════════════════════
# HTML-INJEKSJON
# ═══════════════════════════════════════════════════════════════════════════════

INJECT_MARKER = "/* __NEMO_DASHBOARD_DATA__ */"

# Chart.js-CDN som brukes i template — bakes inn slik at dashboardet fungerer
# offline (file://-protokoll) og uten blokkering av eksterne nettverk.
CDN_CHARTJS_URL = "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"
CDN_CHARTJS_TAG = '<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>'


def inline_chartjs(html: str) -> str:
    """Bak Chart.js inn i HTML slik at dashboardet fungerer offline."""
    if CDN_CHARTJS_TAG not in html:
        return html
    if requests is None:
        log.warning("  requests ikke installert — Chart.js lastes fortsatt fra CDN")
        return html
    try:
        resp = requests.get(
            CDN_CHARTJS_URL,
            timeout=30,
            headers={"User-Agent": "NEMO-Dashboard-Builder/1.0"},
        )
        resp.raise_for_status()
        inline_tag = f"<script>\n{resp.text}\n</script>"
        log.info(f"  Chart.js inlinet ({len(resp.text) // 1024} KB)")
        return html.replace(CDN_CHARTJS_TAG, inline_tag)
    except Exception as exc:
        log.warning(f"  Chart.js ikke innebygd (CDN feilet: {exc}) — bruker CDN-link")
        return html

def inject_into_html(data: Dict, template_path: str, output_path: str) -> None:
    """
    Injiser data-JSON direkte i HTML-template.
    Erstatter markøren /* __NEMO_DASHBOARD_DATA__ */ med faktisk data.
    """
    if not os.path.exists(template_path):
        log.error(f"  Template ikke funnet: {template_path}")
        return

    with open(template_path, encoding="utf-8") as f:
        html = f.read()

    if INJECT_MARKER not in html:
        log.error(f"  Markør '{INJECT_MARKER}' ikke funnet i template.")
        return

    def _serialize(obj: Any) -> Any:
        if isinstance(obj, (float, int)):
            return obj
        if isinstance(obj, bool):
            return obj
        raise TypeError(f"Ikke serialiserbar: {type(obj)}")

    json_str = json.dumps(data, ensure_ascii=False, separators=(",", ":"),
                          default=_serialize)
    # Escape tegn som bryter <script>-blokker (U+2028/U+2029 er ugyldige i JS
    # strengliteraler, og </ må brytes for å unngå tidlig </script>-lukking)
    json_str = (json_str
                .replace("\u2028", "\\u2028")
                .replace("\u2029", "\\u2029")
                .replace("</", "<\\/"))

    # Erstatt markøren med data-tilordning
    inject_block = f"const __NEMO_DATA__ = {json_str};"
    html_out = html.replace(INJECT_MARKER, inject_block)
    html_out = inline_chartjs(html_out)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_out)

    size_kb = os.path.getsize(output_path) / 1024
    log.info(f"\n  HTML injisert: {output_path}  ({size_kb:.0f} KB)")


# ═══════════════════════════════════════════════════════════════════════════════
# KJØRING
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NEMO Dashboard Builder — slår sammen Fase II og III output"
    )
    parser.add_argument("--dsge",     default="analyse_resultater.json",
                        help="Fase II: analyse_resultater.json")
    parser.add_argument("--ck",       default="crosscheck_ensemble.json",
                        help="Fase III: crosscheck_ensemble.json")
    parser.add_argument("--template", default="nemo_dashboard.html",
                        help="HTML-template med injeksjonsmarkør")
    parser.add_argument("--output",   default="nemo_dashboard_built.html",
                        help="Ferdig bygget HTML-fil")
    parser.add_argument("--json-out", default="nemo_dashboard_data.json",
                        help="Kombinert JSON (alltid lagret)")
    parser.add_argument("--json-only", action="store_true",
                        help="Kun lag JSON, ikke injiser i HTML")
    args = parser.parse_args()

    # Slå sammen
    data = merge(args.dsge, args.ck)

    # Lagre JSON
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False,
                  default=lambda x: float(x) if hasattr(x, "__float__") else x)
    log.info(f"  JSON lagret: {args.json_out}")

    # Injiser i HTML
    if not args.json_only:
        inject_into_html(data, args.template, args.output)

    print(f"\n  Ferdig.")
    print(f"  JSON:  {args.json_out}")
    if not args.json_only:
        print(f"  HTML:  {args.output}")
