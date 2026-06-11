"""
[ARK] Dashboard-generator for NEMO v3.

Leser analyse.json og produserer en selvforsynt HTML-fil med inline data og Chart.js.

Bruk:
    python -m nemo.dashboard.build --input data/results/analyse_resultater.json
    python -m nemo.dashboard.build --input ... --output data/results/dashboard.html
"""

import argparse
import datetime as _dt
import json
import logging
import os
import textwrap
from pathlib import Path

logger = logging.getLogger(__name__)

_TEMPLATE_PATH = Path(__file__).parent / "templates" / "dashboard.html"


def _js_color_palette() -> str:
    """Returner JS-array med farger for sjokk (8 sjokk)."""
    return """[
        '#2563eb','#16a34a','#dc2626','#d97706',
        '#7c3aed','#0891b2','#be185d','#65a30d'
    ]"""


def build_dashboard(analyse_path: str, output_path: str) -> None:
    """
    Les analyse.json og skriv selvforsynt dashboard.html.

    Args:
        analyse_path: Sti til analyse_resultater.json
        output_path:  Sti til output-HTML
    """
    with open(analyse_path) as f:
        data = json.load(f)

    irf_json        = json.dumps(data.get("irf", {}))
    fevd_json       = json.dumps(data.get("fevd", {}))
    hist_level_json = json.dumps(data.get("hist_level", {}))
    hist_decomp_json= json.dumps(data.get("hist_decomp", {}))
    fcst_level_json = json.dumps(data.get("forecast_level", {}))
    overview_hist_json = json.dumps(data.get("overview_hist", {}))
    overview_fcst_json = json.dumps(data.get("overview_fcst", {}))
    diagnostics_json   = json.dumps(data.get("diagnostics", {}))
    active_shocks_json = json.dumps(data.get("active_shocks", {}))
    meta            = data.get("meta", {})
    meta_json       = json.dumps(meta)
    # Statisk fallback for headeren (synlig uten JavaScript)
    last_obs        = meta.get("last_obs_date") or "–"
    built_date      = _dt.date.today().isoformat()

    # Les template og fyll inn data
    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    html = template.replace("__IRF_DATA__",         irf_json) \
                   .replace("__FEVD_DATA__",        fevd_json) \
                   .replace("__HIST_LEVEL_DATA__",  hist_level_json) \
                   .replace("__HIST_DECOMP_DATA__", hist_decomp_json) \
                   .replace("__FCST_LEVEL_DATA__",  fcst_level_json) \
                   .replace("__OVERVIEW_HIST__",    overview_hist_json) \
                   .replace("__OVERVIEW_FCST__",    overview_fcst_json) \
                   .replace("__DIAGNOSTICS__",      diagnostics_json) \
                   .replace("__ACTIVE_SHOCKS__",    active_shocks_json) \
                   .replace("__META_DATA__",        meta_json) \
                   .replace("__LAST_OBS_FALLBACK__", last_obs) \
                   .replace("__BUILT_DATE__",       built_date)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html, encoding="utf-8")
    logger.info("Dashboard lagret: %s", output_path)
    print(f"Dashboard lagret: {output_path}")


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description="NEMO dashboard-generator")
    parser.add_argument("--input",  default="data/results/analyse_resultater.json")
    parser.add_argument("--output", default="data/results/dashboard.html")
    args = parser.parse_args()
    build_dashboard(args.input, args.output)


if __name__ == "__main__":
    main()
