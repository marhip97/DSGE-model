# Mappestruktur — målbilde

Dette er strukturen vi sikter mot etter Fase 0 og 1. Filer merket med (n)
skal nyutvikles, (m) skal migreres fra dagens flate struktur, (b) skal
beholdes.

```
nemo/
├── CLAUDE.md                          (b)  Styringsdokument for AI-agent
├── PROSJEKTPLAN.md                    (b)  Faser, milepæler
├── README.md                          (b)  Brukerorientert
├── STRUKTUR.md                        (b)  Denne filen
├── LICENSE
├── pyproject.toml                     (b)  Pakketdefinisjon
├── .gitignore                         (b)
│
├── .github/workflows/
│   ├── tests.yml                      (n)  Kjør pytest ved push
│   ├── data_refresh.yml               (n)  Kvartalsvis datainnhenting
│   ├── estimate.yml                   (n)  Manuell, ~2 timer
│   ├── analyse.yml                    (m)  Erstatter dagens run_analyse.yml
│   └── nowcast.yml                    (n)  Månedlig oppdatering
│
├── src/nemo/
│   ├── __init__.py                    (n)
│   ├── __main__.py                    (n)  `python -m nemo run ...`
│   │
│   ├── model/
│   │   ├── __init__.py                (n)
│   │   ├── equations.py               (m)  Fra equations.py
│   │   ├── parameters.py              (m)  Fra parameters.py
│   │   └── steady_state.py            (n)  Steady-state beregning
│   │
│   ├── solver/
│   │   ├── __init__.py                (n)
│   │   └── blanchard_kahn.py          (m)  Fra blanchard_kahn.py
│   │
│   ├── data/
│   │   ├── __init__.py                (n)
│   │   ├── ssb.py                     (n)  JSON-stat klient
│   │   ├── norges_bank.py             (n)  SDMX-klient
│   │   ├── fred.py                    (n)  FRED-klient
│   │   ├── transforms.py              (n)  HP-filter, log-diff, demean
│   │   └── pipeline.py                (n)  Orkestrering
│   │
│   ├── estimation/
│   │   ├── __init__.py                (n)
│   │   ├── kalman.py                  (m)  Fra nemo_estimering_v3.py
│   │   ├── priors.py                  (m)  Fra nemo_estimering_v3.py
│   │   ├── mcmc.py                    (m)  Fra nemo_estimering_v3.py
│   │   └── diagnostics.py             (m)  PSRF, ESS
│   │
│   ├── analysis/
│   │   ├── __init__.py                (n)
│   │   ├── irf.py                     (m)  Fra nemo_analyse.py
│   │   ├── fevd.py                    (m)  Fra nemo_analyse.py
│   │   ├── decomposition.py           (m)  Fra nemo_analyse.py
│   │   ├── forecast.py                (m)  Fra nemo_analyse.py
│   │   └── nowcast.py                 (n)  Mixed-frequency Kalman
│   │
│   └── dashboard/
│       ├── __init__.py                (n)
│       ├── build.py                   (n)
│       └── templates/
│           └── dashboard.html         (n)
│
├── data/
│   ├── raw/                           .gitignore: cached API-responser
│   ├── processed/
│   │   ├── nemo_data.csv              (m)  Fra nemo_data_faktisk_v2.csv
│   │   └── nemo_demean.json           (n)  Demean-verdier
│   └── results/
│       ├── posterior_v4.json          (n)  Etter Fase 2 reestimering
│       └── analyse_v4.json            (n)  Etter Fase 3
│
├── tests/
│   ├── __init__.py                    (n)
│   ├── conftest.py                    (n)  Fixtures
│   ├── test_solver.py                 (n)  BK-stabilitet, IRF-tegn
│   ├── test_likelihood.py             (n)  Kalman, COVID-hull
│   ├── test_data_pipeline.py          (n)
│   ├── test_irf_signs.py              (n)  15 kvalitative krav
│   └── test_fevd_sum.py               (n)  Andeler summeres til ~100 %
│
├── notebooks/
│   ├── identification.ipynb           (n)  Fase 2: prior vs posterior
│   ├── shock_decomposition.ipynb      (n)  Historisk analyse
│   └── forecast_evaluation.ipynb      (n)
│
├── docs/
│   ├── MODEL.md                       (n)  Likningsoversikt
│   ├── DATA.md                        (n)  Datakilder, transformasjoner
│   └── ESTIMATION.md                  (n)  MCMC, prior, diagnostikk
│
└── scripts/
    ├── run_data.py                    (n)  Tynne wrappers rundt nemo.*
    ├── run_estimate.py                (n)
    └── run_analyse.py                 (n)
```

## Migreringsmappping

| Dagens fil                  | Ny plassering                              |
|-----------------------------|--------------------------------------------|
| `equations.py`              | `src/nemo/model/equations.py`              |
| `parameters.py`             | `src/nemo/model/parameters.py`             |
| `blanchard_kahn.py`         | `src/nemo/solver/blanchard_kahn.py`        |
| `nemo_estimering_v3.py`     | splittes i `src/nemo/estimation/{kalman,priors,mcmc,diagnostics}.py` |
| `nemo_analyse.py`           | splittes i `src/nemo/analysis/{irf,fevd,decomposition,forecast}.py` |
| `nemo_data_innhenting.py`   | erstattes av `src/nemo/data/*.py`          |
| `nemo_data_faktisk_v2.csv`  | `data/processed/nemo_data.csv` (regenereres) |
| `chain_v3_v2_posterior.json`| `data/results/posterior_v3.json` (legacy)  |
| `analyse_resultater.json`   | `data/results/analyse_v3.json` (legacy)    |
| `main.py`                   | `src/nemo/__main__.py`                     |

## Filer som slettes uten erstatning

- `README.txt` (duplikat av `equations.py`)
- `crosscheck_*.py` (Fase III)
- `crosscheck_*.json` (Fase III)
- `nemo_dashboard_*.html/json` (gjenoppbygges i Fase 4 fra `analyse_v4.json`)
- `prosjektplan_fase3.txt` (erstattes av `PROSJEKTPLAN.md`)
- `.github/workflows/run_fase3.yml`
