# Prosjektplan — NEMO

## Mål

Bygge en velfungerende DSGE-modell for norsk økonomi, med automatisk
datainnhenting, bayesiansk estimering, og realtid prognoser. Modellen
skal brukes til å analysere sjokk og optimale pengepolitiske responser.

## Suksesskriterier

1. Pipelinen kjører fra kommandolinjen uten manuelle inngrep:
   `python -m nemo run --refresh-data --analyse`
2. Datainnhenting fra SSB, Norges Bank og FRED uten syntetiske fallback
3. Estimeringen konvergerer (PSRF < 1.10, ESS/n > 0.02) på 200k trekk
4. Alle 15 kvalitative IRF-krav passerer
5. Månedlig nowcast oppdaterer prognosen mellom kvartaler
6. Dashboard viser IRF, FEVD, historisk dekomposisjon, prognose

## Fase 0 — Restart og fundament

**Mål:** Etablere ren mappestruktur og styringsdokumenter.

**Leveranser:**
- [x] `CLAUDE.md` — AI-styringsdokument
- [x] `PROSJEKTPLAN.md` — denne filen
- [x] `README.md` — brukerorientert
- [x] `pyproject.toml` — pakketdefinisjon og avhengigheter
- [ ] Migrere `equations.py`, `parameters.py`, `blanchard_kahn.py` til `src/nemo/model/` og `src/nemo/solver/`
- [ ] Migrere `nemo_estimering_v3.py` til `src/nemo/estimation/`
- [ ] Migrere `nemo_analyse.py` til `src/nemo/analysis/`
- [ ] Fjerne all Fase III-kode (kryssjekk)

**Akseptansekriterier:**
- `pip install -e .` fungerer
- `python -m nemo.solver.blanchard_kahn` løser kalibrert modell
- `pytest tests/` kjører (selv om få tester finnes)

## Fase 1 — Faktisk datainnhenting

**Mål:** Erstatte syntetisk fallback med ekte API-kall.

**Leveranser:**
- [ ] `src/nemo/data/ssb.py` — JSON-stat-klient for SSB tabeller 09189,
      03013, 10235, 05111, 09786, 07241, 08946
- [ ] `src/nemo/data/norges_bank.py` — SDMX-klient for POLICY_RATE,
      NIBOR/3M, EXR/B.EUR.NOK.SP, CREDIT_INDICATOR/K2.HH
- [ ] `src/nemo/data/fred.py` — utenlandsdata (oljepris, handelspartner-BNP)
- [ ] `src/nemo/data/pipeline.py` — transformasjon (log-diff, demean, HP-gap)
- [ ] `data/processed/nemo_data.csv` — endelig 14-variabel observasjonssett
- [ ] `data/processed/nemo_demean.json` — demean-verdier for nivå-rekonstruksjon
- [ ] `tests/test_data_pipeline.py` — sjekk format, missing-håndtering, demean

**Akseptansekriterier:**
- `python -m nemo.data.pipeline` produserer `data/processed/nemo_data.csv`
  fra ferske API-kall
- Filen har samme format som dagens `nemo_data_faktisk_v2.csv`
- Tester bekrefter at API-feil ikke krasjer pipelinen (cached fallback)

**Risiko:** SSB-tabell-IDer kan endres. FRED krever API-nøkkel
(`secrets.FRED_API_KEY` i GitHub Actions).

## Fase 2 — Prior-revisjon og reestimering

**Mål:** Adressere `h_c` og `psi_R` ved prior-grensa, og dokumentere
identifikasjon.

**Leveranser:**
- [ ] `notebooks/identification.ipynb` — prior vs posterior, marginal
      likelihood ved alternative spesifikasjoner
- [ ] Revisjon av `src/nemo/estimation/priors.py` med begrunnelser
- [ ] Reestimering på fersk data (Fase 1)
- [ ] `data/results/posterior_v4.json` med PSRF, ESS, traceplots

**Akseptansekriterier:**
- Ingen posterior-middel < 5 % av prior-grensa
- PSRF < 1.10 for alle parametere
- ESS/n > 0.02 for alle parametere
- Marginal likelihood dokumentert og høyere enn v3

**Risiko:** Reestimeringen tar ~2 timer per kjøring. Vi bør ha
identifikasjonsanalyse klar før vi setter i gang.

## Fase 3 — Analyseverktøy konsolidering

**Mål:** Samle IRF, FEVD, historisk dekomposisjon, og betinget prognose i
ren `nemo.analysis`-pakke.

**Leveranser:**
- [ ] `src/nemo/analysis/irf.py`
- [ ] `src/nemo/analysis/fevd.py`
- [ ] `src/nemo/analysis/decomposition.py` — RTS-smoother + sjokk-bidrag
- [ ] `src/nemo/analysis/forecast.py` — betinget/ubetinget
- [ ] `tests/test_irf_signs.py` — 15 kvalitative krav
- [ ] `tests/test_fevd_sum.py` — andeler summeres til ~100 %

**Akseptansekriterier:**
- `python -m nemo.analysis.run --posterior ... --output analyse.json`
  produserer samme struktur som dagens `analyse_resultater.json`
- Alle tester passerer

## Fase 4 — Dashboard (kun NEMO)

**Mål:** HTML-dashboard som viser IRF, FEVD, prognose, historisk
dekomposisjon. Ingen kryssjekk.

**Leveranser:**
- [ ] `src/nemo/dashboard/templates/dashboard.html`
- [ ] `src/nemo/dashboard/build.py` — fyller template fra `analyse.json`
- [ ] Faner: Oversikt | IRF | FEVD | Historikk | Prognose | Diagnostikk
- [ ] Bruke visualization-design skill for figurene

**Akseptansekriterier:**
- Statisk HTML, ingen serverside-avhengighet
- Funker offline (alle data inline)
- Responsiv: leselig på mobil

## Fase 5 — Realtid og nowcast

**Mål:** Månedlig oppdatering mellom kvartaler. SSB publiserer KPI månedlig,
Norges Bank publiserer styringsrenten i sanntid.

**Leveranser:**
- [ ] `src/nemo/analysis/nowcast.py` — én-stegs Kalman-prediksjon med
      delvis observerte serier (NaN for upubliserte variable)
- [ ] GitHub Actions workflow `nowcast.yml` — kjører månedlig
- [ ] Dashboard viser nowcast med separat indikasjon på "foreløpig"

**Akseptansekriterier:**
- Nowcasten oppdateres innen 48 timer etter SSB KPI-publisering
- Kalman-filteret håndterer mixed-frequency korrekt (kvartal + månedlig)

## Tidslinje (estimat)

| Fase | Estimert tid | Avhengig av               |
|------|--------------|---------------------------|
| 0    | 1 dag        | —                         |
| 1    | 3–5 dager    | SSB-API tilgjengelighet   |
| 2    | 1 uke        | Fase 1 ferdig             |
| 3    | 3 dager      | Fase 2 ferdig             |
| 4    | 4 dager      | Fase 3 ferdig             |
| 5    | 1 uke        | Fase 1, 3 ferdig          |

## Risikoregister

| Risiko                                    | Sannsynlighet | Konsekvens | Tiltak |
|-------------------------------------------|---------------|------------|--------|
| SSB endrer tabell-IDer                    | Lav           | Middels    | Hardkodede IDer i `src/nemo/data/ssb.py`, integrasjonstester |
| FRED API-nøkkel utløper                   | Lav           | Lav        | Cached fallback i `data/raw/`                                |
| Modellen feiler Blanchard-Kahn etter ny data | Lav        | Høy        | `test_solver.py` kjører før hver estimering                  |
| Prior-grensa fortsatt bindende            | Middels       | Middels    | Fase 2 identifikasjonsanalyse                                |
| Nowcasten gir misvisende signaler         | Middels       | Høy        | Tydelig "foreløpig"-merking; bånd inkluderer prognoseusikkerhet |

## Beslutninger som krever prosjekteier

- Endre modellens dimensjon (NZ, NE)
- Endre COVID-hull-periodene
- Legge til ny variabel i observasjonssettet
- Bytte estimeringsalgoritme (f.eks. HMC i stedet for RWMH)
- Publisere dashboardet offentlig
