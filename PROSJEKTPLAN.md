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

## Beslutninger

| Dato | Beslutning | Rasjonale |
|------|------------|-----------|
| 2026-05-14 | Innføre agentstruktur (PL + spesialister) | Se `AGENTER.md` |
| 2026-05-14 | Legge til Fase 0.5 før Fase 1 | Modellen har kjente svakheter (h_c og psi_R ved prior-grense, sigma_rp dominerer FEVD); reestimering på ny data uten å adressere disse vil bare gi nye estimater på samme svakheter |
| 2026-05-14 | Fase 0.5 skal være full revisjon, ikke fokusert | PE-valg: alle likninger gjennomgås mot K&M 2019 |

## Fase 0 — Restart og fundament ✅

**Mål:** Etablere ren mappestruktur og styringsdokumenter.

**Leveranser:**
- [x] `CLAUDE.md` — AI-styringsdokument
- [x] `AGENTER.md` — agentroller og arbeidsflyt
- [x] `PROSJEKTPLAN.md` — denne filen
- [x] `README.md` — brukerorientert
- [x] `pyproject.toml` — pakketdefinisjon og avhengigheter
- [x] Migrere `equations.py`, `parameters.py`, `blanchard_kahn.py` til `src/nemo/model/` og `src/nemo/solver/`
- [x] Migrere estimering til `src/nemo/estimation/`
- [x] Migrere analyse til `src/nemo/analysis/`
- [x] Fjerne all Fase III-kode (kryssjekk)

**Status:** Fullført 2026-05-14.

## Fase 0.5 — Modellkvalitetssikring 🚧

**Mål:** Verifisere at nåværende modellspesifikasjon er korrekt og
veldokumentert *før* vi henter ny data og reestimerer. Adressere kjente
svakheter (h_c og psi_R ved prior-grense, sigma_rp dominerer FEVD,
sigma_A svakt identifisert) og avdekke ukjente.

**Lead:** DSGE-økonom. **Bidrag fra:** NUM (numerisk verifikasjon), STAT
(identifikasjon), QA (review og tester).

### Kildedokumenter (Fase 0.5)

Likningsrevisjonen baseres på følgende offentlig tilgjengelige
dokumenter fra Norges Bank:

1. **Kravik & Mimir (2019), "Navigating with NEMO"**, Staff Memo 5/2019.
   Oversiktsdokument med Tabellene 8, 9, 10 (kalibrering og estimerte parametere).
   URL: https://www.norges-bank.no/contentassets/685d5f91e22a442c81d2e9680483e137/staff_memo_5_2019_eng.pdf

2. **Kravik, Mimir & Paulsen (2019), "A Complete Documentation of Norges Bank's Policy Model NEMO"**.
   Teknisk appendiks med komplette likninger per sektor.
   URL: https://www.norges-bank.no/contentassets/894596a4f3374ffd9a347f48862c2d75/nemo_complete_documentation_2019.pdf

3. **NB Memo 3/2024, "Norges Banks håndbok i pengepolitikk Versjon 2.0"** (desember 2024).
   Inneholder Figur 1: Impulsresponsfunksjoner av et pengepolitikkjokk i NEMO.
   Brukes som validerings-benchmark for vår modell (se Spor B5).
   URL: https://www.norges-bank.no/contentassets/c4425873fc5a4eeb816350cd6470e595/nb_memo_3_24_haandbok_pengepolitikk_rev.pdf

4. **Bergholt et al. (2019), "The Power of Forward Guidance in NEMO"**, Staff Memo 7/2019.
   Sekundær referanse for forventningsmekanismer.

5. **Mæhlum (2025), "Monetary Policy and the Exchange Rate in Norway"**, Staff Memo 3/2025.
   Sekundær referanse for UIP og valutakurs-dynamikk (relevant for `sigma_rp`-diagnose).

Alle referanser i likningsrevisjonen skal være etterprøvbare via disse
dokumentene. Hvis et likningsdetalj ikke kan henvises til en konkret
seksjon i NEMO-dokumentasjonen, må det dokumenteres som "egen
tilpasning" med separat begrunnelse.

### Leveranser

#### Spor A: Likningsrevisjon (DSGE-lead)
- [ ] **A1.** Gjennomgang av `build_matrices` (v1) — hver likning mot K&M (2019) eller NEMO-dokumentasjon, med sidereferanse. Rapport i chat med tabell: ligningsnummer | beskrivelse | K&M-referanse | status (OK / spørsmål / feil).
- [ ] **A2.** Gjennomgang av `build_matrices_v2` — verifiser at de tre fiksene (MC, Q_K, INV) er korrekte og at de blir riktig anvendt oppå v1.
- [ ] **A3.** Gjennomgang av `build_matrices_v3` — verifiser boligprislikning med b_sa / lambda_sa (Gelain et al. 2018), mimicking rule, og at h_c-oppdatering ikke etterlater inkonsistens.
- [ ] **A4.** Spesifikk gjennomgang av mistenkelige punkter identifisert i forhåndssjekken:
  - Banklikning (ligning 26): `G0[26, NB] = 1.0` og senere `G0[26, NB] += p.phi_c` → faktisk verdi 11.0. Intendert?
  - Mimicking rule (ligning 20) i v3: bruker `psi_R` på `(1-psi_R)*psi_Y` — er fortegnet konsistent med NEMO-spesifikasjon?
  - `EPS_PHI_H` i likning 22 og 23: står med samme fortegn (-1.0) — er det riktig at LTV-sjokk virker symmetrisk på begge utlånsrenter?
- [ ] **A5.** Steady-state-konsistens: kontroller at CY+IY+GY+XY-MY ≈ 1, og at IHY (boliginvestering) er konsistent med IY.

#### Spor B: Numerisk verifikasjon (NUM-lead)
- [ ] **B1.** Test at v1, v2, v3 alle gir stabile løsninger på kalibrerte parametere (`max|eig(T)| < 1`).
- [ ] **B2.** Sammenlign IRF mellom v1/v2/v3 for de samme sjokkene — dokumenter hvor versjonene skiller seg.
- [ ] **B3.** Verifiser Kalman-filterets håndtering av COVID-hullet i `kalman_hull` (`mcmc.py`): blir kovariansmatrisen reinitialisert riktig på post-blokken?
- [ ] **B4.** Sjekk at Cholesky-fallback (`LinAlgError → -np.inf`) ikke skjuler legitime numeriske problemer.
- [ ] **B5.** **NB Memo 3/2024 benchmark.** Beregne IRF for pengepolitikkjokk på dagens v3-modell med posterior mean, normalisere til samme styringsrente-topp som NB-figuren (~+1 pp), og sammenlikne mot Figur 1. Tabellere topp-magnitude, topp-tidspunkt og halveringstid for inflasjon, BNP, boligpris, RER, reallønn, styringsrente. Rapportere kvantitative avvik med hypoteser om årsak. Dette er **det viktigste eksterne valideringspunktet vi har**, og leverer kvantitativ grunnlag for diagnose av `sigma_rp`-anomalien (særlig RER-magnituden).

#### Spor C: Identifikasjons- og posterior-analyse (STAT-lead)
- [ ] **C1.** Plott prior vs. posterior for alle 17 estimerte parametere — visualiser hvor langt posterior har beveget seg.
- [ ] **C2.** Spesifikk analyse: hvorfor traff `h_c` (0.989 mot grense 0.9995) og `psi_R` (0.960 mot 0.990) prior-grensa? Tre hypoteser å teste:
  1. Modellen *trenger* veldig høy persistens — prior bør utvides
  2. Identifikasjon er svak — prior dominerer
  3. Modellspesifikasjonen mangler en kanal som ellers ville absorbert persistens
- [ ] **C3.** FEVD-diagnose for `sigma_rp`: 22 % av BNP, 88 % av RER. Test hypotesene fra `CLAUDE.md`:
  1. UIP-likningen mangler dynamikk (kobles til B5 NB-benchmark)
  2. `phi_B` (gjeldsavhengig premie) for lav
  3. Mangler separat finansiell faktor
- [ ] **C4.** Vurder om `sigma_A` (fast = 0.006) er svakt identifisert i nåværende modell, eller om det er en spesifikasjonsfeil. Hvis spesifikasjonsfeil — kan vi rette den og fri parameteren?
- [ ] **C5.** **Designdokument for Fase 2-estimering** (forhåndsregistrering). Skrive ned hvilke prior-revisjoner vi forhåndsforplikter oss til *gitt funn fra C2-C4*, før vi ser ny posterior. Skal hindre p-hacking-ekvivalent (gjentatte reestimeringer med justert prior til "fine" estimater oppnås). Dokumentet skal være versjonskontrollert.
- [ ] **C6.** **Observasjonsekvivalens-vurdering.** Kan vi skille mellom alternative spesifikasjoner med dagens 14 observasjonsserier? Konkret eksempel: høyt `sigma_rp` + flat UIP vs. lavt `sigma_rp` + dynamisk UIP. Hvis observasjonsekvivalente — trenger vi flere observerte serier (eks. terminkurs, kredittspread) eller informative priors.
- [ ] **C7.** **Identifikasjons-styrke per parameter.** Måle prior-til-posterior-bevegelse (KL-divergens eller posterior_std / prior_std). Hvis posterior ≈ prior, er parameteren ikke identifisert av data. Liste alle 17 parametere etter identifikasjonsstyrke.

#### Spor D: Test-suite (QA-lead, samarbeid ARK)
- [ ] **D1.** `tests/conftest.py` med fixtures (kalibrert modell, syntetisk data)
- [ ] **D2.** `tests/test_solver.py` — BK-stabilitet, dimensjoner, T-matrise-egenverdier
- [ ] **D3.** `tests/test_irf_signs.py` — 15 kvalitative krav
- [ ] **D4.** `tests/test_fevd_sum.py` — andeler summeres til ~100 %
- [ ] **D5.** `tests/test_likelihood.py` — Kalman-filter sanity, COVID-hull
- [ ] **D6.** GitHub Actions `tests.yml` som kjører `pytest tests/` ved push

#### Spor E: Dokumentasjon (DSGE/ARK)
- [ ] **E1.** Gjenopprette `docs/`-mappen
- [ ] **E2.** `docs/MODEL.md` — likningsoversikt med alle 48 tilstandsvariable og 13 sjokk, K&M-referanser
- [ ] **E3.** Sluttrapport fra Fase 0.5 med funn, anbefalinger, og åpne spørsmål til PE

### Akseptansekriterier

- Alle Spor A-funn dokumentert i rapport. Hvert "spørsmål" eller "feil" har anbefalt løsning.
- Alle tester i Spor D passerer på dagens modell *eller* feiler på en måte som dokumenterer en kjent svakhet.
- Identifikasjonsanalysen (C2, C3) gir tydelig svar: skal vi endre prior, endre modellen, eller leve med svakheten?
- `docs/MODEL.md` finnes og er komplett.
- PE har godkjent eller revidert sluttrapport.

### Risiko

| Risiko | Tiltak |
|--------|--------|
| Spor A avdekker fundamentale modellfeil | Eskaler til PE — kan kreve omfattende refactoring av `equations.py` |
| Spor C konkluderer at modellen mangler en kanal | Diskuter med PE: utvide modellen (Fase 0.6?) eller leve med svakheten |
| Identifikasjonsanalysen krever ny MCMC-kjøring | 2 timer per kjøring; planlegg eventuelle reestimeringer i bunt |

### Estimert tid

5–7 dager. Hvis Spor A eller C avdekker større problemer, kan dette utvides.

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
identifikasjon. *Mye av forarbeidet skjer i Fase 0.5 Spor C.*

**Leveranser:**
- [ ] `notebooks/identification.ipynb` — bygger på Fase 0.5 Spor C
- [ ] Revisjon av `src/nemo/estimation/priors.py` med begrunnelser
- [ ] Reestimering på fersk data (Fase 1)
- [ ] `data/results/posterior_v4.json` med PSRF, ESS, traceplots

**Akseptansekriterier:**
- Ingen posterior-middel < 5 % fra prior-grensa (eller dokumentert hvorfor)
- PSRF < 1.10 for alle parametere
- ESS/n > 0.02 for alle parametere
- Marginal likelihood dokumentert og høyere enn v3

## Fase 3 — Analyseverktøy konsolidering

**Mål:** Samle IRF, FEVD, historisk dekomposisjon, og betinget prognose i
ren `nemo.analysis`-pakke.

**Leveranser:**
- [ ] `src/nemo/analysis/irf.py`
- [ ] `src/nemo/analysis/fevd.py`
- [ ] `src/nemo/analysis/decomposition.py` — RTS-smoother + sjokk-bidrag
- [ ] `src/nemo/analysis/forecast.py` — betinget/ubetinget
- [ ] `tests/test_irf_signs.py` — 15 kvalitative krav (allerede laget i 0.5)
- [ ] `tests/test_fevd_sum.py` — andeler summeres til ~100 % (allerede laget i 0.5)

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

**Mål:** Månedlig oppdatering mellom kvartaler.

**Leveranser:**
- [ ] `src/nemo/analysis/nowcast.py` — én-stegs Kalman-prediksjon med
      delvis observerte serier (NaN for upubliserte variable)
- [ ] GitHub Actions workflow `nowcast.yml` — kjører månedlig
- [ ] Dashboard viser nowcast med separat indikasjon på "foreløpig"

**Akseptansekriterier:**
- Nowcasten oppdateres innen 48 timer etter SSB KPI-publisering
- Kalman-filteret håndterer mixed-frequency korrekt (kvartal + månedlig)

## Tidslinje (oppdatert)

| Fase | Estimert tid | Avhengig av               |
|------|--------------|---------------------------|
| 0    | 1 dag        | ✅ Fullført               |
| 0.5  | 5–7 dager    | Fase 0 ✅                 |
| 1    | 3–5 dager    | SSB-API tilgjengelighet   |
| 2    | 1 uke (kortere etter 0.5) | Fase 0.5 + Fase 1 |
| 3    | 3 dager      | Fase 2 ferdig             |
| 4    | 4 dager      | Fase 3 ferdig             |
| 5    | 1 uke        | Fase 1, 3 ferdig          |

## Risikoregister

| Risiko                                    | Sannsynlighet | Konsekvens | Tiltak |
|-------------------------------------------|---------------|------------|--------|
| Fase 0.5 avdekker modellfeil              | Middels       | Høy        | Eskaler til PE, juster scope |
| SSB endrer tabell-IDer                    | Lav           | Middels    | Hardkodede IDer i `src/nemo/data/ssb.py`, integrasjonstester |
| FRED API-nøkkel utløper                   | Lav           | Lav        | Cached fallback i `data/raw/` |
| Modellen feiler Blanchard-Kahn etter ny data | Lav        | Høy        | `test_solver.py` kjører før hver estimering |
| Prior-grensa fortsatt bindende            | Middels (lavere etter 0.5) | Middels | Fase 2 identifikasjonsanalyse |
| Nowcasten gir misvisende signaler         | Middels       | Høy        | Tydelig "foreløpig"-merking; bånd inkluderer prognoseusikkerhet |

## Beslutninger som krever prosjekteier

Se `AGENTER.md` for fullstendig eskaleringsliste. Hovedpunkter:

- Endre modellens dimensjon (NZ, NE)
- Endre COVID-hull-periodene
- Legge til ny variabel i observasjonssettet
- Bytte estimeringsalgoritme (f.eks. HMC i stedet for RWMH)
- Publisere dashboardet offentlig
- Endring i prior som strammer eller utvider støtten
