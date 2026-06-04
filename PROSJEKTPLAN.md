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
   **Status pr. 2026-05-14: PSRF oppfylt (1.0046), ESS/n IKKE oppfylt
   (0.0033, faktor 6 under krav).** Se Fase 0.5 Spor C8 og risikoregister.
4. Alle 15 kvalitative IRF-krav passerer
5. Månedlig nowcast oppdaterer prognosen mellom kvartaler
6. Dashboard viser IRF, FEVD, historisk dekomposisjon, prognose

## Beslutninger

| Dato | Beslutning | Rasjonale |
|------|------------|-----------|
| 2026-05-14 | Innføre agentstruktur (PL + spesialister) | Se `AGENTER.md` |
| 2026-05-14 | Legge til Fase 0.5 før Fase 1 | Modellen har kjente svakheter (h_c og psi_R ved prior-grense, sigma_rp dominerer FEVD); reestimering på ny data uten å adressere disse vil bare gi nye estimater på samme svakheter |
| 2026-05-14 | Fase 0.5 skal være full revisjon, ikke fokusert | PE-valg: alle likninger gjennomgås mot K&M 2019 |
| 2026-05-14 | Legge til ESS-mangel som eksplisitt risiko og Spor C8 | PSRF-konvergens er oppfylt, men effektivt antall trekk (ESS_min=662) er en faktor 6 under suksesskriterium 3. Dette gjør h_c=0.989 og psi_R=0.960 tolkningsmessig svake — vi vet ikke om verdiene er datadrevne eller samplingsartefakter. Må diagnostiseres før Fase 2. |
| 2026-05-14 | B5 NB-benchmark bruker posterior-trekk, ikke kun mean | IRF basert på posterior mean med ESS=662 arver Monte Carlo-usikkerhet. Kvantitative avvik fra NB Memo 3/2024 Figur 1 kan ellers feiltolkes som spesifikasjonsproblem når det er samplingsstøy. B5 rapporterer mean-IRF + 5/95-bånd fra trekk. |
| 2026-05-14 | Fase 2 omformulert fra "reestimering" til "revidert estimering med forbedret sampler / informert prior" | Den gamle formuleringen ga inntrykk av at det bare gjenstår å trykke "kjør på nytt". Faktisk arbeid: forbedre mixing (blokksampling for korrelerte parametere, reparametrisering, eller HMC etter PE-godkjenning) **og** revidere prior basert på Fase 0.5 Spor C-funn. |
| 2026-05-20 | phi_I1 frigjøres i kjøring 9 (19 param) | B5-analyse viste at fast phi_I1=4.0 er hovedårsak til BNP-ratio ~10×. Fri phi_I1 (~0.5) traff NB eksakt. |
| 2026-05-23 | **Alt A godkjent — akseptér modellbegrensning, start Fase 1** | sigma_rp=0.017 er strukturelt (oljepris/valutakanal mangler). 7 MCMC-kjøringer viser at ingen parameterfikseringer løser problemet. IRF-avvik (BNP 6×, RER 29×) dokumenteres i begrensningsdokument. Alt B (oljepriskanal) utsettes til etter Fase 1 når nyere data er tilgjengelig. Se `docs/oppgaver/fase05_begrensningsdokument.md`. |
| 2026-05-23 | **MSV godkjent som produksjonslikevekt** | BK-determinisme ikke oppnåelig med minimal v3-endring (Alt D bekreftet, gap=1 strukturelt). MSV (M=0, T=G0⁻¹G1) er korrekt fundamental-likevekt — K&M-konsistent, stabil, 15/15 IRF. |
| 2026-06-01 | **Logit-reparametrisering av psi_R (kj44)** | psi_R=0.99 bekreftet genuint (ikke samplingsartefakt). Posterior i logit-rom: mean=6.99, sd=0.0004. |
| 2026-06-02 | **AR(2) Taylor-regel forkastet (kj45)** | psi_R2→0 entydig. Mean-reversion via autoregressiv struktur umulig. |
| 2026-06-02 | **PLT-kanal implementert og estimert (kj46)** | psi_PL=0.051 identifisert, men PLT-effektvekt≈0.0006 neglisjerbar. Begrensning 6 bekreftet strukturell. |
| 2026-06-03 | **phi_O frigjort, phi_I1=0.50 fast (kj47–kj49)** | phi_O identifisert ~0.21 (>K&M 0.15), men phi_O-frigjøring presser psi_R 0.949→0.989. Ny begrensning 7: phi_O–psi_R-korrelasjon. kj41 forblir beste estimat (RMSE=0.277). |
| 2026-06-03 | **Fase 2 avsluttet — kj41 er referanseestimat** | kj41: PSRF=1.00, ESS/n≈0.003 (under krav), RMSE=0.277. Strukturelle begrensninger 6 og 7 dokumentert. Fase 3 kan starte. |
| 2026-06-04 | **GEORG-læringssteg godkjent og implementert (NZ_GEORG=64)** | `build_matrices_georg()` lagt til ved siden av v3 (urørt) for å isolere om NB-avviket skyldes politikkregel eller transmisjon. Funn: BLANDET — GEORG gir pukkelformet/persistent rentebane (regelen forklarer formen, betydelig forbedring mot NB), men reproduserer ikke I_R.q12-fortegnsskiftet (begrensning 6 består → transmisjon). Se `docs/oppgaver/GEORG_laeringssteg_plan.md` §11. |

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

## Fase 0.5 — Modellkvalitetssikring ✅

**Avsluttet 2026-05-23.** PE godkjente Alt A: akseptér begrensninger, start Fase 1.
Begrensningsdokument: `docs/oppgaver/fase05_begrensningsdokument.md`.

**Statusoppdatering 2026-05-20:** 8 MCMC-kjøringer gjennomført. Nøkkelfunn:
- Modellfix A4a/A4c/CEE/A5/E3E4 implementert og bekreftet ✓
- h_c=0.938 fast, sigma_A=0.006 fast, phi_B=0.0016 og phi_O=0.15 i UIP ✓
- B5-analyse avdekket: **phi_I1=4.0 (fast, K&M) er årsak til for liten BNP-respons** (0.4× NB).
  Fase2v2 med phi_I1 fri (~0.5) traff NB eksakt (-0.447 vs -0.450).
- PE godkjente 2026-05-20: phi_I1 frigjøres igjen i **kjøring 9** (19 param).
- sigma_rp: 0.017→0.014 med phi_O, men strukturelt problem gjenstår.

**Mål:** Verifisere at nåværende modellspesifikasjon er korrekt og
veldokumentert *før* vi reviderer estimeringen. Adressere kjente
svakheter:
- h_c og psi_R ved prior-grense (0.989/0.9995, 0.960/0.990)
- sigma_rp dominerer FEVD (22 % BNP, 88 % RER)
- sigma_A svakt identifisert (kalibrert fast = 0.006)
- **ESS_min=662 langt under krav (4000) til tross for PSRF=1.0046** — vi
  vet ikke om posterior-estimatene er datadrevne eller samplingsartefakter

**Lead:** DSGE-økonom. **Bidrag fra:** NUM (numerisk verifikasjon), STAT
(identifikasjon og mixing), QA (review og tester).

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
  - Mimicking rule (ligning 20) i v3: bruker `G1[20, PI_L]` (lagg av inflasjon), men kommentaren beskriver fremoverskuende `E[π_{t+4}]`. **PE-beslutning kreves:** implementere fremoverskuende eller dokumentere bakseende som "egen tilpasning".
  - `EPS_PHI_H` i likning 22 og 23: står med samme fortegn (-1.0) — er det riktig at LTV-sjokk virker symmetrisk på begge utlånsrenter?
- [ ] **A5.** Steady-state-konsistens: kontroller at CY+IY+GY+XY-MY ≈ 1, og at IHY (boliginvestering) er konsistent med IY.

#### Spor B: Numerisk verifikasjon (NUM-lead)
- [ ] **B1.** Test at v1, v2, v3 alle gir stabile løsninger på kalibrerte parametere (`max|eig(T)| < 1`).
- [ ] **B2.** Sammenlign IRF mellom v1/v2/v3 for de samme sjokkene — dokumenter hvor versjonene skiller seg.
- [ ] **B3.** Verifiser Kalman-filterets håndtering av COVID-hullet i `kalman_hull` (`mcmc.py`): blir kovariansmatrisen reinitialisert riktig på post-blokken?
- [ ] **B4.** Sjekk at Cholesky-fallback (`LinAlgError → -np.inf`) ikke skjuler legitime numeriske problemer.
- [ ] **B5.** **NB Memo 3/2024 benchmark — utvidet versjon.** Beregne IRF for pengepolitikkjokk på dagens v3-modell og sammenlikne mot Figur 1.
  - **Steg 1 — punktestimat:** posterior mean, normalisert til samme styringsrente-topp som NB-figuren (~+1 pp). Tabellere topp-magnitude, topp-tidspunkt og halveringstid for inflasjon, BNP, boligpris, RER, reallønn, styringsrente.
  - **Steg 2 — usikkerhetsbånd (NY).** Trekke N=500 posterior-trekk fra `chain_v3_v2_posterior.json` (eller løpende kjede hvis vi har rå-trekkene tilgjengelig), beregne IRF per trekk, rapportere mean + 5/95-bånd per variabel og horisont. Begrunnelse: posterior mean med ESS=662 har ikke ignorerbar Monte Carlo-usikkerhet; uten bånd vet vi ikke om kvantitative avvik fra NB-figuren er spesifikasjon eller sampling.
  - **Steg 3 — rapportering.** Kvantitative avvik mot NB med hypoteser om årsak. Særlig fokus på RER-magnitude (kobles til `sigma_rp`-diagnose i Spor C3).

#### Spor C: Identifikasjons-, posterior- og mixing-analyse (STAT-lead)
- [ ] **C1.** Plott prior vs. posterior for alle 17 estimerte parametere — visualiser hvor langt posterior har beveget seg.
- [ ] **C2.** Spesifikk analyse: hvorfor traff `h_c` (0.989 mot grense 0.9995) og `psi_R` (0.960 mot 0.990) prior-grensa? Fire hypoteser å teste:
  1. Modellen *trenger* veldig høy persistens — prior bør utvides
  2. Identifikasjon er svak — prior dominerer
  3. Modellspesifikasjonen mangler en kanal som ellers ville absorbert persistens
  4. **(NY)** Likelihood har en *ridge* langs h_c-aksen: når h_c→1 blir a3_W=(1-h_c)/(σ(1+h_c))→0, så rentens påvirkning på konsum forsvinner i grensen. Sjekk: plott log-likelihood (ikke posterior) som funksjon av h_c langs MAP, alle andre parametere fast. Hvis flat for h_c ∈ [0.95, 0.999], er det H4. Tilsvarende for psi_R (renteglatting → 1 betyr renten følger random walk).
- [ ] **C3.** FEVD-diagnose for `sigma_rp`: 22 % av BNP, 88 % av RER. Test hypotesene fra `CLAUDE.md`:
  1. UIP-likningen mangler dynamikk (kobles til B5 NB-benchmark)
  2. `phi_B` (gjeldsavhengig premie) for lav
  3. Mangler separat finansiell faktor
  - **(NY) Kvantitativ test:** Fest `sigma_rp = 0.006` (K&M-verdi) og reestimer de andre 16 parameterne på samme data. Krever ~2 timer MCMC — eskaleres til PE før kjøring. Hvis likelihood faller drastisk, *trenger* modellen høyt sigma_rp (strukturproblem, mest sannsynlig UIP). Hvis likelihood er omtrent uendret med justeringer i andre sjokk-std, er sigma_rp en absorberende parameter (identifikasjonsproblem).
- [ ] **C4.** Vurder om `sigma_A` (fast = 0.006) er svakt identifisert i nåværende modell, eller om det er en spesifikasjonsfeil. Hvis spesifikasjonsfeil — kan vi rette den og fri parameteren?
- [ ] **C5.** **Designdokument for Fase 2-estimering** (forhåndsregistrering). Skrive ned hvilke prior-revisjoner vi forhåndsforplikter oss til *gitt funn fra C2-C4 og C8*, før vi ser ny posterior. Skal hindre p-hacking-ekvivalent (gjentatte reestimeringer med justert prior til "fine" estimater oppnås). Dokumentet skal være versjonskontrollert.
- [ ] **C6.** **Observasjonsekvivalens-vurdering.** Kan vi skille mellom alternative spesifikasjoner med dagens 14 observasjonsserier? Konkret eksempel: høyt `sigma_rp` + flat UIP vs. lavt `sigma_rp` + dynamisk UIP. Hvis observasjonsekvivalente — trenger vi flere observerte serier (eks. terminkurs, kredittspread) eller informative priors.
- [ ] **C7.** **Identifikasjons-styrke per parameter.** Måle prior-til-posterior-bevegelse (KL-divergens eller posterior_std / prior_std). Hvis posterior ≈ prior, er parameteren ikke identifisert av data. Liste alle 17 parametere etter identifikasjonsstyrke.
- [ ] **C8.** **(NY) Mixing-diagnose.** PSRF=1.0046 oppfyller konvergenskravet, men ESS_min=662 (0.33 % av kjedelengden) er en faktor 6 under suksesskriterium 3 (ESS/n > 0.02 → 4000 av 200k). Leveranser:
  1. **Autokorrelasjonsfunksjon per parameter** (ikke bare ESS-tall). Identifiser hvilke parametere som mikser tregest — sannsynlig kandidat: h_c, psi_R, rho_C, rho_rp.
  2. **Korrelasjonsmatrise mellom parametere** fra posterior-trekk. Hvis h_c og rho_C er sterkt korrelerte (sannsynlig), forklarer det treg mixing under komponentvis RWMH.
  3. **Anbefaling for Fase 2-sampler.** Konkrete alternativer å eskalere til PE: (a) blokksampling for korrelerte parametere innenfor RWMH, (b) reparametrisering (f.eks. logit-transformasjon av beta-parametere som binder mot grense), (c) bytte til HMC — sistnevnte krever eksplisitt PE-godkjenning iht. AGENTER.md eskaleringsliste.
  4. **Implikasjon for C2-konklusjoner.** Hvis h_c-ESS i nåværende kjede er ~666, er punktestimatet 0.989 ikke pålitelig nok til å avgjøre H1 vs. H2 alene. C8 må derfor leveres *før* C2 konkluderer.

#### Spor D: Test-suite (QA-lead, samarbeid ARK)
- [ ] **D1.** `tests/conftest.py` med fixtures (kalibrert modell, syntetisk data)
- [ ] **D2.** `tests/test_solver.py` — BK-stabilitet, dimensjoner, T-matrise-egenverdier
- [ ] **D3.** `tests/test_irf_signs.py` — 15 kvalitative krav. **Merknad:** nåværende `blanchard_kahn.py` lister 13 sjekker; D3 må spesifisere de to siste eksplisitt (sannsynlig: boligsjokk → boligpris(+), LTV-sjokk → konsum_NW(-)) før implementering.
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
- Identifikasjons- og mixing-analysen (C2, C3, C8) gir tydelig svar: skal vi endre prior, endre modellen, bytte sampler, eller leve med svakheten?
- `docs/MODEL.md` finnes og er komplett.
- PE har godkjent eller revidert sluttrapport.

### Risiko

| Risiko | Tiltak |
|--------|--------|
| Spor A avdekker fundamentale modellfeil | Eskaler til PE — kan kreve omfattende refactoring av `equations.py` |
| Spor C konkluderer at modellen mangler en kanal | Diskuter med PE: utvide modellen (Fase 0.6?) eller leve med svakheten |
| **(NY) ESS_min=662 (0.33 % av kjede) langt under suksesskriterium 3** | Spor C8 diagnostiserer årsak (treg mixing, korrelerte parametere). Fase 2 må velge tiltak: blokksampling, reparametrisering, eller HMC (sistnevnte krever PE-godkjenning). Konsekvens for tolkning: h_c=0.989 og psi_R=0.960 i nåværende posterior kan være samplingsartefakter, ikke datadrevne — Spor C2 og C8 må leveres samlet før konklusjon. |
| Identifikasjonsanalysen krever ny MCMC-kjøring (C3 sigma_rp-test) | 2 timer per kjøring; eskaleres til PE før kjøring. Planlegg eventuelle reestimeringer i bunt. |

### Estimert tid

5–7 dager opprinnelig estimat. **Justert oppover etter C8-tillegg og B5-utvidelse: 7–10 dager**, fordelt slik:
- B5 (utvidet med posterior-trekk): 2–3 dager
- A-spor (rask v3-fokus først): 1–2 dager
- C-spor inkludert C8 mixing-diagnose: 3–4 dager
- D-spor parallelt: 1–2 dager
- E sluttrapport: 1 dag

Hvis Spor A eller C avdekker større problemer, kan dette utvides.

## Fase 0.75 — Sandkasse: Full modellprestasjon mot NB ✅

**Startet 2026-05-30. Avsluttet 2026-06-01.** PE gir full sandkasse-fullmakt: alle endringer inkludert
modellstruktur, nye sjokk/tilstandsvariabler og observasjonsvariabler — uten
eskalering — så lenge endringene er konsistente med normal DSGE-praksis og
NBs NEMO-dokumentasjon.

**Motivasjon:** kj34 (psi_R=0.88) gir RMSE(16pt NB)=0.200, men alle variabler
returnerer for raskt til null etter q4. Systematisk persistensunderskudd i Y, PI,
I_R og RER q8–q12 vs NB Memo 3/2024 Figur 1.

| Horisont | Y avvik | PI avvik | I_R avvik | RER avvik |
|----------|---------|----------|-----------|-----------|
| q4       | −13%    | −60%     | +11%      | +47%      |
| q8       | −56%    | −90%     | +94%      | −79%      |
| q12      | −87%    | Feil fortegn | +374% | Feil fortegn |

**Exit-strategi:** kj31 (RMSE=0.353) og kj34 (RMSE=0.200) bevares som referanselinjer.

**Suksesskriterium:** RMSE(16pt NB) ≤ 0.150, PI q4 ≥ 0.70× NB, PSRF < 1.10.

### Spor A — Datagrunnlag (kj35, betinget)

- A1: KPI total vs KPI-JAE — sammenlign PI-identifikasjon
- A2: Importprisvekst som observasjonsvariabel (dpM_obs) — NB-konsistent
- A3: Dataperiode post-COVID — sjekk lengde og identifikasjonsevne

### Spor B — Modellstruktur (kj35–kj37)

- B1: Parametersweep — gamma_p, kappa_M, h_c, rho_s, phi_PQ vs RMSE(16pt)
- B2: Dogmatisk prior for vinnende parameter(e) — kj35
- B3: Finansiell friksjon phi_B fri (kj36, betinget)
- B4: xi_w / xi_p lønns-/prisrigiditet fri (kj37, betinget)
- B5: NEMO complete documentation gjennomgang for manglende mekanismer

### Spor C — Estimering (kj38, betinget)

- C1: FEVD — varians-dekomposisjon Y/PI/I_R/RER per sjokk
- C2: sigma_rp diagnose (nå 0.016, K&M: 0.006)
- C3: Tettere prior-struktur basert på Spor A–B-funn
- C4: Taylor-regel koeffisienter (psi_P1, psi_Y) sweep

### Spor D — Modellutvidelser (kj39+, betinget)

Kjøres hvis Spor A–C ikke gir RMSE ≤ 0.150.

- D1: h_c fri med Beta(5,2)-prior (i stedet for fast 0.938)
- D2: Importprisinflasjonskanal — korrekt NEMO-spec verifisering
- D3: Persistent pengepolitikk-komponent Z_t (NE: 13→14)

**Verktøy:** `scripts/sandkasse_diagnostikk.py` — parametersweep, FEVD, datavergining.
**Logging:** `data/results/mcmc_log.md` under seksjoner "Sandkasse A/B/C/D".
**Kjøringer:** kj35–kj49 reservert.

---

## Fase 1 — Faktisk datainnhenting ✅ (Avsluttet 2026-06-01)

**Mål:** Erstatte syntetisk fallback med ekte API-kall.

**Leveranser:**
- [x] `src/nemo/data/ssb.py` — JSON-stat-klient for SSB (PxWeb v0 + v2)
- [x] `src/nemo/data/norges_bank.py` — SDMX-klient for POLICY_RATE, NIBOR/3M, EXR, CREDIT
- [x] `src/nemo/data/fred.py` — utenlandsdata (oljepris Brent, handelspartner-BNP)
- [x] `src/nemo/data/pipeline.py` — transformasjon (log-diff, demean, HP-gap)
- [x] `data/processed/nemo_data_kpi_jae.csv` — 15-variabel observasjonssett, 2001Q1–2025Q4
- [x] `data/processed/nemo_demean_kpi_jae.json` — demean-verdier per variabel
- [x] `tests/test_data_pipeline.py` — 35 tester, alle grønne

**Akseptansekriterier:**
- [x] Pipeline-kode implementert med cache-fallback
- [x] Format identisk med `nemo_data_faktisk_v2.csv` (+ `pi_core_obs`-kolonne)
- [x] Tester bekrefter at API-feil ikke krasjer pipelinen — 35/35 pass

**Kjøre pipeline lokalt (kreves — sky-IPer blokkert av SSB/NB):**
```bash
python -m nemo.data.pipeline --kpi-jae
# Output: data/processed/nemo_data_kpi_jae.csv + nemo_demean_kpi_jae.json
```
SSB PxWeb v2 og Norges Bank SDMX-API blokkerer cloud-IPer (403). Pipeline
må kjøres fra lokal maskin eller whitelistet server. Output commites til repo.

**Siste kjøring:** 2025Q4 (2025-12-31) — `nemo_data_kpi_jae.csv` er oppdatert.

**Risiko (gjenværende):** SSB-tabell-IDer kan endres. FRED krever API-nøkkel.

## Fase 2 — Revidert estimering med forbedret sampler / informert prior ✅

**Status: AVSLUTTET 2026-06-03.** Beste estimat: **kj41** (PSRF=1.00, RMSE=0.277).

**Gjennomført (kj41–kj49, 2001Q1–2025Q4):**

| Kjøring | Hva | Funn |
|---------|-----|------|
| kj41 | Baseline Fase 2 (N=21, build_v3_forward) | **RMSE=0.277**, psi_R=0.949 — beste estimat |
| kj44 | Logit-reparam psi_R | psi_R=0.99 genuint, ikke samplingsartefakt |
| kj45 | AR(2) Taylor-regel | psi_R2→0 entydig — mean-reversion umulig |
| kj46 | PLT prisnivåmål-kanal | psi_PL=0.051 identifisert men neglisjerbar (×0.0006) |
| kj47 | phi_O fri + rho_s fast=0 | phi_I1 kollapser til 0.10 → RMSE=0.603 |
| kj48 | LogNormal phi_I1-prior | phi_I1 kollapser igjen — likelihood for sterk |
| kj49 | phi_I1=0.50 fast + phi_O fri | phi_O=0.206 identifisert, RMSE=0.375 |

**Resultater:**
- **Begrensning 6 (I_R.q12):** Bekreftet strukturell. psi_R=0.949–0.989 i alle kjøringer.
  I_R.q12=0.84–0.86 vs NB −0.15. Ikke løsbart med AR(2), PLT, logit-reparam eller phi_O.
- **Begrensning 7 (phi_O–psi_R):** Ny. Frigjøring av phi_O presser psi_R 0.949→0.989.
  phi_O≈0.21 er identifisert (>K&M 0.15) men kan ikke brukes fritt uten IRF-forringelse.
- **ESS/n:** kj49 oppnådde ESS=1099 (ESS/n=0.0055) — fremdeles under krav 0.02.
  Rho-klusteret er fortsatt ESS-bottleneck.

**Referanseestimat for videre analyse:**
`data/results/chain_kj41_prod_posterior.json` (RMSE=0.277, PSRF=1.00, psi_R=0.949)

**Strukturelle kalibreringer (fast):**
- phi_I1=0.50 (B5-passing), rho_s=0.00, phi_O=0.15 (K&M), h_c=0.938, sigma_A=0.006
- phi_PQ=150, lambda_pi4=0.0, NZ=50

Se `docs/oppgaver/fase05_begrensningsdokument.md` for fullstendig begrensningsdokument.

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
| 0.5  | 7–10 dager   | Fase 0 ✅ (justert opp fra 5–7 etter C8 og B5-utvidelse) |
| 1    | 3–5 dager    | SSB-API tilgjengelighet   |
| 2    | 1–2 uker     | Fase 0.5 + Fase 1. Avhengig av sampler-valg (HMC krever PE-godkjenning og lengre implementasjonstid). |
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
| **(NY) ESS/n langt under krav i v3-posterior (0.0033 vs. krav 0.02)** | **Høy (allerede observert)** | **Høy** | Spor C8 diagnostiserer årsak. Fase 2 omformulert til å inkludere sampler-forbedring. Tolkningsadvarsel: nåværende posterior-estimater for h_c, psi_R, sigma_rp har høyere Monte Carlo-usikkerhet enn ESS-formelen for IID-trekk skulle tilsi. |
| **(NY) Sampler-bytte til HMC kan kreve omfattende refactoring** | Middels | Middels | C8 må vurdere mindre invasive alternativer (blokksampling, reparametrisering) først. HMC-bytte eskaleres eksplisitt før implementering. |

## Beslutninger som krever prosjekteier

Se `AGENTER.md` for fullstendig eskaleringsliste. Hovedpunkter:

- Endre modellens dimensjon (NZ, NE)
- Endre COVID-hull-periodene
- Legge til ny variabel i observasjonssettet
- Bytte estimeringsalgoritme (f.eks. HMC i stedet for RWMH) — **relevant for Fase 2 etter C8-funn**
- Publisere dashboardet offentlig
- Endring i prior som strammer eller utvider støtten
- **(NY) Kjøre sigma_rp-fastpunkts-eksperimentet i C3** (krever ~2 timer MCMC)
- **(NY) Valg av mimicking rule-spesifikasjon** (lagg vs. fremoverskuende inflasjon, jf. Spor A4)
