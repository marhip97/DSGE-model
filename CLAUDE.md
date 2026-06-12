# CLAUDE.md

Styringsdokument for AI-agenter (Claude / Claude Code) som jobber på
NEMO-prosjektet. Les denne filen først ved hver nye samtale.

## Hva du må lese ved start

1. **`CLAUDE.md`** (denne filen) — generelle prosjektregler
2. **`AGENTER.md`** — rolledefinisjoner og arbeidsflyt (Prosjektleder + spesialister)
3. **`PROSJEKTPLAN.md`** — faser, milepæler, status. Sjekk **hvilken fase som er aktiv** (markert med 🚧).
4. **`docs/oppgaver/`** — konkrete oppgavebeskrivelser for aktiv fase

## Hva prosjektet er

NEMO er en DSGE-modell for norsk økonomi, implementert i Python, inspirert
av Kravik & Mimir (2019). Den brukes til tre formål:

1. **Sjokkanalyse** — IRF, FEVD, historisk dekomposisjon
2. **Optimal pengepolitisk respons** — analyse av Norges Banks mimicking rule
3. **Realtid anslag** — kvartalsvise prognoser med månedlige nowcasts

Prosjektet er rendyrket — det skal *kun* være NEMO. Kryssjekkmodeller
(BVAR/AR) er fjernet. Ikke gjeninnfør dem uten eksplisitt instruks fra
prosjekteier.

## Hva prosjektet ikke er

- **Ikke** et generelt prognosesystem. Vi bruker DSGE-strukturen.
- **Ikke** Norges Banks operative NEMO. Vi bruker offentlig dokumentert
  parameterisering fra K&M (2019).
- **Ikke** et ML-prosjekt. Bayesiansk estimering er den eneste
  estimeringsmetoden.

## Aktiv fase

Per 2026-06-04: **Fase 2 — Revidert estimering** (avsluttet). Fase 0.5, 0.75 og 1 avsluttet.
Beste estimering: kj41 (psi_R=0.9490, RMSE=0.2771). Data: 2001Q1–2025Q4.
Se `PROSJEKTPLAN.md` for full beskrivelse, og `docs/oppgaver/` for konkrete oppgaver.

Anbefalt startsekvens for Claude Code:
- `docs/oppgaver/B5_nb_benchmark.md` (rask diagnose mot NB-figur)
- deretter parallelle spor A, C, D

## Arbeidsregler

### Før du gjør endringer

1. Sjekk hvilken rolle oppgaven faller under (se `AGENTER.md`).
   Start svaret/PR-beskrivelsen med rolletag: `[DSGE]`, `[STAT]`, `[NUM]`, `[DATA]`, `[ARK]`, `[QA]`, `[PL]`.
2. Les `PROSJEKTPLAN.md` for fasestatus.
3. Sjekk `tests/` — alle tester må fortsatt passere etter endring.
4. Hvis endringen tilhører eskaleringslisten i `AGENTER.md`, spør prosjekteier (PE) først.

### Når du skriver kode

- **Norske kommentarer og docstrings.** Variabelnavn på engelsk.
- **Type hints på alle offentlige funksjoner.**
- **Ingen syntetisk data i produksjonskode.** Bruk fixtures i `tests/` hvis
  du trenger deterministiske data.
- **Numerisk stabilitet:** alltid `cholesky` framfor `inv` der det er
  mulig, alltid `np.errstate` rundt logaritmer og divisjoner som kan
  produsere NaN.
- **Ingen `print` i bibliotekskode.** Bruk `logging` med `logger = logging.getLogger(__name__)`.
- **Datadrevne parametre må valideres.** Hvis en parameter kommer fra API,
  sjekk at den er innenfor rimelige grenser før den brukes.

### Når du endrer modellen

Endringer i `src/nemo/model/equations.py` eller `parameters.py` krever:

1. Begrunnelse i commit-meldingen med sidereferanse til K&M (2019) eller
   NEMO-dokumentasjon (`docs/references/nemo_complete_documentation_2019.pdf`).
2. Bestått `test_irf_signs.py` — alle 15 kvalitative IRF-krav må holde.
3. Stabilitet: `max |eig(T)| < 1.0` etter Blanchard-Kahn.
4. PSRF < 1.10 og ESS/n > 0.02 hvis modellen reestimeres.
5. Hvis prior strammes: dokumenter hvorfor (se STAT-regler i `AGENTER.md`).

### Når du gjør estimering (MCMC)

Se også `AGENTER.md` § "Bayesiansk statistiker (STAT)" for fulle regler.

- **Forhåndsregistrer prior-endringer** før kjøring. Loggføres i `data/results/mcmc_log.md`.
- **2 timer per kjøring** — ikke kjør på spec; sjekk identifikasjon først.
- **Resultat-fil:** `data/results/posterior_vN.json` med stigende versjonsnummer.

### Når du gjør datainnhenting (Fase 1+)

- **Cache rådata i `data/raw/`** med tidsstempel og kilde i filnavnet.
- **API-feil skal ikke krasje pipelinen.** Hvis FRED er nede, bruk siste
  cached versjon og advar i loggen.
- **Aldri commit `data/raw/`.** Den er i `.gitignore`.
- **SSB-tabeller, NB-serier og FRED-IDer er hardkodet i `src/nemo/data/`.**

### Når du legger til avhengigheter

- Tilføy til `pyproject.toml` med versjon-pin.
- Hold totalavhengighet liten. Vi har: numpy, scipy, pandas, statsmodels, requests.
- **Ingen** tunge avhengigheter (TensorFlow, JAX, pymc, stan) uten
  diskusjon med PE.

## Kjente fallgruver (oppdatert)

### `sigma_rp` dominerer FEVD

`sigma_rp = 0.016` (mot K&M 0.006) gir at risikopremiesjokk forklarer 22 % av
BNP-varians og 88 % av RER-varians. Sannsynligvis fordi UIP-likningen
mangler dynamikk. Adresseres i Fase 0.5 Spor C3 og benchmark mot NB Memo 3/2024
Figur 1 (Spor B5).

### MCMC traff prior-grensen forrige gang

`h_c` (habit konsum) traff 0.989 mot øvre grense 0.9995, og `psi_R` (renteglatting)
traff 0.960 mot 0.990. Adresseres i Fase 0.5 Spor C2.

**Tre hypoteser å teste:**
1. Modellen *trenger* veldig høy persistens — utvide prior
2. Identifikasjon er svak — prior dominerer
3. Modellspesifikasjonen mangler en kanal som ellers ville absorbert persistens

### sigma_A er svakt identifisert

Kalibrert fast = 0.006. Spor C4 vurderer om dette er identifikasjonsproblem
eller spesifikasjonsproblem.

### COVID-hull

Estimeringen splitter likelihood i pre-COVID (≤2019Q4) og post-COVID
(≥2022Q1), med 8 kvartaler hull. Ikke endre periodene uten å sjekke at
Kalman-filteret reinitialiseres riktig i `src/nemo/estimation/kalman.py`.
Endring krever PE-godkjenning (se eskaleringsliste).

### Inputdata er ikke sesongjustert (oppdaget 2026-06, dashbord-QA)

`dy_obs` (og trolig øvrige realserier) viser et sterkt, regelmessig sesongmønster
(snitt q/q: Q1 −3,98, Q2 −1,14, Q3 −0,53, Q4 +5,64 pp) — dvs. seriene er **ikke
sesongjustert**. Dette gir et hopp i dashbordets 4-kvartalersvekst der det rullende
vinduet møter den ikke-sesongbaserte modellprognosen. Dashbordet sesongjusterer nå
ved *visning* (deterministisk, kvartalssnitt trekkes fra i `analysis/run.py::_yoy`),
men den **egentlige** fiksen er sesongjustering i datapipelinen (`src/nemo/data/innhenting.py`).
Det påvirker estimeringen (kj41 er estimert på usesongjusterte data) og krever derfor
PE-godkjenning. Flagget i dashbordets begrensningstabell (Diagnostikk, Tabell 6).

### Styringsrenten faller på konsumsjokk (mimicking rule, ψ_W-dominans)

IRF for konsumpreferansesjokk (E_C) gir BNP-gap +2 % og KPI-inflasjon +0,6 %, men
styringsrenten *faller* (−0,14). Verifisert årsak: regelen vekter lønnsvekst tungt
(ψ_W = 0,87 > ψ_π = 0,56), og konsumsjokket (som treffer lønnslikningen, `Psi[1,E_C]=a2_W`)
demper lønnsveksten kraftig — lønnsveksleddet dominerer regelen og trekker renten ned.
En lærebok-Taylor-regel (ψ_W ≈ 0) ville hevet renten. Dette er en
estimerings-/spesifikasjonsegenskap, ikke en dashbord-feil. Flagget i Diagnostikk (Tabell 6).
Endring av regelvektene krever reestimering (PE).

## Spørsmål du skal stille hvis noe er uklart

- "Skal jeg starte ny estimering, eller bruke eksisterende posterior?"
  (Ny estimering tar ~2 timer på laptop.)
- "Er denne endringen en del av gjeldende fase, eller skal den vente?"
- "Trenger jeg å oppdatere `tests/` etter denne endringen?"
- "Er dette på eskaleringslisten i `AGENTER.md`?"

## Hva som *ikke* skal endres uten eksplisitt godkjenning

Se full liste i `AGENTER.md` § "Eskaleringsregler". Kortversjon:

- Modellens dimensjon (NZ=49 etter Alt. A 2026-05-15, NE=13)
- COVID-hull-periodene
- Mappestruktur (`src/nemo/...`)
- Ny variabel i observasjonssettet
- Bytte estimeringsalgoritme
- Endring i prior (strammere eller utvidet)
- Lisens

## Referansedokumenter

- `AGENTER.md` — agentroller, eskaleringsregler, arbeidsflyt
- `PROSJEKTPLAN.md` — faser og milepæler, statuslog
- `docs/oppgaver/` — konkrete oppgavebeskrivelser
- `docs/references/` — offentlige NEMO-dokumenter (PDF, ikke committet)
  - `staff_memo_5_2019.pdf` — Kravik & Mimir (2019) "Navigating with NEMO"
  - `nemo_complete_documentation_2019.pdf` — komplett likningssett
  - `nb_memo_3_2024_haandbok.pdf` — håndbok med IRF-benchmark (Figur 1)
- `README.md` — brukerorientert

Kjør `scripts/fetch_references.sh` for å laste ned PDF-ene.
