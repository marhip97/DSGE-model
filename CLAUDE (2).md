# CLAUDE.md

Styringsdokument for AI-agenter (Claude) som jobber på NEMO-prosjektet.
Les denne filen først ved hver nye samtale.

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

## Arbeidsregler

### Før du gjør endringer

1. Les `PROSJEKTPLAN.md` for fasestatus
2. Sjekk `tests/` — alle tester må fortsatt passere etter endring
3. Sjekk om endringen tilhører gjeldende fase. Hvis ikke, spør først.

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

1. Begrunnelse i commit-meldingen (helst med sidereferanse til K&M 2019
   eller annen kilde)
2. Bestått `test_irf_signs.py` — alle 15 kvalitative IRF-krav må holde
3. Stabilitet: `max |eig(T)| < 1.0` etter Blanchard-Kahn
4. PSRF < 1.10 og ESS/n > 0.02 hvis modellen reestimeres

Endringer i prior krever begrunnelse. Forrige iterasjon traff grensene for
`h_c` (0.989, mot øvre grense 0.9995) og `psi_R` (0.960, mot øvre grense
0.990). Hvis du strammer prior, dokumenter hvorfor.

### Når du gjør datainnhenting

- **Cache rådata i `data/raw/`** med tidsstempel og kilde i filnavnet.
  Ikke kall API-er flere ganger enn nødvendig.
- **API-feil skal ikke krasje pipelinen.** Hvis FRED er nede, bruk siste
  cached versjon og advar i loggen.
- **Aldri commit `data/raw/`.** Den er i `.gitignore`. `data/processed/`
  commitees, fordi det er reproduserbar input til estimeringen.
- **SSB-tabeller, NB-serier og FRED-IDer er hardkodet i `src/nemo/data/`.**
  Hvis SSB endrer tabell-ID, oppdater der — ikke i kallende kode.

### Når du legger til avhengigheter

- Tilføy til `pyproject.toml` med versjon-pin (`>=` på minor, `<` på major)
- Hold totalavhengighet liten. Vi har: numpy, scipy, pandas, statsmodels,
  requests. Det er nesten alt vi trenger.
- **Ingen** tunge avhengigheter (TensorFlow, JAX, pymc, stan) uten
  diskusjon. Vi har skrevet vår egen MCMC fordi vi trenger kontroll over
  COVID-hullet og adaptiv kovarians.

## Kjente fallgruver

### MCMC traff prior-grensen forrige gang

`h_c` (habit konsum) og `psi_R` (renteglatting) har posterior-middel veldig
nær øvre grense i prior. To mulige forklaringer:

1. Modellen *trenger* veldig høy persistens for å matche data — i så fall
   må vi utvide prior-støtten.
2. Identifikasjon er svak og prior dominerer — i så fall må vi sjekke
   marginal likelihood ved alternative spesifikasjoner.

Test dette før du justerer prior. Se `notebooks/identification.ipynb`.

### sigma_A er svakt identifisert

Vi kalibrerer `sigma_A = 0.006` fast. Ikke fri den uten å sjekke at
posterior nå er informativ.

### Risikopremie dominerer FEVD

`sigma_rp = 0.016` (mot K&M 0.006) gir at risikopremiesjokk forklarer 22 %
av BNP-varians og 88 % av RER-varians. Dette er mistenkelig høyt. Mulige
forklaringer:

1. UIP-likningen fanger ikke faktisk valutakursdynamikk
2. `phi_B` (gjeldsavhengig premie) er for lav
3. Vi mangler en separat finansiell faktor

Ikke ignorer dette. Det er den mest sannsynlige svakheten i modellen nå.

### COVID-hull

Estimeringen splitter likelihood i pre-COVID (≤2019Q4) og post-COVID
(≥2022Q1), med 8 kvartaler hull. Ikke endre periodene uten å sjekke at
Kalman-filteret reinitialiseres riktig i `src/nemo/estimation/kalman.py`.

## Spørsmål du skal stille hvis noe er uklart

- "Skal jeg starte ny estimering, eller bruke eksisterende posterior?"
  (Ny estimering tar ~2 timer på laptop.)
- "Er denne endringen en del av gjeldende fase, eller skal den vente?"
- "Trenger jeg å oppdatere `tests/` etter denne endringen?"

## Hva som *ikke* skal endres uten eksplisitt godkjenning

- Modellens dimensjon (NZ=48, NE=13)
- COVID-hull-periodene
- Mappestruktur (`src/nemo/...`)
- Lisens

## Referansedokumenter

- `PROSJEKTPLAN.md` — faser og milepæler
- `README.md` — brukerorientert
- `docs/MODEL.md` — modellbeskrivelse (når den er skrevet)
- Kravik & Mimir (2019), Norges Bank Staff Memo 5/2019
