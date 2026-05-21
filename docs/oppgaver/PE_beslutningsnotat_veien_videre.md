# PE-beslutningsnotat — Veien videre etter Fase 0.5

**Avsender:** STAT/DSGE  
**Mottaker:** Prosjekteier (PE)  
**Dato:** 2026-05-19  
**Status:** Krever PE-beslutning

---

## Sammendrag

Fem MCMC-kjøringer i Fase 0.5 viser et konsistent mønster: modellen klarer
ikke å reprodusere Norges Banks pengepolitikk-IRF (NB Memo 3/2024). BNP-responsen
er 6–10× for stor, RER-responsen 29–39× for stor. Tre ulike fikseringer er forsøkt
uten å løse problemet. Diagnosen peker mot en strukturell svakhet i UIP-likningen
som krever en PE-beslutning om videre retning.

---

## Dokumentert bevisgrunnlag

### Alle kjøringer — nøkkelresultater

| Kjøring | Parametre | psi_R | sigma_rp | BNP-ratio | KPI-ratio | Endring |
|---------|-----------|-------|----------|-----------|-----------|---------|
| 1 — reparam | 20 (phi_I1 fri) | 0.960 | 0.017 | ~15× | ~5× | Baseline |
| 2 — phi1fix | 19 (phi_I1=4.0) | 0.964 | 0.017 | ~10× | ~3× | phi_I1 fast |
| 3 — postfix | 19 (modellfix) | 0.842 | 0.017 | **6.3×** | 2.9× | Likningsrettelser |
| 4 — c3fix | 18 (sigma_rp=0.006) | **0.911** | 0.006 | 8.5× | ~2× | C3-fix forverret |
| 5 — hcfix | 18 (h_c=0.938) | **0.912** | 0.017 | **10.2×** | 0.2× | h_c-fix forverret |

### Observert mønster: kompensatorisk likevekt

Posterioret er i en tilstand der parametrene kompenserer for hverandre:

- **Fiksér sigma_rp=0.006** → psi_R stiger til 0.911 (kjøring 4). BNP-ratio 8.5×.
- **Fiksér h_c=0.938** → psi_R stiger til 0.912 (kjøring 5). BNP-ratio 10.2×.
- **Frigjør alt** → sigma_rp=0.017, psi_R=0.842 (kjøring 3). BNP-ratio 6.3×.

Sigma_rp er robust 0.017 på tvers av alle kjøringer. Data «vil ha» sigma_rp=0.017,
og psi_R justerer seg for å passe. Enkeltparameter-fikseringer endrer ikke
underliggende dynamikk.

### Mekanisme

sigma_rp inngår direkte i UIP-likningen som standardavvik på risikopremiesjokket.
Med sigma_rp=0.017 (mot K&M 0.006) forklarer risikopremiesjokket:
- **22%** av BNP-varians (FEVD)
- **88%** av RER-varians (FEVD)

Sannsynlig årsak: UIP-likningen absorberer all uforklart valutadynamikk fordi
modellen mangler eksplisitte finansielle friksjoner (kapitalstrømsregulering,
bankpremie separat fra risikopremie). Spor C3-hypotesen fra PE-eskalering 2026-05-18.

---

## Alternativer

### Alt. 1 — Simultanfiksering: h_c=0.938 OG sigma_rp=0.006 (anbefales ikke)

**Hva:** Fiksér begge K&M-parametrene samtidig, estimer de resterende 17 fri.

**Fordeler:** Tvinger modellen til K&M-kalibrering.

**Ulemper:**
- psi_R vil sannsynligvis stige ytterligere (>0.92) for å kompensere for begge fikseringene.
- Kan gi dramatisk likelihood-fall (>100 log-enheter) — modellen passer ikke data.
- KPI-responsen ble nesten null i kjøring 5 med h_c fast — kan forverres videre.
- Er en dobbel overstyring som ikke adresserer den strukturelle årsaken.

**Ressurser:** ~2 timer MCMC.

---

### Alt. 2 — Strukturell UIP-utvidelse: separat finansiell friksjon

**Hva:** Legg til eksplisitt kapitalstrømsfriskjon (φ_B) i UIP-likningen, separat
fra EPS_RP. Tilsvarer det Norges Bank har i sin produksjonsversjon av NEMO.

**Mekanisme:** I dag: `q_t = E_t[q_{t+1}] - (i_t - i*_t - rp_t)`.
Utvidet: `q_t = E_t[q_{t+1}] - (i_t - i*_t - rp_t - φ_B·b_t)`,
der b_t er netto utenlandsgjeld som demper UIP-volatile.

**Fordeler:**
- Adresserer den strukturelle årsaken (sigma_rp absorberer manglende kanal).
- Vil gi modellen et eget «ankerfeste» for valutakursen.
- Konsistent med K&M (2019) §3.4 som nevner φ_B som kalibreringsvalg.

**Ulemper:**
- Krever likningsendring i `equations.py` (NZ øker evt.).
- Krever ny kalibrering av φ_B (ikke estimert i K&M).
- 1–2 dagers arbeid pluss ny MCMC-kjøring (~2 timer).
- Modellendring som krever formell PE-godkjenning og validering mot testpakken.

**Ressurser:** 1–2 dagers implementering + ~2 timer MCMC.

---

### Alt. 3 — Akseptere avvik og dokumentere modellbegrensning

**Hva:** Avslutt B5-benchmark-diagnostikken. Dokumenter at modellen har
kjent overestimering av valutakanal (sigma_rp 2.8× K&M, RER-respons 29–39× NB).
Gå videre til Fase 1 (datainnhenting) og Fase 2 (full reestimering med norske data).

**Begrunnelse:**
- Avviket mot NB Memo 3/2024 er kjent og dokumentert.
- Modellen er stabil (BK OK), konvergerer (PSRF<1.02), og har rimelig in-sample fit.
- IRF-avvik fra NB kan skyldes at NB bruker proprietær modellversjon med utvidelser
  som ikke er offentlig tilgjengelige.
- Prosjektets primærformål er analyse av norsk økonomi, ikke replikering av NB.

**Ulemper:**
- Risiko for at overestimert sigma_rp gir misleadende sjokkanalyser og prognoser.
- Hvis modellen brukes til policy-råd, vil pengepolitikk-effekter overvurderes
  dramatisk (~10×).

**Ressurser:** Minimal (dokumentasjon ~1 time).

---

### Alt. 4 — Bytte observasjonssett: utelat RER fra estimering

**Hva:** Fjern RER (valutakurs) fra observasjonslikningene og estimer på nytt.
sigma_rp vil da ikke lenger tvinges til å forklare valutadynamikk.

**Fordeler:**
- Ingen likningsendring — bare Sv og H-matrise.
- Hurtig å implementere (< 1 time) + ~2 timer MCMC.
- Tester om sigma_rp-problemet er datadrevet (UIP-feiltilpasning).

**Ulemper:**
- Mister informasjon fra valutakursen.
- sigma_rp vil sannsynligvis falle mot K&M, men modellen vil gi dårligere
  valutakursprognoser.
- Ikke en strukturell løsning — symptombehandling.
- Krever PE-godkjenning (endring i observasjonssett).

**Ressurser:** ~1 time implementering + ~2 timer MCMC.

---

## Tilrådning

**Anbefalt rekkefølge: Alt. 4 → (om mislykket) Alt. 2**

### Steg 1 — Alt. 4: Utelat RER (1 uke)

Billigste test av hypotesen. Hvis sigma_rp faller til ~0.006–0.010 uten RER i
observasjonssettet, bekreftes det at problemet er datadrevet UIP-feiltilpasning,
ikke manglende kanal. B5-benchmarken vil da automatisk forbedres.

Hvis sigma_rp **fortsatt** er 0.017 uten RER → problemet er strukturelt (Alt. 2
nødvendig).

### Steg 2 — Alt. 2: UIP-utvidelse (om steg 1 mislykkes)

φ_B-friksjon er den strukturelt korrekte løsningen og er konsistent med K&M §3.4.
Gir modellen et ankerfeste for valutakursen uavhengig av risikopremiesjokket.

### Hvorfor ikke Alt. 1 (dobbeltfiksering)?
Fordi kompensasjonsmønsteret er robust — psi_R vil stige til ~0.92+ og likelihood
vil falle dramatisk. Det løser ikke problemet, bare gjemmer det.

### Hvorfor ikke Alt. 3 (akseptere avvik)?
Overestimering av pengepolitikk-effekter med faktor 10 gjør modellen uegnet for
policy-analyse uten eksplisitt kalibrert demper. Bør ikke aksepteres uten
dokumentert begrensning og varsel til sluttbrukere.

---

## PE-beslutning som trengs

☐ **Godkjenn Alt. 4** — utelat RER fra observasjonssettet, re-estimer 18 param  
☐ **Godkjenn Alt. 2** — strukturell UIP-utvidelse med φ_B-friksjon  
☐ **Godkjenn Alt. 1** — simultan fiksering h_c=0.938 + sigma_rp=0.006  
☐ **Godkjenn Alt. 3** — akseptér avvik, dokumentér og gå videre til Fase 1  
☐ **Annet** — ytterligere diagnose nødvendig

---

## Vedlegg: IRF-profiler kv1–kv8 (kjøring 3 vs. NB Memo 3/2024)

| Kvartal | BNP modell | BNP NB | KPI modell | KPI NB | Rente |
|---------|-----------|--------|-----------|--------|-------|
| kv1 | −3.7% | ~−0.1% | ~0% | ~0% | +1.0pp |
| kv2 | −3.5% | ~−0.3% | ~−0.1% | ~−0.1% | +0.9pp |
| kv3 | −3.2% | ~−0.4% | ~−0.2% | ~−0.1% | +0.8pp |
| kv4 | **−2.85%** | **−0.45%** | **−0.44%** | **−0.15%** | +0.7pp |
| kv8 | ~−1.5% | ~−0.3% | ~−0.3% | ~−0.1% | ~+0.3pp |

Kilde: Kjøring 3 posterior-middel; NB Memo 3/2024 Figur 1 (avlest).
