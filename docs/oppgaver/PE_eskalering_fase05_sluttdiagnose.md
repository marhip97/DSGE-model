# PE-eskalering — Fase 0.5 sluttdiagnose og veien videre

**Avsender:** STAT/DSGE  
**Mottaker:** Prosjekteier (PE)  
**Dato:** 2026-05-20  
**Status:** Krever PE-beslutning om prosjektretning

---

## Sammendrag

Sju MCMC-kjøringer og tre strukturelle fikseringer later til å bekrefte det samme: modellen
er i en robust kompensatorisk likevekt der `sigma_rp=0.017` og `psi_R≈0.91` er hva
de norske dataene krever, uavhengig av hvilke parametere som fikseres eller hvilke
ligninger som justeres. B5-benchmarken mot NB Memo 3/2024 viser BNP-ratio 6–10× og
er ikke vesentlig forbedret av noen av tiltakene i Fase 0.5.

**Konklusjonen er at modellen slik den er spesifisert ikke kan reprodusere NB Memo
3/2024 Figur 1 innenfor rimelig avvik.** Dette er en modellbegrensning, ikke et
estimerings- eller konvergensproblem.

---

## Gjennomført arbeid — 7 kjøringer

| Kjøring | Endring | sigma_rp | psi_R | BNP-ratio | lp |
|---------|---------|----------|-------|-----------|-----|
| 1 | Baseline (phi_I1 fri) | 0.017 | 0.960 | ~15× | — |
| 2 | phi_I1=4.0 fast | 0.017 | 0.964 | ~10× | — |
| 3 | Modellfix A4a/A4c/CEE/A5 | 0.017 | 0.842 | **6.3×** | 3404 |
| 4 | sigma_rp=0.006 fast (C3) | 0.006* | 0.911 | 8.5× | — |
| 5 | h_c=0.938 fast (C2 Alt A) | 0.017 | 0.912 | 10.2× | 3404 |
| 6 | RER utelatt (Alt. 4) | **0.020** | 0.912 | ~10× | 3296 |
| 7 | φ_B=0.0016 i UIP (Alt. 2) | 0.017 | 0.912 | 10.1× | **3424** |

*fast verdi, ikke estimert

**Tre observasjoner:**

1. `sigma_rp=0.017` er robust på tvers av alle kjøringer der den estimeres fritt
2. Enhver enkeltfiksering øker psi_R til ~0.91 som kompensasjon
3. φ_B forbedrer likelihood (lp 3404→3424) men endrer ikke posterior

---

## Hva vi vet med sikkerhet

**Modellen treffer godt på:**
- Konvergens: PSRF=1.00–1.01, ESS/n=1–7% — ingen sampler-problem
- In-sample fit: lp≈3404–3424 — rimelig likelihood
- Stabilitet: BK OK i alle kjøringer, max|eig(T)|=0.998
- Mange IRF-kvalitetskrav: 14/15 passerer (TFP-sjokket er unntaket)

**Modellen treffer ikke på:**
- Pengepolitikk-IRF: BNP 6–10× for stor vs. NB Memo 3/2024
- KPI-respons: 0.2× for liten (h_c-fikseringen drepte inflasjonskabelen)
- RER: 29–39× — valutakanalen er dominerende

---

## Sannsynlig grunnårsak

Norges økonomi har over estimeringsperioden (2001–2019, 2022–2024) hatt
**store, persistente valutakursbevegelser** som modellens UIP-ligning ikke
kan forklare med renteforskjell og risikoaversjon alene. Modellen absorberer
dette i `sigma_rp` (og delvis `psi_R`). Dette er et **modellspesifikasjonsproblem**,
ikke et dataproblem.

Mulig årsak: Norsk valutakurs påvirkes sterkt av oljepris og kapitalbevegelser
knyttet til Oljefondet — kanaler som ikke er eksplisitt modellert i den offentlig
tilgjengelige K&M-spesifikasjonen.

---

## Alternativer og tilrådninger

### Alt. A — Akseptér modellbegrensningen og gå videre (tilrådning 1)

**Handling:** Dokumenter at modellen har kjent overestimering av pengepolitikk-IRF
(BNP ~6×, RER ~29×). Definer eksplisitt hvilke analyser modellen er og ikke er
egnet for. Gå videre til Fase 1 (datainnhenting) og Fase 2 (full reestimering
med norske kvartalsvise data til og med 2025).

**Begrunnelse:**
- Modellen er stabil og konvergerer godt
- In-sample fit er rimelig (lp~3420)
- 14/15 IRF-krav passerer
- Prosjektets primærformål er sjokkanalyse og historisk dekomposisjon, ikke
  replikering av NB sine pengepolitikk-IRFer
- NB bruker proprietær modellversjon med utvidelser som ikke er offentlig dokumentert

**Risiko:** Pengepolitikk-effekter overvurderes ~6–10× i sjokkanalyser og prognoser.
Må kommuniseres tydelig til sluttbrukere.

**Ressurser:** ~1 dag (sluttrapport + begrensningsdokument).

---

### Alt. B — Oljepris/valutakanal-utvidelse (tilrådning 2 ved behov)

**Handling:** Modeller eksplisitt koblingen mellom oljepris og valutakurs via en
«Dutch disease»-kanal eller en Oljefondsformue-variabel som inngår i UIP.

**Begrunnelse:** Norsk valutakurs er strukturelt koblet til oljeprisen (NOK/USD
korrelasjon ~0.7 over 2001–2023). Denne kanalen er ikke i K&M og absorberes av
sigma_rp.

**Ulemper:**
- Betydelig modellutviklingsarbeid (2–4 uker)
- Krever ny kalibrering mot NB-dokumentasjon som ikke er offentlig tilgjengelig
- Kan introdusere nye identifikasjonsproblemer

**Ressurser:** 2–4 uker implementering + 2–4 MCMC-kjøringer.

---

### Alt. C — Bytte til et BVAR-supplement for pengepolitikk-IRF

**Handling:** Bruk NEMO-modellen til strukturell sjokkanalyse (FEVD, historisk
dekomposisjon), men bruk en enkel BVAR for pengepolitikk-IRF-referanser.

**Begrunnelse:** NEMO er godt egnet for å dekomponere historiske forløp.
BVARer er typisk bedre kalibrert mot empiriske pengepolitikk-IRFer.

**Merk:** Dette er en arkitekturendring fra «rendyrket NEMO» som definert i
CLAUDE.md. Krever PE-godkjenning som prosjektomfangsendring.

---

## Tilrådning

**Primær: Alt. A** — akseptér modellbegrensningen og gå videre til Fase 1.

Begrunnelse: Vi har brukt ~3 uker og 7 MCMC-kjøringer på å diagnostisere
sigma_rp-problemet. Tre ulike strukturelle fikseringer (h_c, no-RER, φ_B) har
ikke løst det. Den sannsynlige årsaken (oljepris-valutakanal) krever utvidelser
som er utenfor offentlig dokumentert K&M-spesifikasjon. Å bruke mer tid her
har lav forventet avkastning.

Modellen leverer det den kan levere innenfor K&M-rammeverket. Neste
skritt med høyere forventet verdi er Fase 1 (norske data til 2025) og Fase 2
(full reestimering) — der vil vi se om nyere data endrer posterioret vesentlig.

**Sekundær: Alt. B** — hvis Fase 2-reestimering med nye data ikke bedrer IRF,
vurder oljepris-valutakanal-utvidelse som Fase 3-prosjekt.

---

## Fasit fra Fase 0.5

Følgende er levert og dokumentert:

✓ Modellfix A4a (bank), A4c (LTV), CEE (Q_K), A5 (BNP-balanse), E3/E4 (LTV-fortegn)  
✓ φ_B=0.0016 aktivert i UIP (K&M Tabell 8)  
✓ h_c=0.938 kalibrert fast (K&M)  
✓ Testpakke: 54 tester, 15 IRF-krav (14/15 passerer)  
✓ B5-benchmark: avvik dokumentert og forstått  
✓ 7 MCMC-kjøringer med fullstendig konvergensrapport  
✓ sigma_rp-diagnose: strukturell (ikke datadrevet, ikke sampler-problem)  

Åpent (Spor A, lav hastegrad):  
☐ TFP-sjokk gir negativ BNP (manglende MPK-ledd i Q_K-ligning)

---

## PE-beslutning som trengs

☐ **Godkjenn Alt. A** — akseptér begrensning, skriv sluttrapport, start Fase 1  
☐ **Godkjenn Alt. B** — oljepris/valutakanal-utvidelse før Fase 1  
☐ **Godkjenn Alt. C** — BVAR-supplement for pengepolitikk-IRF  
☐ **Annet** — ytterligere diagnose nødvendig
