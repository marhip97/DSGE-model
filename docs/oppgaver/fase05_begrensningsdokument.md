# Fase 0.5 — Begrensningsdokument for NEMO v3

**Dato:** 2026-05-23  
**Godkjent av PE:** Ja (Alt A, 2026-05-23)  
**Neste fase:** Fase 1 — datainnhenting (SSB/NB/FRED til 2025)

---

## Hva modellen gjør bra

- **Stabilitet:** max|eig(T)| = 0.998 < 1.0 på alle kjøringer
- **Konvergens:** PSRF = 1.00–1.01 (godt under krav 1.10)
- **Kvalitative IRF:** 15/15 fortegnskrav bestått
- **In-sample fit:** log-posterior ≈ 3420 (kjøring 7 med φ_B)
- **Strukturelle fikser:** A4a (bank), A4c (LTV), CEE (Q_K), A5 (BNP-balanse),
  E3/E4 (LTV-fortegn), φ_B=0.0016 (UIP), h_c=0.938 fast (K&M)

---

## Kjente begrensninger

### 1. Pengepolitikk-IRF er for stor (primær begrensning)

Sammenlignet med NB Memo 3/2024, Figur 1 (ett standardavvik = +1pp rente, kv4):

| Variabel | Vår modell (kj. 8) | NB Memo 3/2024 | Ratio |
|---|---|---|---|
| BNP | −2.85 % | −0.45 % | **6.3×** |
| KPI | −0.44 % | −0.15 % | 2.9× |
| RER (appreciering) | −11.6 % | −0.40 % | **29×** |
| Boligpris | −34.9 % | −0.80 % | **44×** |

**Årsak:** σ_rp = 0.017 (estimert) vs. K&M kalibrert 0.006. Risikopremiesjokket
absorberer oljepris/kapitalstrøm-bevegelser i NOK som ikke er eksplisitt modellert.
Bekreftet i 7 kjøringer — robust posterior-konsentrering, ikke sampling-støy.

**Implikasjon for bruk:** Pengepolitiske sjokkanalyser overvurderer effekten ~6×.
Strukturelle sjokk (TFP, boligpreferanse, risikopremie) er bedre kalibrert.
Historisk dekomposisjon og FEVD er meningsfull, men pengepolitikkens andel
er sannsynlig overvurdert.

**Planlagt løsning:** Vurderes på nytt etter Fase 1 (data til 2025) + kjøring 9
(φ_I1 fri). Hvis problemet vedvarer, vurderes Alt B (oljepriskanal) som
Fase 3-utvidelse.

---

### 2. ESS lavt (statistisk begrensning)

ESS_min = 662 av 200 000 trekk (0.33 %). Krav: ESS/n > 0.02 (4 000 trekk).

**Implikasjon:** Posterior-estimater for h_c, psi_R og sigma_rp har høyere
Monte Carlo-usikkerhet enn standardfeil antyder. Punktestimatene er
sannsynligvis korrekte i retning, men konfidensbåndene er upresise.

**Planlagt løsning:** Fase 2 — forbedret sampler (blokksampling eller HMC
etter PE-godkjenning).

---

### 3. TFP-sjokk gir negativ BNP-respons (åpen, lav prioritet)

Teknologisjokk skal gi positiv BNP i standard RBC/NK. Sannsynlig årsak:
manglende MPK-ledd i Q_K-likning. Påvirker ikke pengepolitikk-IRF eller
estimering direkte.

**Planlagt løsning:** Spor A (likningsrevisjon) i Fase 2-forberedelse.

---

### 4. Fremoverskuende struktur / BK-determinisme (dokumentert, akseptert)

Modellen er indeterminert i Klein/BK-forstand (n_exp=5, rank(Pi)=7).
MSV-likevekten (M=0) er valgt som produksjonslikevekt — korrekt fundamental
RE-likevekt for et indeterminert system (Blanchard & Kahn 1980, Sims 2002).

Alt D (BK-determinisme via minimal strukturell endring) er forsøkt og bekreftet
ikke oppnåelig uten store strukturelle endringer utenfor K&M-rammeverket.

**Implikasjon:** Sunspot-ekvilibria er ekskludert per konstruksjon.
IRF til fundamentale sjokk er uendret ift. en hypotetisk deterministisk løsning.

---

## Parametere kalibrert fast (ikke estimert)

| Parameter | Verdi | Kilde |
|---|---|---|
| h_c | 0.938 | K&M Tabell 9 |
| sigma_A | 0.006 | K&M Tabell 10 |
| phi_B | 0.0016 | K&M Tabell 8 |
| phi_O | 0.15 | K&M Tabell 8 |

---

### 5. RER/bolig FEVD-dominans ved lang horisont (ny, 2026-06-01)

FEVD viser at boligsjokket forklarer 71 % av RER-variansen ved q20, mot kun 4 % ved q4.

**Årsak:** sigma_H = 0.333 (posterior mean kj41) er det største estimerte sjokket.
Boligpriskanalen til valutakursen er aktiv i modellen via UIP og risikopremieleddet.
Over lengre horisonter dominerer det langsomme boligsjokket (rho_H ≈ 0.85).

**Implikasjon for bruk:** Langsiktige RER-projeksjoner vil være boligdrevet.
Kortsiktig RER (q4) er oljepris-dominert (69 %) — mer i tråd med empiri.

**Planlagt løsning:** Ingen umiddelbar tiltak. Dokumenteres som kjent begrensning.
Vurderes på nytt etter Fase 1 med oppdaterte data.

---

### 6. I_R.q12 feil fortegn vs. NB-benchmark (ny, 2026-06-01)

Alle MCMC-kjøringer (kj41–43) gir I_R.q12 > 0, mens NB Memo 3/2024 Figur 1 viser -0.15 pp.

**Årsak:** AR(1) Taylor-regel med høy psi_R≈0.95 gir geometrisk forfall uten reversering.
Mean-reversion i styringsrenten krever en PLT/LQ-mekanisme eller ekstern reverserende kraft.

**Implikasjon for bruk:** Pengepolitiske sjokkanalyser viser ikke korrekt rentenormalisering.

**Planlagt løsning:** kj44+ med LQ/PLT-mekanisme — krever PE-godkjenning.
Utsatt til etter Fase 1.

---

## Anbefalte begrensninger på bruk (frem til Fase 2)

1. **Bruk ikke** modellen til kvantitative pengepolitikk-IRF uten å skalere ned ~6×
2. **Bruk gjerne** modellen til FEVD, historisk dekomposisjon, og sjokk-identifikasjon
3. **Rapporter alltid** usikkerhetsbånd basert på posterior-trekk (ikke kun mean)
4. **Referer** til dette dokumentet ved presentasjon av resultater
