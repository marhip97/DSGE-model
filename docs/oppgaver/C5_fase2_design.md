# C5 — Forhåndsregistrert designdokument for Fase 2-estimering

**Rolle:** [STAT]  
**Dato registrert:** 2026-05-16  
**Kildegrunnlag:** C8-rapport (`data/results/C8_acf_rapport.md`), C2/C7-rapport
(`data/results/C2_C7_rapport.md`), C3-rapport (`data/results/C3_rapport.md`),
B5-avvikstabell (`data/results/B5_avvik_tabell.md`), posterior fra
`data/results/chain_fase2v2_prod_posterior.json`.

> **Forhåndsregistrering:** Ingen prior-endringer, sampler-valg eller
> valideringskriterier som er spesifisert her kan justeres etter at Fase 2
> er startet. Avvik fra planen eskaleres til prosjekteier (PE) med
> skriftlig begrunnelse.

---

## 1. Sampler-strategi

### Bakgrunn fra C8

C8-kjøringen (5 000 burn-in + 20 000 produksjon, komponentvis RWMH) viste:

- IAT ≈ 150 for nesten alle parametre (unntak: `sigma_H`, IAT=33.8).
- ACF(1) > 0.99 for 15 av 17 parametre; ACF(50) > 0.6 for alle unntatt `sigma_H`.
- ESS/n ≈ 0.006 for de fleste, med minimum ESS/n = 0.0057 (`rho_rp`).
- Sterkeste korrelasjon: `sigma_C`–`h_c`, r = −0.811. Dernest
  `rho_Ys`–`sigma_Ys` (r = −0.469) og `rho_A`–`sigma_Ys` (r = −0.335).

Den dominerende årsaken til høy IAT er den sterke negative korrelasjonen
mellom `sigma_C` og `h_c`. Med komponentvis RWMH foreslås én parameter om
gangen, mens posterior-tettheten har en dyp diagonal dal. Sampleren vandrer
langs dalen ved å ta mange små skritt, noe som gir svært høy autokorrelasjon.

### Anbefalt strategi

**Blokksampling** (prioritet: HØY, ingen PE-godkjenning nødvendig):

- **Blokk 1 — {sigma_C, h_c}:** Felles RWMH-proposal med 2×2 kovariansmatrise
  estimert fra C8-kjeden. Korrelasjonen r = −0.811 tilsier at et rotert
  forslag (langs den korrelerte retningen) reduserer IAT betraktelig. Etter
  reparametrisering (se §2) kjøres blokken i transformert rom
  (sigma_C, logit(h_c)).
- **Alle andre parametre:** Komponentvis RWMH som i C8. Skaleringene fra C8
  gjenbrukes som startpunkt (final_scale = 0.436 fra meta).

**Adapt-strategi:**

- Kjør 2 000 trekk med fast proposal, estimer empirisk kovariansmatrise for
  blokk 1, bytt til adaptert blokk. Deretter adapt hvert 500. steg i
  burn-in.
- Mål: akseptrate 0.20–0.40 for blokk 1, 0.20–0.35 komponentvis.

**Notis om HMC:** HMC (Hamiltonian Monte Carlo) ville gi vesentlig lavere IAT
og er den foretrukne langsiktige løsningen. HMC krever gradient-implementasjon
av likelihood og eskaleres til PE. Dersom blokksampling + reparametrisering
ikke oppfyller ESS/n > 0.02 etter full kjøring, eskaleres HMC-implementasjon
automatisk.

---

## 2. Reparametrisering

### Problemet

`h_c` (habit i konsum) og `psi_R` (renteglatting) treffer den øvre
priorbegrensningen konsistent:

- `h_c`: C8-mean = 0.9924, std = 0.0015, p95 = 0.9950 (øvre grense 0.9995).
  9.2 % av trekk ved øvre grense. Id_ratio = 0.012 (sterk identifikasjon).
- `psi_R`: C8-mean = 0.9627, std = 0.0103. 3.0 % ved øvre grense.

C2-analyse støtter H4 (likelihood-rygg langs grensen): ekstremt lav std ved
øvre grense er klassisk tegn på en likelihood-rygg som fortsetter inn i det
ikke-tillatte området. Reparametrisering avslører om dette er en ekte modus
eller numerisk artefakt.

### Implementasjon

Logit-transformasjon for begge parametre:

```
logit(x) = log(x / (1 - x))
```

Konkret: la θ_h = logit(h_c) og θ_R = logit(psi_R). Sampleren opererer i
(θ_h, θ_R) ∈ (−∞, +∞), og vi tilbake-transformerer:

```
h_c  = 1 / (1 + exp(−θ_h))
psi_R = 1 / (1 + exp(−θ_R))
```

Jacobian-leddet inkluderes i log-posterior:

```
log |J| = log(h_c * (1 − h_c)) + log(psi_R * (1 − psi_R))
```

**Praktisk:** Sampleren tar ubegrensede steg i θ-rom. Prioren spesifiseres
fortsatt på den originale skalaen (se §3), men evalueres etter
tilbake-transformasjon. Dermed forsvinner priorbegrensningsproblemet.

**For blokk 1 {sigma_C, h_c}:** sigma_C forblir i original skala (den er
halvt-normal og aldri nær sin nedre grense). Blokken opererer på
(sigma_C, θ_h).

**Ingen PE-godkjenning nødvendig** — dette er en numerisk endring som ikke
endrer priorens matematiske innhold.

---

## 3. Prior-revisjoner (forhåndsregistrert)

Tabellen under spesifiserer prior-status for alle 20 estimerte parametre i
`chain_fase2v2_prod_posterior.json`. Revisjoner er begrunnet i C8/C2/C7/C3-funn.

**Generell regel:** Prior endres kun der det er konkret evidens fra Fase 0.5.
Ingen "forsiktig stramming" uten begrunnelse. For svakt identifiserte parametre
(id_ratio > 0.88) brukes informativ prior kun hvis det finnes teoretisk
forankring.

### 3.1 Parametre der prior IKKE endres

| Parameter | Fase 2v2 mean | Id_ratio (C8) | Begrunnelse for å beholde |
|-----------|--------------|---------------|--------------------------|
| `rho_O`   | 0.437 | 0.475 | God identifikasjon. Posterior langt fra prior-grenser. Uendret. |
| `rho_Ys`  | 0.874 | 0.350 | God identifikasjon. Posterior godt innenfor støtte. Uendret. |
| `sigma_C` | 0.016 | 0.092 | Sterk identifikasjon. Blokk-sampler vil forbedre mixing. Uendret. |
| `sigma_O` | 0.088 | 0.153 | Sterk identifikasjon, men posterior mean = 0.108 vs K&M 0.079. Avvik forklares bedre med data enn prior-stramming. Uendret. |
| `sigma_Ys`| 0.0057| 0.060 | Sterk identifikasjon. Uendret. |
| `sigma_i` | 0.000275 | 0.053 | Sterk identifikasjon, posterior tett rundt K&M. Uendret. |
| `sigma_P` | 0.0055| 0.089 | Sterk identifikasjon. Uendret. |
| `sigma_H` | 0.144 | 0.175 | Eneste parameter med tilfredsstillende mixing i C8 (IAT=33.8, ESS/n=0.030). Posterior konsistent. Uendret. |
| `psi_Y`   | 0.263 | 0.956 | Svak identifikasjon (prior dominerer), men posterior mean nær K&M (0.242). Å stramme prior uten teoretisk grunnlag er uakseptabelt. Uendret. |
| `phi_I1`  | 0.505 | — | Ikke i C8-kjeden (nytt parameter i v2). ESS = 302 i fase2v2, PSRF = 1.007. Uendret inntil videre. |
| `phi_I2`  | 9.98  | — | ESS = 479, PSRF = 1.051. Noe lav ESS men ikke kritisk. Uendret. |
| `phi_u`   | 0.704 | — | ESS = 381, PSRF = 1.040. Uendret. |

### 3.2 Parametre der prior VURDERES/ENDRES

#### `rho_A` — Svak identifikasjon, vurder informativ prior

- C8: id_ratio = 0.902, IAT = 146, PSRF = 1.098 (nær grense).
- Fase2v2 mean = 0.186 (merk: dette er ikke den samme kjøringen som C8;
  C8 ga mean = 0.738 med en annen modellversjon).
- K&M-verdi: 0.804.

**Beslutning: UENDRET.** Diskrepansen mellom fase2v2 (0.186) og C8 (0.738)
er stor og skyldes ulike modellversjoner. Å bruke K&M-verdi som informativ
prior ville styre estimatet mot 0.804 uten empirigrunnlag. Problemet er
mixing (ESS/n), ikke prior-spesifikasjon. Blokksampling adresserer dette.

#### `rho_C` — Svak identifikasjon, posterior nær øvre grense

- C8: id_ratio = 0.893, IAT = 158, 3.5 % mot øvre grense.
- Fase2v2 mean = 0.876.
- K&M-verdi: 0.725.

**Beslutning: UENDRET.** Posterior er informativ (std er smalere enn prior).
At posterior er høy reflekterer data mer enn prior-dominans. Stramming mot
K&M ikke begrunnet av C2/C7.

#### `rho_rp` — Svak identifikasjon (id_ratio > 1), komplisert

- C8: id_ratio = 1.093 (posterior *bredere* enn prior — multimodal eller
  identifikasjonsproblem). IAT = 174.9 (tregeste parameter). 5.8 % mot
  øvre grense. PSRF = 1.215 (ikke konvergert).
- Fase2v2 mean = 0.701.
- K&M-verdi: 0.737.

**Beslutning: UENDRET, men med spesifikk diagnose.** Id_ratio > 1 kan skyldes
(a) multimodalitet, (b) samplingsartefakter fra lav ESS, eller (c) genuine
identifikasjonsproblemer. Med ESS/n ≈ 0.002 i C8 kan vi ikke tolke
posterior pålitelig. Blokksampling + 200k trekk i Fase 2 gir grunnlag for
ny evaluering. Dersom id_ratio fortsatt > 1 etter full kjøring, eskaleres
informativ prior til PE.

#### `rho_H` — Svak identifikasjon, posterior nær øvre grense

- C8: id_ratio = 0.961, IAT = 165.9, 3.9 % mot øvre grense.
- Fase2v2 mean = 0.611.
- K&M-verdi: 0.694.

**Beslutning: UENDRET.** Identifikasjon er svak, men stramming rundt K&M
ville gi en prior som er smalere enn den informatiserte posteriors
begrunnelse tillater. Mixing er problemet.

#### `psi_R` — Priorbegrensning, reparametrisering i stedet for prior-endring

- C8: mean = 0.963, std = 0.010, id_ratio = 0.084 (sterk id), H4 støttes.
- Fase2v2 mean = 0.963, std = 0.010.
- K&M-verdi: 0.666.

**Beslutning: REPARAMETRISERING, IKKE PRIOR-ENDRING.** Logit-transformasjon
(§2) fjerner priorbegrensningsproblemet. Prior endres ikke — modellen er
sterkt identifisert og posterior-mean er langt fra K&M, noe som er en
empirisk observasjon vi ikke skal styre bort fra ved prior-stramming.

#### `psi_P1` — Svak identifikasjon, posterior nær K&M

- C8: id_ratio = 0.886, IAT = 141.
- Fase2v2 mean = 0.311, std = 0.054.
- K&M-verdi: 0.292.

**Beslutning: UENDRET.** Posterior er nær K&M-verdi, og det er ikke grunnlag
for stramming selv om identifikasjon er svak.

#### `h_c` — Priorbegrensning, reparametrisering i stedet for prior-utvidelse

- C8: mean = 0.992, std = 0.0015, id_ratio = 0.012 (sterk id), H4 støttes.
- Fase2v2 mean = 0.987, std = 0.002.
- K&M-verdi: 0.938.
- C3 støtter H3: fast sigma_rp = 0.006 gir h_c = 0.987 (fortsatt høy men
  litt lavere). Likevel treffer h_c grensen.

**Beslutning: REPARAMETRISERING, IKKE PRIOR-UTVIDELSE.** Per CLAUDE.md-instruks:
"h_c: IKKE utvide prior — implementer heller logit-reparametrisering."
Logit-transformasjon (§2) implementeres. Prior beholdes på original skala.

#### `sigma_rp` — C3 støtter trangere prior sentrert på K&M

- C3: Når sigma_rp holdes fast til K&M-verdi (0.006), forbedres h_c og
  psi_R delvis. H3 støttes.
- Fase2v2 mean = 0.0163, std = 0.00152. K&M = 0.006.
- Posterior mean er 2.7× over K&M. B5-avvik viser at modellen er for
  "treig" i rente (Rente q8: vår modell 0.731, NB 0.200).
- Forhøyet `sigma_rp` absorberer dynamikk som burde gå via UIP-leddet
  (manglende modellkanal, C3/C2 H3-funn).

**Beslutning: STRAMMERE PRIOR.** Ny prior:

```
sigma_rp ~ HalfNormal(scale=0.005)
```

Dette sentrerer prioren nær K&M-verdien (0.006) og styrer estimatet bort
fra den ukalibrerte høye verdien (0.016) som sannsynligvis er artefakt av
manglende UIP-dynamikk. Denne endringen er PE-godkjent implisitt via
CLAUDE.md-instruksen ("sigma_rp: kan vurdere trangere prior sentrert på
K&M 0.006 siden C3 støtter H3"). Dokumenteres eksplisitt her.

**Begrunnelse:** C3-rapport konkluderer: "H3 STØTTES — Når sigma_rp ikke
kan absorbere risikopremiedynamikk, flyttes h_c og/eller psi_R fra
priorbegrensningen." Trangere prior på sigma_rp adresserer dette direkte,
uten å kalibrere fast (som ville fjerne graden av frihet som data trenger).

### 3.3 Oppsummering av prior-endringer

| Parameter | Gammel prior | Ny prior | Endring |
|-----------|-------------|----------|---------|
| `sigma_rp`| HalfNormal(0.006) ca. | HalfNormal(scale=0.005) | Strammet, sentrert K&M |
| `h_c`     | Beta-lignende, øvre=0.9995 | Uendret, men logit-transformert | Reparametrisering |
| `psi_R`   | Beta-lignende, øvre=0.9995 | Uendret, men logit-transformert | Reparametrisering |
| Alle andre | — | Uendret | — |

---

## 4. Kjørekonfigurasjon

### 4.1 Primærkjøring

```
N_burnin  = 20 000       # 4× C8-kjøringens burn-in
N_samples = 200 000      # 10× C8-kjøringen; nødvendig for ESS/n > 0.02
adapt_every = 500        # Adapt step-size og blokk-kovarians
check_every = 50 000     # Mellomlagre PSRF og ESS under kjøring
n_chains  = 4            # Minst 4 for PSRF
```

**Rasjonale for N_samples = 200 000:** Med IAT ≈ 150 trenger vi
200 000 / 150 ≈ 1 333 effektive trekk per parameter. Kravet ESS/n > 0.02
tilsvarer ESS > 4 000 for N = 200 000. Blokksampling bør redusere IAT for
`sigma_C`/`h_c` til ≈ 30–50, men vi dimensjonerer for worst-case IAT = 150
for de resterende parametrene.

**Estimert kjøretid:** C8-kjøringen (20k trekk) tok tilsvarende ca. 3–5
min (basert på metadata). 200k trekk ≈ 30–50 min. Full kjøring med 4 chains
≈ 2 timer. Akseptabelt per CLAUDE.md-regelen om 2 timer maks.

### 4.2 Adaptasjonslogg

Under burn-in lagres:
- Akseptrate per blokk og per komponentvis parameter hvert 500. steg.
- Mål: 0.234 (for høydimensjonal RWMH) ± 0.05 for komponentvis, 0.20–0.40
  for blokk 1.
- Juster skalaparameter (scale) dersom akseptrate er utenfor mål etter 2 000 steg.

### 4.3 Mellomlagring

Kjeden lagres til `data/results/chain_fase2_produksjon_vX.npy` med stigende
versjonsnummer X. Mellomlagringer (checkpoint) hvert 50k steg slik at
kjøringen kan gjenopptas ved avbrudd.

### 4.4 Målkriteria

| Kriterie | Krav | Stoppegel |
|----------|------|-----------|
| PSRF | < 1.05 for ALLE 20 parametre | Stopp ved PSRF < 1.05 etter min. 100k prod. |
| ESS/n | > 0.02 for ALLE 20 parametre | Stopp ved oppfylt + PSRF-kravet |
| Akseptrate blokk 1 | 0.20–0.40 | Advar hvis utenfor etter burn-in |
| Akseptrate komponentvis | 0.15–0.35 | Advar hvis utenfor etter burn-in |

### 4.5 Stopp-regler

1. **Tidsstopp:** Maks 200k produksjonstrekk × 4 chains = 800k totalt.
   Dersom kriteriene ikke er oppfylt innen dette, eskaleres til PE med
   diagnostikk-rapport.
2. **Divergens:** Dersom PSRF > 2.0 etter 50k trekk for mer enn én
   parameter, stopp og diagnose. Ikke prøv på ny uten endring i startpunkt
   eller proposal.
3. **NaN i likelihood:** Logg og avbryt umiddelbart. Sjekk numerisk
   stabilitet i Kalman-filteret (se `src/nemo/estimation/kalman.py`).

---

## 5. Validering før godkjenning

### 5.1 Syntetisk data-test (gjøres FØR full kjøring)

Kjør sampleren på kjent parameter-sett ("SYN-test"):

1. Bruk K&M 2019 Table 8–10 som "sann" parametrisering.
2. Simuler data fra modellen med disse parametrene (T = 100 kvartaler).
3. Kjør sampler med 20k trekk på simulerte data.
4. Sjekk gjenfinning: sann verdi skal ligge innenfor posterior 90 %-intervall
   for minst 18 av 20 parametre (kalibreringsnivå: 90 %).
5. Dokumenter i `data/results/syn_test_fase2.json`.

Formål: Verifisere at logit-reparametrisering og blokksampling er korrekt
implementert *før* vi kjører på ekte data.

### 5.2 Konvergensssjekk etter kjøring

Følgende sjekkes *etter* kjøring, *før* posterioren godkjennes:

| Sjekk | Verktøy | Krav |
|-------|---------|------|
| PSRF per parameter | `psrf()` i `src/nemo/estimation/` | < 1.05 alle |
| ESS/n per parameter | `ess_n()` | > 0.02 alle |
| ACF(1) < 0.95 | `acf()` | Minst 15/20 parametre |
| IAT < 100 | `iat()` | Minst 15/20 parametre |
| Trace-plot | Visuell inspeksjon | Ingen synlig trend |
| Posterior marginal plot | `plot_prior_posterior()` | Id_ratio < 0.7 for sigma-parametre |

### 5.3 IRF-validering

Etter godkjent posterior:

1. **Fortegnsjekk:** Alle 15 kvalitative IRF-krav i `test_irf_signs.py` må
   passere.
2. **Kvantitativ benchmark (B5-oppdatert):** Kjør IRF for pengepolitikksjokk
   (+1 pp rente) og sammenlign med NB Memo 3/2024 Figur 1. Dokumenter avvik
   i `data/results/B5_avvik_fase2.md`.
   - Spesielt: BNP-gap q1-avvik (vår modell −0.624 vs NB −0.200) skal
     reduseres mot ≤ |0.3| etter prior-revisjon på sigma_rp.
   - Rente-persistens (q8: vår 0.731 vs NB 0.200) er det mest dramatiske
     avviket. Forventes bedret med trangere sigma_rp-prior.

### 5.4 Blanchard-Kahn-sjekk

Etter posterior: verifiser at `max|eig(T)| < 1.0` for posterior mean og
for minst 95 % av posterior-trekk.

### 5.5 Godkjenningsbeslutning

Posterioren godkjennes (og lagres som offisiell versjon i
`data/results/posterior_vN.json`) dersom ALLE av følgende er oppfylt:

- PSRF < 1.05 alle 20 parametre
- ESS/n > 0.02 alle 20 parametre
- Alle 15 IRF-krav passerer
- BK-stabilitet for posterior mean
- Ingen åpenbar divergens i trace-plot

Dersom ett eller flere krav ikke er oppfylt, følges §6 eskaleringsplan.

---

## 6. Eskaleringsplan

### 6.1 Beslutninger STAT kan ta uten PE-godkjenning

- Justere akseptrate-mål innenfor [0.15, 0.50] for komponentvis og blokk.
- Endre `adapt_every` mellom 200 og 2 000.
- Kjøre ekstra diagnostikk-kjøringer (< 50k trekk) for å identifisere
  mixingproblemer.
- Velge startpunkt for kjeden (posterior mean fra Fase 2v2 anbefales).
- Dele kjede i ≥ 4 chains med ulike startpunkter.
- Implementere blokksampling og logit-reparametrisering (spesifisert i §1–2).
- Øke N_samples inntil 400k dersom PSRF/ESS-krav ikke er møtt etter 200k.

### 6.2 Beslutninger som eskaleres til PE

| Situasjon | Eskaleringstrigger | Hva PE bestemmer |
|-----------|-------------------|-----------------|
| HMC-implementasjon | ESS/n < 0.02 etter 400k trekk | Godkjenne HMC eller justere modellspesifikasjon |
| Modell-endringer | Liknings-endring for å adressere B5-avvik | Godkjenne spesifikasjonsendring |
| Prior-endringer utover sigma_rp | Ny informasjon etter kjøring som tilsier endring | Godkjenne endret prior |
| Utvidelse av observasjonssett | Ny variabel i observasjonslikningen | Godkjenne + sjekke COVID-hull |
| Endring av estimeringsperiode | Splitt-punkt endret | Godkjenne + sjekke Kalman-reinit |
| Justering av COVID-hull-periodene | Noe som helst | PE-godkjenning alltid |
| rho_rp id_ratio > 1 etter full kjøring | Diagnostisk funn | Informativ prior eller modellendring |
| psi_Y id_ratio > 0.9 etter full kjøring | Diagnostisk funn | Kalibrer fast? Informativ prior? |
| PSRF > 2.0 etter 50k trekk | Alvorlig divergens | Stopp og full gjennomgang |

### 6.3 Eskaleringsformat

Eskalering til PE dokumenteres som en kort notis i `data/results/mcmc_log.md`
med:

```
Dato: YYYY-MM-DD
Trigger: [beskriv situasjon]
Diagnostikk: [legg ved relevante tall]
Anbefaling fra STAT: [konkret forslag]
Beslutning PE: [fylles ut av PE]
```

---

## Vedlegg A — Komplett prior-tabell Fase 2

Parametre uten endring fra Fase 2v2. Spesifisert for referanse:

| Parameter | Dist. | Mean/Scale | Std | Nedre | Øvre | Endret? |
|-----------|-------|-----------|-----|-------|------|---------|
| `rho_A`   | Beta-lignende | 0.75 | 0.15 | 0.01 | 0.9995 | Nei |
| `rho_C`   | Beta-lignende | 0.75 | 0.15 | 0.01 | 0.9995 | Nei |
| `rho_O`   | Beta-lignende | 0.75 | 0.15 | 0.01 | 0.9995 | Nei |
| `rho_Ys`  | Beta-lignende | 0.75 | 0.15 | 0.01 | 0.9995 | Nei |
| `rho_rp`  | Beta-lignende | 0.75 | 0.15 | 0.01 | 0.9995 | Nei |
| `rho_H`   | Beta-lignende | 0.75 | 0.15 | 0.01 | 0.9995 | Nei |
| `sigma_C` | HalfNormal | 0.03 | ~0.028 | 0 | ∞ | Nei |
| `sigma_O` | HalfNormal | 0.08 | ~0.069 | 0 | ∞ | Nei |
| `sigma_Ys`| HalfNormal | 0.011 | ~0.012 | 0 | ∞ | Nei |
| `sigma_rp`| HalfNormal | — | 0.005 | 0 | ∞ | **JA** |
| `sigma_i` | HalfNormal | 0.0003 | ~0.0004 | 0 | ∞ | Nei |
| `sigma_P` | HalfNormal | 0.003 | ~0.005 | 0 | ∞ | Nei |
| `sigma_H` | HalfNormal | 0.05 | ~0.072 | 0 | ∞ | Nei |
| `psi_R`   | Beta-lignende | 0.7 | 0.12 | 0.01 | 0.9995 | Reparametrisert |
| `psi_P1`  | Normal (trunkert) | 0.29 | 0.10 | 0.0 | 1.0 | Nei |
| `psi_Y`   | Normal (trunkert) | 0.24 | 0.05 | 0.0 | 1.0 | Nei |
| `h_c`     | Beta-lignende | 0.9 | 0.12 | 0.01 | 0.9995 | Reparametrisert |
| `phi_I1`  | — | — | — | — | — | Nei |
| `phi_I2`  | — | — | — | — | — | Nei |
| `phi_u`   | — | — | — | — | — | Nei |

> Eksakte prior-parameterverdier leses fra `src/nemo/estimation/priors.py`
> (eller tilsvarende). Tabellen er indikativ; Python-koden er autoritativ.
> Eneste endring: `sigma_rp` scale reduseres til 0.005.

---

## Vedlegg B — Konsistenssjekk mot kildefunn

| Anbefaling i §1–6 | Kildefunn | Konsistent? |
|-------------------|-----------|-------------|
| Blokksampling {sigma_C, h_c} | C8: r(sigma_C, h_c) = −0.811, IAT ≈ 120–118 | Ja |
| Logit(h_c) og logit(psi_R) | C2: H4 støttes (std < 0.003 ved øvre grense) | Ja |
| sigma_rp: trangere prior (0.005) | C3: H3 støttes; C8: sigma_rp mean = 0.016 vs K&M 0.006 | Ja |
| rho_rp: ingen prior-endring | C8: PSRF = 1.215, ESS/n = 0.002 — ikke pålitelig nok for konklusjon | Ja |
| rho_A, psi_P1, psi_Y: uendret | C7: id_ratio høy, men mixing er årsak, ikke prior | Ja |
| N_samples = 200k | C8: IAT ≈ 150; ESS/n > 0.02 krever IAT-normalisert ESS > 200k × 0.02 = 4 000 | Ja |
| B5-benchmark: forvent bedret rente-persistens | B5: Rente q8-avvik = +0.531; sigma_rp-prior adresserer | Ja (forventning) |
| SYN-test før full kjøring | Standard validering for ny sampler-implementasjon | Ja |
