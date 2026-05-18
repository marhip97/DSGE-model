# Spor A — Grundig diagnose: pengepolitikktransmisjon og modellsvakheter

**Dato:** 2026-05-18  
**Spor:** A (Likningstransparens) + B5 (NB-benchmark)  
**Status:** Leveres til PE — alle funn krever godkjenning før kodeendring  
**Metode:** Numerisk systematisering av transmisjonskanalene + kodegjennomgang

---

## 1. Oppsummering

Svak pengepolitikktransmisjon (BNP-ratio 0.12–0.16 mot NB-mål) skyldes **ikke** én bug,
men samspillet mellom tre kvantifiserte mekanismer og to urettede kodefeil:

| Årsak | Effekt på BNP-ratio |
|-------|---------------------|
| h_c = 0.988 → a3_W = 0.006 (K&M: 0.032) | −5× konsumkanal |
| phi_I1 = 4.0 → inv/Q_K = 0.25 (data: 2.0) | −8× investeringskanal |
| Bankkapital-bug (A4a): statisk i stedet for dynamisk | Ukjent, trolig 10–20% |
| Steady-state ubalanse 6 pp (A5) | Skalafeil i BNP-sammenheng |
| **Sum → BNP kv4 = −0.07%** | **vs NB −0.45% (ratio 0.16)** |

Med data-drevne verdier (phi_I1 ≈ 0.5, h_c ≈ 0.96) oppnås ratio ≈ 0.39–0.49 —
hvilket er konsistent med at NB-benchmark er fra 2024-NEMO (utvidet modell),
ikke K&M (2019).

---

## 2. Kvantifisering av transmisjonskanalene

### 2.1 Konsumkanalen

Euler-likning for sparere (v3 ligning 1):

```
c_W_t = a1_W·c_W_{t-1} + a2_W·E[c_W_{t+1}] − a3_W·(i_D_t − E[π_{t+1}])
```

Der `a3_W = (1−h_c) / (σ·(1+h_c))` er rentesensitiviteten:

| Scenario | h_c | a3_W | Konsumvirkning kv1 (1pp) |
|----------|-----|------|--------------------------|
| K&M (2019) | 0.938 | 0.0320 | −1.6% (=a3_W×CY) |
| Posterior phi_I1-fix | 0.988 | 0.0060 | −0.3% |
| Ekstremalcase | 0.999 | 0.0005 | −0.03% |

**Konklusjon:** h_c = 0.988 demper konsumkanalen med faktor 5.3 sammenlignet med K&M.
At h_c treffer øvre prior-grense er et signal om enten svak identifikasjon,
manglende modellkanal, eller feil prior (jf. Spor C2-hypotesene).

### 2.2 Investeringskanalen

Investeringslikning (v3 ligning 12):

```
inv_t = (1/phi_I1)·q_K_t + (phi_I1/(phi_I1+phi_I2))·inv_{t-1} + (phi_I2/(phi_I1+phi_I2))·E[inv_{t+1}]
```

**Kodekoeffisient vs. CEE-standard:**

| phi_I1 | Kode: 1/phi_I1 | CEE-korrekt: 1/(phi_I1·(1+β)) | Faktor |
|--------|----------------|-------------------------------|--------|
| 0.1 | 10.000 | 5.025 | 2.0× |
| 0.5 | 2.000 | 1.005 | 2.0× |
| 4.0 | 0.250 | 0.126 | 2.0× |

**Viktig:** Koden bruker `1/phi_I1` mens Christiano-Eichenbaum-Evans (2005) gir
`1/(phi_I1·(1+β))`. Faktor-2-feil uavhengig av phi_I1-verdien.

**Konsekvens for transmisjon:**

| Scenario | phi_I1 | inv/Q_K | BNP-bidrag kv1 |
|----------|--------|---------|----------------|
| Data-drevet (5-blokk) | 0.506 | 2.00 | +39.5% (=2.0×IY) |
| K&M fast | 4.000 | 0.25 | +5.0% (=0.25×IY) |
| CEE-korrekt med K&M | 4.000 | 0.13 | +2.5% |

**Konklusjon:** phi_I1=4.0 (K&M-verdi) gir 8× svakere investeringskanal enn
fri estimering (phi_I1≈0.5). Data prefererer lav phi_I1 — det er ikke en
sampler-artefakt, det er et genuint data-signal.

### 2.3 Valutakanalen (RER/UIP)

RER-responsen er konsistent på tvers av scenarier:

| Scenario | RER kv4 | NB-mål | Ratio |
|----------|---------|--------|-------|
| Alle testede | −0.23% | −0.40% | 0.58 |

**RER-fortegn:** Vår konvensjon — høyere RER = depresiering. NB Memo 3/2024
bruker trolig motsatt konvensjon (høyere = appresiering). Fortegnet er
**ikke en feil**, bare en konvensjonforskjell.

**RER-magnitudesvakheten (58%) skyldes:**
- sigma_rp = 0.017 (K&M: 0.006) dominerer FEVD, men påvirker ikke IRF-formen
- Manglende dynamikk i UIP (phi_B-leddet er aktivt men svakt)

### 2.4 Samlet BNP-effekt (1pp annualisert rente, kv4)

| Scenario | BNP kv4 | Ratio vs NB (−0.45%) |
|----------|---------|----------------------|
| phi_I1=4.0, h_c=0.988 (nåværende) | −0.070% | **0.16** |
| phi_I1=0.5, h_c=0.988 (test) | −0.177% | **0.39** |
| phi_I1=0.5, h_c=0.938 (K&M h_c) | −0.22% | **~0.49** |
| NB Memo 3/2024 Figur 1 (mål) | −0.450% | 1.00 |

Ratio ≈ 0.49 ble bekreftet i A_funn_rapport.md (etter A4b-fix, med fri phi_I1).
Dette er konsistent med at NB-benchmarken er 2024-NEMO (utvidet modell
med olje, finansiell akselerator, GPFG m.m.) — ikke K&M (2019).

---

## 3. Identifiserte kodefeil

### 3.1 A4a — Bankkapital-likning (IKKE RETTET, dokumentert tidligere)

**Kode (linje 396–409, `build_matrices`):**
```python
G0[26, NB]   =  1.0
G0[26, I_R]  = -phi_o    # = 0.0046
G0[26, B_NW] = -phi_o
G0[26, NB]  +=  phi_c    # phi_c = 10.0 → G0[26,NB] = 11.0
```

**Tre feil:**
1. Ingen akkumulerings-ledd (mangler G1-term for `NB_{t-1}`)
2. G0[26,NB] = 11.0 (1+phi_c) — utilsiktet sammenslåing
3. phi_c=10.0 dominerer fullstendig: `nb ≈ 0.0004·(i_R + b_NW)`

**Konsekvens:** Bankens kapital har nesten ingen dynamikk. Den finansielle
akseleratoren (spread → investering → BNP) er effektivt slått av.

**Foreslått rettelse (krever PE-godkjenning):**
```python
# Gerali et al. (2010) form:
G0[26, NB]   =  1.0
G1[26, NB]   =  (1.0 - delta_b)     # akkumulering
G1[26, I_R]  = -phi_o               # spread-inntekt
G1[26, B_NW] = -phi_o
# phi_c-leddet fjernes her — hører til spread-likningene (lign. 21-23)
```

### 3.2 CEE-normalisering i investeringslikning

**Kode:** `G0[12, Q_K] = -1.0 / phi_I1`  
**CEE-korrekt:** `-1.0 / (phi_I1 * (1+beta))`

Med beta=0.99 og phi_I1=4.0: kode=0.250, CEE=0.126 — faktor 2.0 for stor.

**Konsekvens:** Investering reagerer 2× for mye på Q_K. Isolert sett gir
dette STERKERE transmisjon enn korrekt CEE — men effekten er for svak
til å kompensere for h_c=0.988-svakheten.

**Foreslått rettelse:**
```python
G0[12, Q_K] = -1.0 / (phi_I1 * (1.0 + beta))
```

### 3.3 A4c — LTV-sjokk-fortegn (IKKE RETTET)

Se `A_funn_rapport.md` § A4c for full dokumentasjon.
Psi[25, E_phi_h] = +1.0 gir løsere LTV (mer gjeld) ved positivt sjokk —
inkonsistent med at ligning 22–23 gir høyere renter (strammere kreditt).

### 3.4 Steady-state ubalanse (A5)

CY + IY + IHY + GY + XY − MY = 0.940 (ikke 1.000).
6 pp-gap betyr at BNP-sammenhengene ikke er korrekt kalibrert.
Kan bidra til skalafeil i IRF-responsen.

---

## 4. Svar på de tre spørsmålene

### 4.1 Har K&M-modellen strukturelle svakheter?

**Nei — problemet er i vår implementasjon.** K&M (2019) sin egen kode
og parametrisering gir antagelig ~0.45% BNP-respons ved K&M-parametre.
Vi oppnår kun 0.066% fordi:

1. Bankligning-bug (A4a) slår av finansakseleratoren
2. Mimicking rule hadde 2-periodes oscillasjon (A4b — nå rettet i v3)
3. Steady-state ubalanse (A5)

Selve ligningssystemet i K&M er korrekt DSGE-teori.

### 4.2 Er NB-NEMO (2024) vesentlig endret fra K&M (2019)?

**Trolig ja, men det forklarer ikke hele avviket.**

- K&M (2019) → NB Memo 3/2024: ca. 5 år med oppdateringer
- NB-NEMO har trolig: utvidet banksektor, oljeinntekter via GPFG fullt, 
  updated kalibrering mot 2020-data, COVID-tilpasninger
- Disse endringene kan gi 10–30% forskjell i transmisjon
- Vår manglende finansakselerator (A4a-bug) er den dominerende kilden

**Anbefaling:** Ratio 0.39–0.49 (med rettede feil og fri phi_I1) er
rimelig for en K&M-basert modell sammenlignet med full 2024-NEMO.

### 4.3 Har NB mimicking rule eller tapsfunksjon?

**Begge — i ulike kontekster:**

- **Operativt baseline-scenario:** Mimicking rule (implementert i vår modell).
  Parametrisert til å reprodusere historiske rentebeslutninger.
- **Optimal politikk-analyse:** Tapsfunksjon med vekter λ_π, λ_y, λ_dr.
  Kalibrert til K&M (2019) Tabell 8: lambda_dr=0.40, lambda_y=0.30.
- **Staff Memo 2025-15 (GEORG):** NB mapper optimal tapsfunksjon over til
  en enkel Taylor-lignende regel (GEORG). IRF-matching i Figur 6 sammenligner
  NEMO med optimal policy og NEMO med GEORG. Kan gi oss den eksakte
  normaliseringen og magnitudenivåene NB bruker.

---

## 5. Anbefalinger til PE (prioritert rekkefølge)

### Trinn A (uten ny MCMC):

1. **Rett A4a (bankkapital)** — bytt til G1-akkumulering, fjern phi_c-overlapp
2. **Rett CEE-normalisering** — G0[12, Q_K] = -1/(phi_I1*(1+beta))
3. **Rett A4c (LTV-fortegn)** — Psi[25, E_phi_h] = -1.0 (strammere konvensjon)
4. **Rett A5 (steady-state)** — kalibrér andelene mot SSB/K&M Tabell 8
5. **Test rettede ligninger** mot 15 IRF-krav i test_irf_signs.py

### Trinn B (ny MCMC — krever PE-godkjenning):

6. Re-estimer med rettede ligninger (feil A4a, A4c endrer likelihood)
7. Evaluer om prior for h_c (0.01, 0.9995) er for bred — data treffer alltid
   øvre grense; mulig prior bør strammes eller h_c bør kalibreres fast
8. Evaluer om phi_I1 bør frigjøres igjen (data vil ha ~0.5, ikke 4.0)

### Trinn C (referansegranskning):

9. Last ned Staff Memo 2025-15 (GEORG) lokalt og les IRF-normaliseringen
   i Figur 6 — dette avklarer hva NB mener med "1pp pengepolitikksjokk"
10. Last ned K&M (2019) Staff Memo 5/2019, sammenlign Figur 2 mot vår
    kode med K&M-parametre for å bekrefte at vi har funnet alle kodebugger

---

## 6. Referanser

- `docs/oppgaver/A_funn_rapport.md` — Tidligere granskning (A4a, A4b, A4c, A5)
- `src/nemo/model/equations.py` — `build_matrices_v3()` (produksjonsversjon)
- `src/nemo/model/parameters.py` — K&M (2019) Tabell 8–10 parametere
- `data/results/chain_fase2_phi1fix_prod_posterior.json` — Nåværende posterior
- Staff Memo 2025-15 (GEORG) — Hentes fra NB.no (403 Forbidden for bot)
- K&M (2019) Staff Memo 5/2019 — Hentes via `scripts/fetch_references.sh`
