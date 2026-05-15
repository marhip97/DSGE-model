# Fase 0.5 — Sluttrapport: Modellkvalitetssikring

**Dato:** 2026-05-15  
**Status:** Under avslutning  
**Neste fase:** Fase 1 (datainnhenting) og Fase 2 (reestimering med blokksampling)

---

## Sammendrag

Fase 0.5 ble introdusert for å adressere kjente svakheter i NEMO v3 før
datainnhenting og reestimering i full skala. Fem spor ble gjennomført.
Konklusjonen er at modellens kjernemekanikk er korrekt, men tre problemer
krever tiltak i Fase 2 — ingen krever PE-godkjenning alene.

---

## Gjennomførte leveranser

### Spor D — Testpakke (PR #13)

Opprettet komplett testpakke med 44 tester (40 pass, 4 xfail):

| Test | Status |
|------|--------|
| `test_solver.py` — BK-stabilitet, dimensjoner | ✅ 12 pass |
| `test_irf_signs.py` — 15 kvalitative IRF-krav | ✅ 13 pass, 2 xfail |
| `test_fevd_sum.py` — FEVD summer til ~100 % | ✅ 4 pass |
| `test_likelihood.py` — Kalman sanity, COVID-gap | ✅ 7 pass |
| `test_steady_state.py` — ressursbetingelse | ✅ 3 pass, 2 xfail |
| `.github/workflows/tests.yml` | ✅ CI aktiv |

xfail-tester dokumenterer kjente svakheter (ikke regresjoner).

### Spor B5 — NB Memo 3/2024 Benchmark (PR #14, oppdatert etter A4b)

Reproduserte IRF for pengepolitikkssjokk og sammenlignet mot NB Memo 3/2024 Figur 1.

Avvik etter A4b-rettelse:

| Variabel | q1 | q4 | q8 | Vurdering |
|----------|----|----|-----|-----------|
| Styringsrente | +1.00 pp | +0.60 pp | +0.20 pp | ✅ Riktig profil |
| BNP-gap | Korrekt sign | Korrekt | Korrekt | ✅ |
| KPI-inflasjon | Korrekt sign | Korrekt | Korrekt | ✅ |
| RER-gap | Ca. 2× for stort | 2× for stort | OK | ⚠ sigma_rp |

Resterende RER-avvik peker entydig på sigma_rp-overestimering (H3, Spor C3).

### Spor A4b — Mimicking rule-rettelse (PR #16)

Identifiserte og rettet to feil i `build_matrices_v3()` (ligning 20):

1. **Samtid π vs. lagget π:** G0[20, PI] i stedet for G1[20, PI_L]
   — Korrekt iht. K&M (2019) § 2.13 (framoverskuende Taylor)
2. **Lag-state bug:** G0[20, I_R_L] = −psi_R i stedet for G1[20, I_R_L] = psi_R
   — G1 på lag-tilstand ga 2-periodes oscillasjon (I_R_L(t-1) = I_R(t-2))

Effekt: rente-persistens gjenopprettet (T[I_R,I_R] ≠ 0), B5-benchmark dramatisk forbedret.

**Systemic lag-state bug dokumentert:** Rader 5 (W), 7 (H_W), 8 (H_NW), 11 (K),
12 (INV) har samme G1-på-lag-tilstand-problem. Krever separat analyse og
PE-godkjenning (A4a, A4c i Fase 2).

### Spor C8 — MCMC Mixing-diagnose (PR #17)

Kort MCMC-kjøring (5k burnin + 20k prod.) avdekket:

| Funn | Verdi | Tolkning |
|------|-------|----------|
| ACF(1) for rho-param | > 0.99 | Ekstremt treig kjede |
| ESS/n for rho-param | ~0.006 | 6× under kravet 0.02 |
| sigma_C–h_c korrelasjon | r = −0.811 | Nøkkelblokkering |
| PSRF > 1.10 | rho_Ys, rho_rp | Ikke-konvergert |

Anbefaling: **Blokksampling sigma_C/h_c** og **logit-reparametrisering** i Fase 2
(ingen PE-godkjenning nødvendig).

### Spor C1/C2/C7 — Prior vs. Posterior (denne PR)

Identifikasjonsstyrke og H1–H4-vurdering for h_c og psi_R:

**C7 — Identifikasjon:**
- Sterk id: h_c (0.012), sigma_i (0.053), psi_R (0.084)
- Svak id: rho_rp (1.09), rho_H (0.96), psi_Y (0.96), rho_A (0.90)

**C2 — H1–H4 konklusjon (indikativ):**
- **H4 (likelihood-rygg):** MEST SANNSYNLIG primær mekanisme for begge
  — std=0.0015 (h_c) og 0.010 (psi_R) ved øvre grense
- **H3 (manglende UIP-kanal):** STERK EVIDENS — sigma_rp=0.016 vs. K&M 0.006
- **H1 (genuint høy persistens):** KAN IKKE AVVISES
- **H2 (prior dominerer):** USANNSYNLIG alene

---

## Åpne spørsmål til PE

Følgende krever eksplisitt PE-godkjenning før handling:

| # | Spørsmål | Begrunnelse |
|---|----------|-------------|
| 1 | **Spor C3: Reestimering med sigma_rp=0.006 (fast)** | Vil avklare H3. ~2 timer MCMC. |
| 2 | **Spor A4a/A4c: Rette bank- og LTV-likninger** | Retter 5 systemic lag-state bugs (rader 5,7,8,11,12) + LTV-fortegn |
| 3 | **Prior-utvidelse h_c og psi_R** | Basert på C2-konklusjon. Avhenger av C3-resultat. |

---

## Anbefalinger for Fase 2

### Uten PE-godkjenning (kan startes umiddelbart)

1. **Blokksampling sigma_C/h_c** (r=−0.811): Implementer blokkvis proposal i
   `adaptive_mcmc_with_monitoring`. Estimert ESS-gevinst: 5–10×.
2. **Logit-reparametrisering** av h_c og psi_R: Transformer til
   `logit(h_c)` i sampler, tilbakestransformer i likelihood. Avdekker om
   posterior er genuint konsentrert ved grensen eller numerisk artefakt.
3. **Fase 1 datainnhenting** kan parallellkjøres med (1) og (2).

### Med PE-godkjenning (krever avklaring)

4. **C3 sigma_rp-eksperiment**: Fix sigma_rp=0.006, reestimer 16 parametre.
   Forventes å redusere RER-overreaksjon og muligens flytte h_c fra grensen.
5. **A4a/A4c likningsrettelser**: Rette systemic lag-state bug i 5 likninger.
   Mulig BK-instabilitet — krever grundig testing.

---

## Testpakke-status

```
pytest tests/ -v
44 tester: 40 pass, 4 xfail (strict)
```

xfail-tester og deres eskaleringsreferanse:

| Test | Avvik | Spor |
|------|-------|------|
| `test_09_tfp_bnp_opp` | TFP → BNP = −0.001 (neg.) | A + C4 |
| `test_15_ltv_laantakerkonsum_ned` | LTV → C_NW = +0.004 (feil fortegn) | A4c |
| `test_ressursbetingelse_uten_bolig` | sum = 0.84, avvik −0.16 | A5 |
| `test_ressursbetingelse_med_bolig` | sum = 0.94, avvik −0.06 | A5 |

---

## Referanser

- `docs/oppgaver/A_funn_rapport.md` — likningsgransking, A4a–A5
- `data/results/B5_avvik_tabell.md` — benchmark mot NB Memo 3/2024
- `data/results/C8_acf_rapport.md` — mixing-diagnose, ACF, IAT, korrelasjon
- `data/results/C2_C7_rapport.md` — prior vs. posterior, H1–H4, identifikasjon
- `docs/MODEL.md` — komplett modellbeskrivelse
