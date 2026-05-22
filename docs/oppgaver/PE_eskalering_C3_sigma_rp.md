# PE-eskalering C3 — sigma_rp fiksering

**Avsender:** Claude Code (STAT/DSGE)
**Mottaker:** Prosjekteier (PE)
**Dato:** 2026-05-18
**Status:** Krever PE-beslutning før re-estimering

---

## Sammendrag

B5-benchmark mot NB Memo 3/2024 viser at `sigma_rp=0.017` (estimert posterior)
gir IRF-responser 6–44× for store sammenlignet med NB. Med K&M-kalibrert
`sigma_rp=0.006` er BNP-ratio 1.8× og KPI-ratio 1.0× — vesentlig bedre.

**Foreslått tiltak:** Fiks `sigma_rp=0.006` (som `sigma_A`) og re-estimer de
resterende 18 parametrene. Krever ~2 timer MCMC.

---

## Evidensgrunnlag

### B5-benchmark (pengepolitikkssjokk +1pp, kv4)

| Variabel | Estimert sigma_rp=0.017 | K&M sigma_rp=0.006 | NB Memo | Vinner |
|----------|------------------------|---------------------|---------|--------|
| BNP      | −2.85%                 | −0.81%              | −0.45%  | K&M    |
| KPI      | −0.44%                 | −0.15%              | −0.15%  | **K&M (perfekt)** |
| RER      | −11.6%                 | −4.1%               | −0.40%  | K&M (begge for store) |
| Boligpris| −34.9%                 | −21.3%              | −0.80%  | K&M    |

### Årsaksmekanisme

`sigma_rp` er standardavviket til risikopremiesjokket (EPS_RP), som inngår direkte
i UIP-likningen. Stor `sigma_rp` betyr:

1. Risikopremiesjokk forklarer 22% av BNP-varians og 88% av RER-varians (FEVD)
2. UIP-kanalen amplifikerer alle sjokk som går gjennom realrenten (inkl. pengepolitikk)
3. Med `psi_R=0.842` holder renten seg høy i mange kvartaler → kumulativ RER-bevegelse
   blir enorm

`sigma_rp` har vært stabilt 0.017 på tvers av alle tre MCMC-kjøringer — det er
ikke en sampling-fluktuasjon, men en genuine posterior-konsentrering langt fra K&M.

### Sannsynlig forklaring på overestimering

UIP-likningen mangler finansiell friksjon (f.eks. kapitalstrømsregulering,
bankpremie separat fra risikopremie). Alt som ikke kan forklares av andre kanaler
absorberes av `sigma_rp`. Dette er Spor C3-hypotesen.

---

## Foreslåtte tiltak (tre alternativer)

### Alternativ A — Fiksere sigma_rp=0.006 (anbefalt)

**Handling:** Legg til `SIGMA_RP_FIXED = 0.006` i `mcmc.py`, fjern fra `PARAM_NAMES`
(N_PARAMS: 19 → 18). Re-estimer.

**Fordeler:**
- Direkte test av K&M-hypotesen
- Raskest vei til rimelig IRF-benchmark
- Konsistent med sigma_A-behandlingen

**Ulemper:**
- Tvinger en parameter som data ikke godtar — kan gi dårligere likelihood
- Krever validering mot pre-COVID data for å unngå modellmisfit

### Alternativ B — Strammere prior (konservativ)

**Handling:** Endre `sigma_rp` prior fra `inv_gamma(2, 0.0037)` til
`inv_gamma(5, 0.006)` — mer konsentrert rundt K&M-verdi.

**Fordeler:** Holder sigma_rp som fri parameter, gir data mer rom
**Ulemper:** Posterior vil trolig fortsatt ligge over 0.010 basert på likelihood

### Alternativ C — Strukturell UIP-utvidelse (lang sikt)

Legg til eksplisitt kapitalstrømsfriskjon i UIP (separat kanal fra risikopremie).
**Krever:** Modellandring + ny kalibrering + PE-godkjenning som modellendring.
**Anbefales ikke** i Fase 0.5 — utenfor scope.

---

## Anbefaling

**Alternativ A** for rask diagnose av sigma_rp-hypotesen. Hvis likelihood
faller dramatisk (> 50 log-enheter), er det evidens for at modellen genuint
trenger høy sigma_rp og Alternativ C er nødvendig på sikt.

---

## Ressursbruk

- Implementering: ~30 min
- MCMC: ~2 timer (18 parametre, enklere sampler enn 19)
- Analyse: ~1 time (B5-benchmark + FEVD)

---

## Tillegg: psi_R prior-utvidelse

Uavhengig av sigma_rp-beslutningen: posterior `psi_R=0.842` treffer den nye
prior-grensen 0.85. Hvis PE ønsker å la data tale fritt for psi_R, anbefales
utvidelse til (0.01, 0.92). Dette er en lavrisiko-endring som kan bundles med
sigma_rp-kjøringen.

---

## PE-beslutning som trengs

☐ Godkjenn **Alternativ A** (fiks sigma_rp=0.006, re-estimer 18 param)  
☐ Godkjenn **Alternativ B** (strammere prior, re-estimer 19 param)  
☐ Godkjenn **psi_R prior-utvidelse** til (0.01, 0.92) bundlet med ovenfor  
☐ Avvis — videre diagnose nødvendig
