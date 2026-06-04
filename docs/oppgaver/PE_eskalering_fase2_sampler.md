# PE-eskalering — Fase 2: Sampler-forbedring (blokksampling)

**Avsender:** STAT  
**Mottaker:** Prosjekteier (PE)  
**Dato:** 2026-06-03  
**Prioritet:** Høy — Fase 2-suksesskriterium ikke oppfylt

---

## Sammendrag

ESS/n-kravet (> 0.02) er ikke oppfylt i noen MCMC-kjøring. Etter kj44–kj46
har logit-reparametrisering (psi_R) forbedret mixing for enkeltparametre,
men det globale ESS/n er fortsatt ~0.005 — faktor 4 under krav. Problemet
er nå diagnostisert til to uavhengige årsaker:

1. **rho-kluster** (rho_A, rho_C, rho_O, rho_Ys, rho_rp): Korrelerte
   persistence-parametre i AR(1)-blokken — komponentvis RWMH mikser tregt
   pga. posteriorkorrelasjon.

2. **rho_s = 0.003 ± 0.003** (ESS=1044): Nærmest degenerert posterior —
   data sier at glatting av utenlandsk rente er neglisjerbar, men sampler
   bruker trekk på en parameter med nesten null varians og svak identifikasjon.

**Anbefalt tiltak:** Blokksampling for rho-klusteret + deaktivere/kalibrere rho_s fast.

---

## Nåværende ESS — kj46 (200k trekk)

| Parameter | ESS | ESS/n | Merknad |
|-----------|-----|-------|---------|
| rho_s | **1044** | 0.0052 | Degenerert — nær null |
| rho_C | 1062 | 0.0053 | Korrelert med rho_A |
| sigma_O | 1117 | 0.0056 | — |
| sigma_P | 1367 | 0.0068 | — |
| sigma_C | 1390 | 0.0070 | — |
| psi_R | 1770 | 0.0089 | Logit-reparam hjelper |
| rho_H | 3261 | 0.0163 | Nær krav |
| **Krav** | **4000** | **0.02** | Fase 2-mål |

Kj41 hadde ESS_min=620 (0.0031). Logit-reparam (kj44) hevet til 1077 (0.0054).
Kj46 (PLT) 1044 (0.0052). Forbedringen har flatnet ut — marginal avkastning
på ytterligere reparametrisering alene er lav.

---

## Diagnose: to uavhengige flaskehalser

### Flaskehals 1 — rho-kluster (rho_A, rho_C, rho_O, rho_Ys, rho_rp)

Burn-in fra kj46 viste konsistent sakte konvergens for nettopp disse fem:
```
Sjekk runde 0–10: IKKE OK — Problemer: ['rho_A', 'rho_C', 'rho_O', 'rho_Ys', 'rho_rp', 'rho_H']
```

**Årsak:** AR(1)-persistensparametrene deler lignende prior (Beta-form) og
er sannsynligvis positivt korrelert i posterior (felles sjokk-dynamikk).
Komponentvis RWMH tvinger dem til å bevege seg én og én — posterior-korrelasjonen
gir høy forkastningsrate og treg mixing.

**Løsning:** Blokksampling — foreslå hele {rho_A, rho_C, rho_O, rho_Ys, rho_rp}
simultant med en 5×5 blokkproposal-kovarians. Krever ingen ekstern pakke.

### Flaskehals 2 — rho_s (glatting utenlandsk rente)

kj46: rho_s = 0.003 ± 0.003, [0.000, 0.009]. I praksis degenerert ved null.
ESS=1044 er minimum fordi sampler bruker en full Gaussian-step for en parameter
med ~null variasjon i posterior.

**Alternativ A:** Kalibrér rho_s fast = 0.00 (K&M-verdi). Fjerner én parameter
fra estimering, eliminerer flaskehalsen uten strukturell endring. 19 estimerte
parametre.

**Alternativ B:** Beholdes estimert men diagnostiseres separat. ESS forbedres
ikke uten sampler-endring.

---

## Alternativer for PE-beslutning

### Alt. 1 — Blokksampling (anbefales)

**Hva:** Implementer blokkvis RWMH for rho-klusteret:
- Blokk A: {rho_A, rho_C, rho_O, rho_Ys, rho_rp} — 5 parametre simultant
- Blokk B: {psi_R} — beholder logit-reparam fra kj44
- Enkeltvis: resterende 14 parametre uendret

**Blokkproposal:** Adaptiv 5×5 kovarians fra kjede-historikk (samme adaptive
RWMH-prinsipp som nå, men i blokkdimensjon). Implementeres i
`src/nemo/estimation/mcmc.py` — ny funksjon `block_proposal_step()`.

**Fordeler:**
- Ingen ekstern avhengighet — ren numpy
- Lav risiko: faller tilbake til komponentvis om blokken feiler
- Forventet ESS-forbedring: 2–5× for rho-klusteret (litteraturanslag
  for korrekt spesifisert blokksampler på korrelerte parametre)
- Adresserer den diagnostiserte årsaken direkte

**Ulemper:**
- Krever PE-godkjenning per eskaleringsliste (bytte av sampler-arkitektur)
- ~0.5 dag implementering + testsuite
- Usikker effektstørrelse — kan kreve eksperimentell kj47

**Ressurser:** ~4 timer implementering + 2 timer MCMC (kj47, 200k trekk).

---

### Alt. 2 — HMC/NUTS

**Hva:** Erstatt RWMH med Hamilton Monte Carlo (gradienter via numerisk
differensiering eller automatisk differensiering).

**Fordeler:** Typisk 10–50× ESS-forbedring vs. RWMH for glatte posterior.

**Ulemper:**
- Tung implementering (gradient-evaluering av Kalman-filteret)
- Krever tunge avhengigheter (jax eller PyMC) — mot CLAUDE.md-regler
- ~3–5 dager implementering
- Høy risiko for numeriske problemer i DSGE-kontekst (ikke-glatt likelihood)

**Ressurser:** 3–5 dager + PE-godkjenning for ny avhengighet.

**Anbefaling:** Utsett til etter blokksampling er prøvd.

---

### Alt. 3 — Kalibrér rho_s fast (delvis løsning)

**Hva:** Sett rho_s = 0.00 fast (K&M-kalibrering, tab. 8). Eliminerer
flaskehals 2 umiddelbart uten sampler-endring.

**Fordeler:** ~1 time endring, ingen risiko, 20 → 19 estimerte parametre.

**Ulemper:** Adresserer ikke rho-klusteret (flaskehals 1). ESS vil stige
til ~1300–1500, men fortsatt under krav 4000.

**Anbefaling:** Gjør dette **uansett** — kombinerer med Alt. 1.

---

## Tilrådning

**Anbefalt kombinasjon: Alt. 3 + Alt. 1**

Steg 1 (Alt. 3, ~1 time): Kalibrér rho_s=0.00 fast. Eliminerer degenerert parameter.

Steg 2 (Alt. 1, PE-godkjent): Implementer blokksampling for rho-klusteret.
Kjøring kj47 med blokksampler + kj41 som warm start → forventer ESS/n > 0.02.

Steg 3: Hvis ESS/n > 0.02 i kj47 → Fase 2-suksesskriterium oppfylt.
Hvis ikke → diagnostiser videre (evt. HMC som steg 3, krever separat PE-godkjenning).

---

## PE-beslutning som trengs

☐ **Godkjenn Alt. 1** — blokksampling for rho-klusteret implementeres  
☐ **Godkjenn Alt. 3** — rho_s kalibreres fast = 0.00 (kombineres med Alt. 1 eller alene)  
☐ **Godkjenn Alt. 2** — HMC (utsatt anbefaling)  
☐ **Avvent** — akseptér ESS/n=0.005 og gå videre til oljepriskanal (Fase 3)

---

## Vedlegg: ESS-historikk på tvers av kjøringer

| Kjøring | Sampler | ESS_min | ESS/n | RMSE |
|---------|---------|---------|-------|------|
| kj41 | RWMH komponentvis | 620 | 0.0031 | 0.2771 |
| kj44 | RWMH + logit(psi_R) | 1077 | 0.0054 | 0.2924 |
| kj45 | RWMH + logit(psi_R) + AR(2) | — | — | — |
| kj46 | RWMH + logit(psi_R) + PLT | 1044 | 0.0052 | 0.3609 |
| **Krav** | — | **4000** | **0.02** | — |
| Forventet kj47 | Blokksampling + rho_s fast | ~3000–5000 | ~0.015–0.025 | ~0.27 |
