# Oppgave C8 — Mixing-diagnose (ESS-problem)

**Spor:** C (Identifikasjon, posterior, mixing)
**Ansvarlig:** STAT
**Fase:** 0.5 (aktiv)
**Prioritet:** Høy — må leveres FØR C2 konkluderer

---

## Bakgrunn

MCMC-kjøringen (200 000 produksjonstrekkninger) møtte suksesskriteriet for
PSRF (1.0046 < 1.10), men **ESS_min = 662** er faktor 6 under kravet
ESS/n > 0.02 (tilsvarer 4 000 av 200 000 trekk).

Lav ESS betyr at punkt-estimatene for de tregeste parametrene (særlig h_c, psi_R)
har høy Monte Carlo-usikkerhet, og at konklusjoner fra C2 om "modellen trenger
veldig høy persistens" kan være upålitelige.

Nåværende ESS per parameter (fra `chain_v3_v2_posterior.json`):
```
sigma_C:  662  ← minimum (kritisk)
h_c:      667
rho_rp:   706
psi_R:    744
rho_C:    802
rho_A:    882
...
```

---

## Steg

### C8.1 — Autokorrelasjonsfunksjon (ACF)

For de 5 parametrene med lavest ESS:
1. Plot ACF opp til lag 500
2. Identifiser integrert autokorrelasjon τ (lag-sum til første negative AC)
3. ESS ≈ n / (1 + 2τ) — verifiser mot rapportert ESS

**Problem:** Full MCMC-kjede (ikke bare summary) trengs. Sjekk om
`data/raw/` inneholder `.npy`-filer fra siste kjøring.

### C8.2 — Posteriorkorrelasjonsmatrise

1. Beregn korrelasjonsmatrise fra chain (alle 17 parametre)
2. Identifiser sterke korrelasjoner (|corr| > 0.5) som hemmer komponentvis RWMH
3. Spesifikk test: korrelasjon mellom h_c og rho_C (hypotese: positiv korrelasjon
   bremser sampler)

### C8.3 — Anbefalinger for Fase 2

Ranger følgende alternativer basert på C8.1 og C8.2:

1. **Blokksampling** (lavest risiko): Grupper korrelerte parametre og foreslå
   dem simultant. Krever ingen ekstern pakke.
2. **Reparametrisering**: Transformer h_c → logit(h_c) for å unngå grenseproblem.
3. **HMC/NUTS** (høyest effektivitet, høyest kostnad): Krever gradienter og
   PE-godkjenning (eskaleringsregel).

### C8.4 — Implikasjon for C2

Dokumenter eksplisitt: Hvis h_c-ESS ≈ 667, hva er 95 % Monte Carlo-konfidensintervall
for punkt-estimatet 0.989? (Formel: ±1.96 × posterior_std / √ESS)

---

## Akseptansekriterier

- [ ] ACF-plot for de 5 tregeste parametrene
- [ ] Korrelasjonsmatrise dokumentert med klare klustre
- [ ] Klar anbefaling: hvilken sampler-endring for Fase 2?
- [ ] MC-usikkerhet for h_c og psi_R kvantifisert
- [ ] C2 kan ikke konkludere før C8 er levert

---

## Avhengigheter

- Krever full MCMC-kjede (`.npy`-fil) — ikke bare summary JSON
- Hvis kjede mangler: kort diagnostisk kjøring (10 000 trekk, ~5 min) er tilstrekkelig
  for ACF-mønster — krever PE-godkjenning
