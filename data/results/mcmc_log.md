# MCMC-kjøringslogg — NEMO Fase 0.5/2

Loggføres per AGENTER.md-krav: alle MCMC-kjøringer skal dokumenteres her.

---

## Kjøring 1 — chain_fase2_reparam_prod (2026-05)
- **Parametre:** 20 (inkl. phi_I1 fri)
- **Trekk:** 200k × 2 kjeder
- **PSRF_max:** 1.002, **ESS_min:** 2861 (1.79%)
- **Funn:** phi_I1 estimert til ~0.5 (vs K&M=4.0), h_c=0.988, psi_R=0.960

## Kjøring 2 — chain_fase2_phi1fix_prod (2026-05-17)
- **Parametre:** 19 (phi_I1 fast=4.0, PE-godkjent)
- **Trekk:** 160k
- **PSRF_max:** 1.002, **ESS_min:** 2861 (1.79%)
- **Funn:** psi_R=0.964 (treffer prior-grense 0.990), sigma_rp=0.017

## Kjøring 3 — chain_fase2_postfix_prod (2026-05-18)
- **Parametre:** 19 (phi_I1 fast=4.0)
- **Trekk:** 140k etter burn-in (prosess avbrutt ved 175k/200k, salvaged)
- **PSRF_max:** 1.012, **ESS_min:** 724 (0.5%) — AR-blokk tregeste
- **ESS/n>2%:** 13/19
- **Modellfix:** A4a (bank), A4c (LTV-gjeld), CEE (Q_K), A5 (BNP-balanse), LTV-fortegn E3/E4
- **Prior-endring:** psi_R Beta(4,2)/(0.30,0.990) → Beta(2,2)/(0.01,0.85) (PE-godkjent)

### Postfix nøkkelresultater

| Parameter | Kjøring 3 | Kjøring 2 | K&M |
|-----------|-----------|-----------|-----|
| psi_R     | 0.842 **†** | 0.964 | 0.667 |
| h_c       | 0.988 **†** | 0.988 | 0.938 |
| psi_P1    | 0.108     | 0.279 | 0.292 |
| psi_Y     | 0.141     | 0.255 | 0.240 |
| phi_I2    | 4.73      | 0.936 | 8.000 |
| sigma_rp  | 0.017 **‡** | 0.017 | 0.006 |

**†** Treffer prior-grense  
**‡** Uendret på tvers av alle kjøringer — se C3-eskalering

### B5-benchmark (pengepolitikkssjokk +1pp, kv4)

| Variabel | Kjøring 3 | NB Memo 3/2024 | Ratio |
|----------|-----------|----------------|-------|
| BNP      | −2.85%    | −0.45%         | 6.3×  |
| KPI      | −0.44%    | −0.15%         | 2.9×  |
| RER      | −11.6%    | −0.40%         | 29×   |
| Boligpris| −34.9%    | −0.80%         | 44×   |

Med K&M sigma_rp=0.006: BNP-ratio = 1.8×, KPI-ratio = 1.0×.
Konklusjon: sigma_rp er den dominerende kilden til IRF-avvik. Se PE_eskalering_C3.md.

### Anbefalinger for neste kjøring

1. Fiks sigma_rp=0.006 (som sigma_A) — krever PE-godkjenning
2. Vurder psi_R prior-utvidelse til (0.01, 0.92) — data trykker mot 0.85-grensen
