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

---

## Kjøring 4 — sigma_rp fast (C3-eksperiment, 2026-05-18)
- **Parametre:** 18 (sigma_rp fast=0.006, phi_I1 fast=4.0, h_c fri → traff 0.988)
- **Trekk:** ~100k (salvaged)
- **Funn:** sigma_rp=0.006*, psi_R steg til 0.911 (kompensasjon). BNP-ratio 8.5×.
- **Konklusjon:** Fiksering av sigma_rp løser ikke IRF — psi_R kompenserer.

## Kjøring 5 — h_c fast (C2 Alt A, 2026-05-18)
- **Parametre:** 18 (h_c=0.938 fast, phi_I1=4.0 fast, sigma_rp fri)
- **Fil:** chain_fase2_hcfix_prod_posterior.json
- **Trekk:** 60k (salvaged, container-grense)
- **PSRF_max:** 1.00, **ESS_min:** ~500
- **Funn:** sigma_rp=0.017, psi_R=0.912. BNP-ratio 10.2×.
- **Konklusjon:** h_c-fiksering endret ikke sigma_rp — kompensatorisk likevekt bekreftet.

## Kjøring 6 — RER utelatt (Alt. 4, 2026-05-19)
- **Parametre:** 18 (13 obs, ds_obs utelatt), h_c=0.938 fast, phi_I1=4.0 fast
- **Fil:** chain_fase2_norer_prod_posterior.json
- **Trekk:** 80k (salvaged)
- **PSRF_max:** 1.00
- **Funn:** sigma_rp STEG til 0.020 (opp fra 0.017). psi_R=0.912.
- **Konklusjon:** sigma_rp er ikke datadrevet via RER — det er strukturelt.

## Kjøring 7 — φ_B i UIP (Alt. 2, 2026-05-19)
- **Parametre:** 18 (phi_B=0.0016 i UIP-ligning), h_c=0.938 fast, phi_I1=4.0 fast
- **Fil:** chain_fase2_phib_prod_posterior.json
- **Trekk:** 120k (salvaged)
- **PSRF_max:** 1.00, **ESS_min:** ~800
- **Funn:** sigma_rp=0.017 (uendret), psi_R=0.912. lp forbedret 3404→3424.
- **Konklusjon:** φ_B bedrer modellfit men løser ikke sigma_rp-problemet.

## Kjøring 8 — φ_O i UIP (olje-valuta-kanal, 2026-05-20)
- **Parametre:** 18 (phi_O=0.15 og phi_B=0.0016 i UIP), h_c=0.938 fast, phi_I1=4.0 fast
- **Fil:** chain_fase2_phio_prod_posterior.json
- **Trekk:** 60k (salvaged, container-grense)
- **PSRF_max:** 1.004, **ESS_min:** 703, **ESS/n>2%:** 14/18
- **Funn:** sigma_rp=0.014 (↓ fra 0.017), psi_R=0.912. Delvis effekt.
- **B5-benchmark (normalisert, posterior mean):**

| Variabel | Kj8 | NB Figur 1 | Ratio |
|----------|-----|------------|-------|
| BNP q4   | -0.189 | -0.450 | 0.4× |
| RER q4   | -0.621 | -0.400 | 1.6× |
| KPI q4   | -0.025 | -0.150 | 0.2× |
| Rente q4 | +0.743 | +0.600 | 1.2× |

- **Konklusjon:** phi_O gir delvis sigma_rp-effekt men løser ikke B5. Sammenligning med
  fase2v2 (phi_I1 fri, BNP=-0.447≈NB) avslørte at phi_I1=4.0 (fast) er
  **hovedårsaken til for liten BNP-respons** (0.4×). PE godkjente å frigi phi_I1 i kjøring 9.

## Kjøring 9 — phi_I1 fri + phi_B + phi_O (2026-05-20) [PLANLAGT]
- **Parametre:** 19 (phi_I1 fri igjen, h_c=0.938 fast, sigma_A fast)
- **Prior phi_I1:** Normal(2.0, 2.0) på (0.1, 15.0)
- **Fil:** chain_fase2_phio_phi1_prod_posterior.json (ennå ikke kjørt)
- **Hypotese:** phi_I1 vil estimeres til ~0.5 (som fase2v2), BNP-respons vil treffe NB
- **Skript:** scripts/fase2_phio_phi1_production.py
