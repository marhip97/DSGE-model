# C8 — MCMC Mixing og ESS-diagnose

Kjøring: 5,000 burn-in + 20,000 produksjon (Fase 0.5, Spor C8)

## 1. ESS og IAT per parameter

| Parameter | K&M-verdi | Posterior mean | IAT | ESS | ESS/n | PSRF | Vurdering |
|-----------|-----------|---------------|-----|-----|-------|------|-----------|
| rho_A      |    0.8040 |        0.7376 | 146.4 |   137 | 0.0068 | 1.098 | ⚠ lav ESS, ⚑ grense |
| rho_C      |    0.7250 |        0.8099 | 158.2 |   126 | 0.0063 | 1.012 | ⚠ lav ESS, ⚑ grense |
| rho_O      |    0.8740 |        0.4087 | 157.9 |   127 | 0.0063 | 1.001 | ⚠ lav ESS |
| rho_Ys     |    0.7830 |        0.8183 | 156.3 |   128 | 0.0064 | 1.152 | ⚠ lav ESS, ⚠ PSRF |
| rho_rp     |    0.7370 |        0.7686 | 174.9 |   114 | 0.0057 | 1.215 | ⚠ lav ESS, ⚠ PSRF, ⚑ grense |
| rho_H      |    0.6940 |        0.7925 | 165.9 |   121 | 0.0060 | 1.024 | ⚠ lav ESS, ⚑ grense |
| sigma_C    |    0.0300 |        0.0131 | 118.2 |   169 | 0.0085 | 1.001 | ⚠ lav ESS |
| sigma_O    |    0.0790 |        0.1078 | 158.6 |   126 | 0.0063 | 1.021 | ⚠ lav ESS |
| sigma_Ys   |    0.0110 |        0.0059 | 144.9 |   138 | 0.0069 | 1.059 | ⚠ lav ESS |
| sigma_rp   |    0.0060 |        0.0159 | 132.0 |   152 | 0.0076 | 1.037 | ⚠ lav ESS |
| sigma_i    |    0.0003 |        0.0003 | 130.0 |   154 | 0.0077 | 1.005 | ⚠ lav ESS |
| sigma_P    |    0.0030 |        0.0059 | 123.1 |   162 | 0.0081 | 1.012 | ⚠ lav ESS |
| sigma_H    |    0.0500 |        0.1543 |  33.8 |   592 | 0.0296 | 1.006 | OK |
| psi_R      |    0.6660 |        0.9627 | 134.1 |   149 | 0.0075 | 1.036 | ⚠ lav ESS |
| psi_P1     |    0.2920 |        0.2912 | 141.3 |   142 | 0.0071 | 1.005 | ⚠ lav ESS |
| psi_Y      |    0.2420 |        0.2629 | 144.4 |   139 | 0.0069 | 1.003 | ⚠ lav ESS |
| h_c        |    0.9380 |        0.9924 | 121.6 |   164 | 0.0082 | 1.006 | ⚠ lav ESS |

## 2. ACF ved lag 1, 5, 10, 20, 50

| Parameter | ACF(1) | ACF(5) | ACF(10) | ACF(20) | ACF(50) |
|-----------|--------|--------|---------|---------|---------|
| rho_A      | +0.991 | +0.957 | +0.918 | +0.852 | +0.710 |
| rho_C      | +0.993 | +0.967 | +0.938 | +0.887 | +0.782 |
| rho_O      | +0.994 | +0.972 | +0.945 | +0.896 | +0.775 |
| rho_Ys     | +0.995 | +0.974 | +0.949 | +0.899 | +0.766 |
| rho_rp     | +0.997 | +0.983 | +0.967 | +0.937 | +0.865 |
| rho_H      | +0.996 | +0.978 | +0.958 | +0.920 | +0.820 |
| sigma_C    | +0.983 | +0.925 | +0.863 | +0.759 | +0.562 |
| sigma_O    | +0.995 | +0.974 | +0.949 | +0.902 | +0.780 |
| sigma_Ys   | +0.992 | +0.962 | +0.927 | +0.861 | +0.696 |
| sigma_rp   | +0.991 | +0.957 | +0.916 | +0.834 | +0.638 |
| sigma_i    | +0.991 | +0.953 | +0.908 | +0.826 | +0.615 |
| sigma_P    | +0.987 | +0.936 | +0.880 | +0.783 | +0.588 |
| sigma_H    | +0.918 | +0.672 | +0.487 | +0.273 | +0.079 |
| psi_R      | +0.991 | +0.957 | +0.917 | +0.842 | +0.653 |
| psi_P1     | +0.991 | +0.957 | +0.916 | +0.845 | +0.689 |
| psi_Y      | +0.992 | +0.962 | +0.927 | +0.858 | +0.694 |
| h_c        | +0.983 | +0.923 | +0.862 | +0.764 | +0.580 |

## 3. Sterkeste korrelasjoner (|r| > 0.3)

| Par | Korrelasjon |
|-----|-------------|
| sigma_C–h_c | -0.811 |
| rho_Ys–sigma_Ys | -0.469 |
| rho_A–sigma_Ys | -0.335 |

## 4. Diagnose og anbefalinger

**Tregeste parametre (høyest IAT):**
- `rho_rp`: IAT=174.9, ESS/n=0.0057
- `rho_H`: IAT=165.9, ESS/n=0.0060
- `sigma_O`: IAT=158.6, ESS/n=0.0063
- `rho_C`: IAT=158.2, ESS/n=0.0063
- `rho_O`: IAT=157.9, ESS/n=0.0063

**Priorbegrensningstreff:**
- `rho_A`: mean=0.7376, std=0.19070 [0.0100, 0.9995] — 0.0% mot øvre, 0.0% mot nedre
- `rho_C`: mean=0.8099, std=0.18895 [0.0100, 0.9995] — 3.5% mot øvre, 0.0% mot nedre
- `rho_rp`: mean=0.7686, std=0.23119 [0.0100, 0.9995] — 5.8% mot øvre, 0.0% mot nedre
- `rho_H`: mean=0.7925, std=0.20330 [0.0100, 0.9995] — 3.9% mot øvre, 0.0% mot nedre

**Anbefalinger for Fase 2-sampler:**

Basert på diagnosen over:

1. **Blokksampling (prioritet HØY):** Identifiserte sterke korrelasjonsblokker mellom parametre bremser komponentvis RWMH kraftig. Implementer blokkvis proposal for korrelerte parametre.
2. **Reparametrisering:** Parametre med IAT > 50 indikerer sterk autokorrelasjon. Vurder logit-transformasjon for beta-parametre nær priorbegrensning.
3. **Prierjustering (krever PE-godkjenning):** Parametre nær øvre grense antyder at prieren er for trang eller at modellen mangler en kanal. Se Spor C2 for H1–H4-analyse.

> **C8-konklusjon for C2:** ESS-estimatene fra denne korte kjøringen er indikative. Konklusjoner om h_c og psi_R ved priorbegrensning kan ikke trekkes med sikkerhet før ESS/n > 0.02 i en full kjøring. C2 H1–H4-analyse bør starte etter blokksampling er implementert (Fase 2).
