# C3 — sigma_rp-eksperiment: v3 vs. C3 (fast sigma_rp=0.006)

C3-kjøring: 5,000 burn-in + 30,000 produksjon. sigma_rp holdes fast til 0.006 (K&M-verdi).

## 1. Parameterjustering: v3 vs. C3

| Parameter | K&M | v3 mean | C3 mean | Endring | Merknad |
|-----------|-----|---------|---------|---------|---------|
| sigma_rp   | 0.0060 |  0.0161 |  0.0060 | -0.0101 | FAST = 0.006 |
| h_c        | 0.9380 |  0.9891 |  0.9870 | -0.0021 |  |
| psi_R      | 0.6660 |  0.9602 |  0.9515 | -0.0087 |  |
| rho_rp     | 0.7370 |  0.8019 |  0.7775 | -0.0244 | ⚑ fortsatt ved grense |
| sigma_C    | 0.0300 |  0.0198 |  0.0157 | -0.0041 |  |
| sigma_O    | 0.0790 |  0.1059 |  0.1059 | +0.0000 |  |
| psi_P1     | 0.2920 |  0.2620 |  0.2480 | -0.0141 |  |
| psi_Y      | 0.2420 |  0.2590 |  0.2713 | +0.0123 |  |
| rho_A      | 0.8040 |  0.7059 |  0.7201 | +0.0141 | ⚑ fortsatt ved grense |
| rho_C      | 0.7250 |  0.7916 |  0.8360 | +0.0444 | ⚑ fortsatt ved grense |

## 2. IRF-sammenligning: pengepolitikkssjokk (+1 pp rente)

| Variabel | Horisont | v3 | C3 | Endring |
|----------|----------|----|----|---------|
| BNP-gap   | q1 | -0.273 | -0.273 | -0.000 |
| BNP-gap   | q4 | -0.220 | -0.212 | +0.007 |
| BNP-gap   | q8 | -0.180 | -0.167 | +0.013 |
| BNP-gap   | q12 | -0.151 | -0.135 | +0.017 |
| KPI-infl. | q1 | -0.042 | -0.042 | -0.000 |
| KPI-infl. | q4 | -0.036 | -0.035 | +0.001 |
| KPI-infl. | q8 | -0.030 | -0.028 | +0.002 |
| KPI-infl. | q12 | -0.025 | -0.022 | +0.003 |
| Rente     | q1 | +1.000 | +1.000 | +0.000 |
| Rente     | q4 | +0.875 | +0.849 | -0.026 |
| Rente     | q8 | +0.734 | +0.684 | -0.050 |
| Rente     | q12 | +0.615 | +0.550 | -0.064 |
| RER-gap   | q1 | -1.044 | -1.044 | -0.000 |
| RER-gap   | q4 | -0.918 | -0.891 | +0.027 |
| RER-gap   | q8 | -0.770 | -0.718 | +0.052 |
| RER-gap   | q12 | -0.646 | -0.579 | +0.067 |

## 3. H3-vurdering

**h_c:** v3=0.9891 → C3=0.9870 (UNDER grense)
**psi_R:** v3=0.9602 → C3=0.9515 (UNDER grense)

**Konklusjon: H3 STØTTES** — Når sigma_rp ikke kan absorbere risikopremiedynamikk, flyttes h_c og/eller psi_R fra priorbegrensningen. Dette indikerer at den opprinnelige overestimeringen av sigma_rp tvang høy persistens-via-habit som kompensasjon.
