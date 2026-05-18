# Fase 2 — Sluttanalyserapport

**Dato:** 2026-05-17  
**Chain:** phi_I1-fix (19 param), 5-blokks RWMH + logit-reparam, 160,000 trekk  
**PSRF_max:** 1.0018  
**ESS_min:** 2861 (1.79%)  

## 1. Parametertabell

| Parameter | K&M | 5-blokk | **phi_I1-fix** | std | p05 | p95 | ESS/n% | PSRF |
|-----------|-----|---------|---------------|-----|-----|-----|--------|------|
| rho_A | 0.804 | 0.0909 | **0.8590** | 0.0529 | 0.7644 | 0.9357 | 2.08% | 1.000 |
| rho_C | 0.725 | 0.7927 | **0.7980** | 0.2146 | 0.3345 | 0.9982 | 1.79% ⚠ | 1.002 |
| rho_O | 0.874 | 0.3988 | **0.3659** | 0.1056 | 0.1877 | 0.5410 | 2.05% | 1.002 |
| rho_Ys | 0.783 | 0.8218 | **0.8190** | 0.0720 | 0.6950 | 0.9322 | 4.52% | 1.001 |
| rho_rp | 0.737 | 0.7952 | **0.8029** | 0.2136 | 0.3405 | 0.9988 | 1.88% ⚠ | 1.000 |
| rho_H | 0.694 | 0.8028 | **0.7990** | 0.2141 | 0.3390 | 0.9984 | 3.07% | 1.000 |
| sigma_C | 0.030 | 0.0161 | **0.0150** | 0.0025 | 0.0110 | 0.0192 | 3.29% | 1.000 |
| sigma_O | 0.079 | 0.0889 | **0.1009** | 0.0087 | 0.0873 | 0.1159 | 3.89% | 1.000 |
| sigma_Ys | 0.011 | 0.0058 | **0.0059** | 0.0008 | 0.0048 | 0.0073 | 4.26% | 1.001 |
| sigma_rp | 0.006 | 0.0163 | **0.0167** | 0.0015 | 0.0143 | 0.0193 | 3.81% | 1.001 |
| sigma_i | 0.000 | 0.0003 | **0.0003** | 0.0000 | 0.0002 | 0.0003 | 4.49% | 1.000 |
| sigma_P | 0.003 | 0.0055 | **0.0057** | 0.0005 | 0.0050 | 0.0066 | 4.27% | 1.000 |
| sigma_H | 0.050 | 0.1455 | **0.1487** | 0.0126 | 0.1293 | 0.1706 | 4.05% | 1.000 |
| psi_R | 0.666 | 0.9616 | **0.9637** | 0.0107 | 0.9449 | 0.9797 | 6.43% | 1.000 |
| psi_P1 | 0.292 | 0.2835 | **0.2793** | 0.0866 | 0.1391 | 0.4255 | 6.25% | 1.000 |
| psi_Y | 0.242 | 0.2555 | **0.2551** | 0.0468 | 0.1774 | 0.3323 | 4.35% | 1.000 |
| h_c | 0.938 | 0.9869 | **0.9876** | 0.0022 | 0.9838 | 0.9911 | 3.60% | 1.000 |
| phi_I2 | 8.000 | 11.6610 | **0.9360** | 0.3756 | 0.5397 | 1.6223 | 5.54% | 1.000 |
| phi_u | 0.219 | 0.6970 | **0.3892** | 0.0749 | 0.2710 | 0.5173 | 8.89% | 1.000 |
| phi_I1 (fast) | 4.000 | — | **4.0000** | — | — | — | fast | fast |

⚠ = ESS/n < 2% (rho_A, rho_C, rho_rp: genuint bred posterior, ikke sampler-feil)

### Nøkkelfunn

- **h_c = 0.9876** (p95=0.9911) — innenfor prior-grense 0.9995 ✓ (logit-reparam)
- **psi_R = 0.9637** — høy renteglatting; data foretrekker nær øvre grense
- **rho_A = 0.8590** (K&M: 0.804) — teknologisjokk nær IID i norske data
- **phi_I1 = 4.0000** (K&M: 4.0) — fiksert til K&M-verdi (PE-godkjent 2026-05-17)
- **sigma_rp = 0.0167** (K&M: 0.006) — risikopremie 2.7× høyere enn K&M

## 2. IRF — Pengepolitikksjokk (+1pp annualisert rente)

Horisont 1–20 kvartaler. Enheter: %-avvik fra SS (BNP, RER, boligpris), pp annualisert (KPI, rente).

| Kvartal | BNP | KPI (ann.) | Rente (ann.) | RER | Boligpris |
|---------|-----|-----|-----|-----|-----|
|       1 | -0.065 [-0.07,-0.06] | -0.040 [-0.04,-0.04] | +1.000 [+1.00,+1.00] | -0.261 [-0.26,-0.26] | -0.261 [-0.26,-0.26] |
|       2 | -0.067 [-0.07,-0.07] | -0.039 [-0.04,-0.04] | +0.960 [+0.94,+0.98] | -0.251 [-0.26,-0.25] | -0.409 [-0.41,-0.40] |
|       3 | -0.069 [-0.07,-0.07] | -0.038 [-0.04,-0.04] | +0.922 [+0.88,+0.96] | -0.241 [-0.25,-0.23] | -0.490 [-0.50,-0.48] |
|       4 | -0.070 [-0.08,-0.07] | -0.037 [-0.04,-0.03] | +0.885 [+0.83,+0.93] | -0.232 [-0.24,-0.22] | -0.529 [-0.55,-0.51] |
|       5 | -0.071 [-0.08,-0.06] | -0.036 [-0.04,-0.03] | +0.849 [+0.78,+0.91] | -0.223 [-0.24,-0.20] | -0.545 [-0.57,-0.51] |
|       6 | -0.072 [-0.08,-0.06] | -0.035 [-0.04,-0.03] | +0.815 [+0.73,+0.89] | -0.214 [-0.23,-0.19] | -0.545 [-0.58,-0.50] |
|       7 | -0.072 [-0.08,-0.06] | -0.034 [-0.04,-0.03] | +0.782 [+0.68,+0.87] | -0.206 [-0.23,-0.18] | -0.537 [-0.58,-0.48] |
|       8 | -0.071 [-0.08,-0.06] | -0.033 [-0.04,-0.03] | +0.750 [+0.64,+0.85] | -0.198 [-0.22,-0.17] | -0.523 [-0.58,-0.46] |
|       9 | -0.070 [-0.09,-0.06] | -0.032 [-0.04,-0.03] | +0.720 [+0.60,+0.83] | -0.190 [-0.22,-0.16] | -0.507 [-0.57,-0.44] |
|      10 | -0.069 [-0.09,-0.06] | -0.031 [-0.04,-0.03] | +0.690 [+0.56,+0.81] | -0.182 [-0.21,-0.15] | -0.490 [-0.56,-0.41] |
|      11 | -0.068 [-0.09,-0.05] | -0.030 [-0.04,-0.02] | +0.662 [+0.53,+0.80] | -0.175 [-0.21,-0.14] | -0.472 [-0.55,-0.39] |
|      12 | -0.066 [-0.09,-0.05] | -0.028 [-0.04,-0.02] | +0.635 [+0.49,+0.78] | -0.168 [-0.20,-0.13] | -0.454 [-0.54,-0.37] |
|      13 | -0.064 [-0.09,-0.05] | -0.027 [-0.03,-0.02] | +0.609 [+0.46,+0.76] | -0.161 [-0.20,-0.12] | -0.436 [-0.53,-0.35] |
|      14 | -0.062 [-0.09,-0.05] | -0.026 [-0.03,-0.02] | +0.584 [+0.43,+0.74] | -0.154 [-0.20,-0.11] | -0.419 [-0.52,-0.32] |
|      15 | -0.061 [-0.09,-0.04] | -0.025 [-0.03,-0.02] | +0.560 [+0.41,+0.73] | -0.148 [-0.19,-0.11] | -0.402 [-0.50,-0.30] |
|      16 | -0.059 [-0.09,-0.04] | -0.024 [-0.03,-0.02] | +0.537 [+0.38,+0.71] | -0.142 [-0.19,-0.10] | -0.386 [-0.49,-0.29] |
|      17 | -0.057 [-0.08,-0.04] | -0.023 [-0.03,-0.02] | +0.514 [+0.36,+0.69] | -0.136 [-0.18,-0.09] | -0.370 [-0.48,-0.27] |
|      18 | -0.055 [-0.08,-0.04] | -0.022 [-0.03,-0.02] | +0.493 [+0.33,+0.68] | -0.131 [-0.18,-0.09] | -0.355 [-0.47,-0.25] |
|      19 | -0.053 [-0.08,-0.04] | -0.022 [-0.03,-0.01] | +0.473 [+0.31,+0.66] | -0.125 [-0.17,-0.08] | -0.341 [-0.46,-0.24] |
|      20 | -0.051 [-0.08,-0.03] | -0.021 [-0.03,-0.01] | +0.453 [+0.29,+0.65] | -0.120 [-0.17,-0.08] | -0.327 [-0.45,-0.22] |

## 3. FEVD (horisont 20 kv) — andel av varians

| Sjokk | BNP | KPI | RER | Boligpris |
|-------|---|---|---|---|
| TFP | 1.1% | 0.5% | 0.0% | 0.4% |
| Konsum | 8.1% | 0.0% | 0.0% | 0.0% |
| Pris | 0.5% | 98.0% | 10.4% | 35.9% |
| Olje | 64.6% | 0.2% | 0.0% | 0.5% |
| Ettersp. | 6.3% | 0.0% | 0.0% | 0.1% |
| Risikopremie | 19.3% | 1.2% | 89.2% | 0.4% |
| Pengepol. | 0.1% | 0.0% | 0.3% | 3.3% |
| Bolig | 0.0% | 0.0% | 0.0% | 59.4% |

## 4. B5-benchmark mot NB Memo 3/2024 Figur 1

| Variabel | NB-topp | NB-kvartal | Modell-topp | Modell-kvartal | Ratio |
|----------|---------|-----------|-------------|---------------|-------|
| BNP | -0.60% | kv5 | -0.072% | kv6 | 0.12 ⚠ |
| KPI (ann.) | -0.40% | kv6 | -0.040% | kv1 | 0.10 ⚠ |
| RER | +1.50% | kv1 | -0.120% | kv20 | -0.08 ⚠ |

## 5. Konklusjon

phi_I1 fiksert til K&M=4.0 (PE-godkjent 2026-05-17). 19 frie parametre estimert.
PSRF_max=1.0018 (alle < 1.05). ESS_min=2861 (1.79%).

**Viktigste effekter av phi_I1-fix:**
- rho_A gjenopprettet mot K&M (0.859 vs K&M 0.804)
- phi_u redusert mot K&M (0.389 vs K&M 0.219)
- PSRF og ESS (17/19 over 2%) vesentlig forbedret

**B5-benchmark forverret, ikke forbedret (ratio 0.12 vs 0.26 for 5-blokk).**

### Kritisk funn: Modellspesifikasjonsproblem

Sjekk mot K&M-kalibrering viser at modellen gir BNP-respons=-0.066% (peak kv1)
selv ved originale K&M-parametre (h_c=0.938, sigma_rp=0.006, phi_I1=4.0).
NB Memo 3/2024 Figur 1 viser BNP-respons ≈ -0.60% (ratio ≈ 0.11).

**Konklusjon: svak pengepolitikktransmisjon er et modellkodeingsproblem (Spor A),
ikke et estimerings- eller parameteriserings-problem.** Parameterendringer alene
kan ikke rette dette.

RER viser feil fortegn i begge tilfeller (poster og K&M-kalibrering), noe som
tyder på konvensjonsproblem eller manglende UIP-dynamikk i likningstabellen.

### Gjenstående svakheter

- 2 parametre (rho_C, rho_rp) har ESS/n ~1.8% (marginalt under 2%-kravet)
- h_c=0.988 og psi_R=0.964 fortsatt ved øvre del av prior-intervallet
- sigma_rp=0.017 (K&M: 0.006) — risikopremie dominerer FEVD (BNP 19%, RER 89%)

### Anbefalt neste steg (krever PE-godkjenning)

Spor A (likningstransparentgransking): gjennomgå `build_matrices_v3()` mot K&M (2019)
for å identifisere feilen i pengepolitikktransmisjon. Spesielt:
- Mimicking rule-implementering (framoverskuende vs. bakoverblikkende)
- UIP-likning og RER-konvensjon
- Kalibreringstest: kjør KM-parametre og sammenlign med K&M IRF-figurer