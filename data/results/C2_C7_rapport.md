# C1/C2/C7 — Prior vs. Posterior og Identifikasjon

Basert på C8-kjeden: 20,000 produksjonstrekkninger (5 000 burn-in, startpunkt = posterior mean v3).

> **ESS-forbehold:** ESS/n ≈ 0.006 for de fleste parametre. Konklusjoner er **indikative**. Sikre resultater krever blokksampling + full kjøring i Fase 2 (ingen PE-godkjenning nødvendig for blokk).

## C1 — Prior vs. Posterior: alle parametre

| Parameter | K&M | Prior std | Post mean | Post std | Post [5%, 95%] | id_ratio | ESS/n |
|-----------|-----|-----------|-----------|----------|----------------|---------|-------|
| rho_A      | 0.8040 |   0.21149 |    0.7376 |  0.19070 | [0.322, 0.944] |    0.902 | 0.0033 |
| rho_C      | 0.7250 |   0.21149 |    0.8099 |  0.18895 | [0.429, 0.998] |    0.893 | 0.0022 |
| rho_O      | 0.8740 |   0.21149 |    0.4087 |  0.10038 | [0.236, 0.577] |    0.475 | 0.0026 |
| rho_Ys     | 0.7830 |   0.21149 |    0.8183 |  0.07394 | [0.685, 0.930] |    0.350 | 0.0024 |
| rho_rp     | 0.7370 |   0.21149 |    0.7686 |  0.23119 | [0.284, 0.999] |    1.093 | 0.0017 |
| rho_H      | 0.6940 |   0.21149 |    0.7925 |  0.20330 | [0.370, 0.998] |    0.961 | 0.0022 |
| sigma_C    | 0.0300 |   0.02780 |    0.0131 |  0.00256 | [0.009, 0.017] |    0.092 | 0.0062 |
| sigma_O    | 0.0790 |   0.06929 |    0.1078 |  0.01063 | [0.092, 0.128] |    0.153 | 0.0030 |
| sigma_Ys   | 0.0110 |   0.01208 |    0.0059 |  0.00073 | [0.005, 0.007] |    0.060 | 0.0027 |
| sigma_rp   | 0.0060 |   0.00724 |    0.0159 |  0.00146 | [0.014, 0.018] |    0.201 | 0.0032 |
| sigma_i    | 0.0003 |   0.00040 |    0.0003 |  0.00002 | [0.000, 0.000] |    0.053 | 0.0035 |
| sigma_P    | 0.0030 |   0.00545 |    0.0059 |  0.00048 | [0.005, 0.007] |    0.089 | 0.0058 |
| sigma_H    | 0.0500 |   0.07223 |    0.1543 |  0.01261 | [0.135, 0.177] |    0.175 | 0.0296 |
| psi_R      | 0.6660 |   0.12237 |    0.9627 |  0.01025 | [0.945, 0.978] |    0.084 | 0.0040 |
| psi_P1     | 0.2920 |   0.09935 |    0.2912 |  0.08802 | [0.147, 0.451] |    0.886 | 0.0038 |
| psi_Y      | 0.2420 |   0.05007 |    0.2629 |  0.04788 | [0.184, 0.336] |    0.956 | 0.0031 |
| h_c        | 0.9380 |   0.12178 |    0.9924 |  0.00151 | [0.990, 0.995] |    0.012 | 0.0060 |

## C7 — Identifikasjonsstyrke (post_std / prior_std)

Ratio nær 0 → posterior er mye smalere enn prior → sterk identifikasjon.  
Ratio nær 1 → posterior ≈ prior → svak identifikasjon (prior dominerer).

| Parameter | id_ratio | Vurdering |
|-----------|----------|-----------|
| h_c        |    0.012 | sterk id. |
| sigma_i    |    0.053 | sterk id. |
| sigma_Ys   |    0.060 | sterk id. |
| psi_R      |    0.084 | sterk id. |
| sigma_P    |    0.089 | sterk id. |
| sigma_C    |    0.092 | sterk id. |
| sigma_O    |    0.153 | sterk id. |
| sigma_H    |    0.175 | sterk id. |
| sigma_rp   |    0.201 | god id. |
| rho_Ys     |    0.350 | god id. |
| rho_O      |    0.475 | god id. |
| psi_P1     |    0.886 | ⚠ svak id. (prior dominerer) |
| rho_C      |    0.893 | ⚠ svak id. (prior dominerer) |
| rho_A      |    0.902 | ⚠ svak id. (prior dominerer) |
| psi_Y      |    0.956 | ⚠ svak id. (prior dominerer) |
| rho_H      |    0.961 | ⚠ svak id. (prior dominerer) |
| rho_rp     |    1.093 | ⚠ svak id. (prior dominerer) |

## C2 — h_c og psi_R ved priorbegrensning: H1–H4

**h_c:** mean=0.9924, std=0.00151, id_ratio=0.012, 9.2% ved øvre grense  
**psi_R:** mean=0.9627, std=0.01025, id_ratio=0.084, 3.0% ved øvre grense

### Hypotesevurdering

**H1 — Modellen trenger høy persistens (biologisk):**  
Både h_c og psi_R er langt over K&M-verdiene (0.938 og 0.666). Post-mean 0.992 og 0.963 kontra K&M antyder at data *krever* høyere persistens enn K&M-spesifikasjonen. Mulig forklaring: norsk konsum-smoothing er sterkere enn i originalparameterisering.

**H2 — Svak identifikasjon (prior dominerer):**  
h_c: SVAK H2 (posterior informativ)  
psi_R: SVAK H2 (posterior informativ)  
Lav id_ratio (< 0.10) betyr at posterior er mye smalere enn prior — data er informative om *nivået*, men nivået er ved grensen. Ikke en entydig H2-situasjon (prior dominerer vanligvis ved *flat* posterior).

**H3 — Manglende modellkanal:**  
sigma_rp=0.0159 (K&M 0.006) → STØTTER H3 (manglende UIP-kanal)  
Forhøyet sigma_rp absorberer risikopremiedynamikk som ideelt burde gå via UIP-leddet. Dette kan 'tvinge' høy h_c for å matche konsum-banen. Spor C3 (sigma_rp-eksperiment, PE-godkjenning) vil teste dette direkte.

**H4 — Likelihood-rygg langs h_c→1:**  
h_c: STØTTER H4 (std<0.003, ved grense)  
psi_R: STØTTER H4 (std<0.015, ved grense)  
Svært lav posterior-std ved øvre grense er et klassisk tegn på likelihood-rygg. Når h_c → 1, går a3_W = (1-h_c)/(σ(1+h_c)) → 0, noe som gjør lønnsblokken nær-singulær. Modellen 'liker' denne grensen. Reparametrisering (f.eks. log(1-h_c)) kan avsløre om dette er en ekte modus eller numerisk artefakt.

### Foreløpig C2-konklusjon (indikativ)

De fire hypotesene er **ikke gjensidig utelukkende**. Mest sannsynlig scenario:

- **H3 (manglende UIP/finanskanal)** er sannsynligvis den primære driveren: sigma_rp=0.016 vs. K&M 0.006 er sterk evidens for at modellen mangler en kanal. Spor C3-eksperiment (PE-godkjenning) vil gi svar.
- **H4 (likelihood-rygg)** støttes av ekstremt lav std ved grensen. Logit-reparametrisering i Fase 2 (ingen PE nødvendig) bør prioriteres.
- **H1** kan ikke avvises — norsk data kan genuint kreve høyere habit.
- **H2** er usannsynlig som *eneste* forklaring: posterior er for smal.

**Blokkering:** Sikre konklusjoner om H1 vs. H3 vs. H4 krever:
1. Blokksampling (sigma_C/h_c, r=−0.811) — Fase 2, ingen PE
2. Logit-reparametrisering av h_c og psi_R — Fase 2, ingen PE
3. C3 sigma_rp-eksperiment — krever PE-godkjenning

