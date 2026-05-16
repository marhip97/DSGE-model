# C4 — sigma_A identifikasjonsvurdering

**Dato:** 2026-05-16  
**Kjede:** chain_fase2v2_prod_posterior.json (100k trekk, 20 param)

## Analysemetode

Beregnet log-posterior ved å variere `sigma_A` (holdt fast i estimering) og `rho_A` 
(estimert parameter), alle andre parametre fiksert ved posterior mean.

## Resultat 1: sigma_A-sensitivitet (rho_A = 0.186, posterior mean)

| sigma_A | Log-posterior | Delta |
|---------|--------------|-------|
| 0.003 | 3512.4 | −51.9 |
| 0.004 | 3529.5 | −34.7 |
| 0.005 | 3547.3 | −17.0 |
| **0.006** | **3564.3** | **0.0** (gjeldende) |
| 0.008 | 3593.2 | +28.9 |
| 0.010 | 3614.2 | +50.0 |
| 0.012 | 3628.4 | +64.1 |
| 0.015 | 3640.2 | +75.9 |

**Likelihood stiger monotont og kraftig med sigma_A.** Differansen 0.006→0.012 er Δlp=+64, 
tilsvarende Bayes factor på e^64 ≈ 6×10^27. Data vil sterkt ha høyere sigma_A enn K&M-kalibrasjon.

## Resultat 2: (rho_A, sigma_A)-grid

| rho_A | sa=0.004 | sa=0.006 | sa=0.008 | sa=0.010 | sa=0.012 |
|-------|----------|----------|----------|----------|----------|
| 0.10 | 3529.8 | 3565.5 | 3595.2 | 3616.5 | 3630.7 |
| **0.18** | 3529.5 | **3564.4** | 3593.3 | 3614.4 | 3628.5 |
| 0.30 | 3529.1 | 3562.3 | 3589.9 | 3610.3 | 3624.3 |
| 0.50 | 3528.7 | 3558.2 | 3582.8 | 3601.5 | 3615.0 |
| 0.70 | 3527.7 | 3552.4 | 3573.5 | 3590.3 | 3603.2 |
| 0.80 | 3526.1 | 3548.3 | 3567.7 | 3583.8 | 3596.4 |

## Funn

**1. sigma_A er STERKT identifisert** — ikke svakt.  
Likelilhoodflatens kraftige stigning i sigma_A-retningen (Δlp=76 fra 0.006 til 0.015) viser at 
data er svært informative om TFP-sjokkets størrelse. sigma_A=0.006 er en bindende kalibreringsbetingelse 
som begrenser modellens passform betydelig.

**2. rho_A er SVAKT identifisert** (bekrefter C7).  
Variasjon i rho_A fra 0.10 til 0.80 gir bare Δlp≈4–17 (avhengig av sigma_A-nivå). 
Posterior rho_A=0.186 (mot K&M 0.804) er trolig en artefakt av fast lav sigma_A: 
når TFP-sjokket er for lite (sigma_A=0.006), kompenserer MCMC med lavere persistens (rho_A→0) 
for å matche observert TFP-varians.

**3. Ingen rho_A × sigma_A-rygg** — disse to er ikke observasjonsekvivalente langs en rygg. 
Data foretrekker høy sigma_A uavhengig av rho_A.

## C4-konklusjon

| Parameter | Vurdering | Anbefaling |
|-----------|-----------|------------|
| sigma_A | Sterkt identifisert; K&M-kalibrasjon (0.006) er suboptimal | Fri sigma_A i Fase 2 med prior N(0.010, 0.004), [0.001, 0.050] |
| rho_A | Svakt identifisert, men avhenger av sigma_A-verdien | Informativ prior sentrert på K&M 0.804 etter sigma_A er frigjort |

**PE-godkjenning:** Å fristille sigma_A endrer N_PARAMS fra 20 til 21 og representerer en 
prior-endring fra fast verdi til estimert — krever PE-godkjenning per AGENTER.md eskaleringsliste.  
Alternativt: endre fast kalibrering til sigma_A=0.010 uten å estimere (ingen PE nødvendig, men 
suboptimalt siden usikkerheten ikke fanges opp).
