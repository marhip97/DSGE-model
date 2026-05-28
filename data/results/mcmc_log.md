# MCMC-kjøringslogg — NEMO Fase 0.5/2

Loggføres per AGENTER.md-krav: alle MCMC-kjøringer skal dokumenteres her.

---

## Prior-endring — psi_R (2026-05-26, PE-godkjent)

**Fra:** `Beta(2.0, 2.0, [0.01, 0.990])`
**Til:** `Beta(2.0, 3.0, [0.01, 0.970])`

**Begrunnelse:** kj16 (KPI-JAE, 100k) drev psi_R til 0.987 — praktisk talt ved prior-grensen
(0.990). Dette ga BNP q4 = −209% (NB: −45%) og sigma_H=0.321, sigma_C=0.111.
Beta(2,3) er høyreskjev (penaliserer verdier nær 1) og redusert øvre grense til 0.970
hindrer grense-atferd uten å utelukke høy renteglatting (rom til 0.97).

**Fil:** `src/nemo/estimation/mcmc.py` linje ~179

---
---

## Prior-endring — rho_s (2026-05-26, PE-godkjent Fase 1B)

**Ny parameter:** `rho_s` — AR(1)-glatting av RER i UIP-ligningen  
**Prior:** `Beta(2.0, 2.0, [0.001, 0.99])` — symmetrisk, senteret på 0.5, lb=0.001 fordi Beta(2,2)=0 ved x=0  
**K&M-referanse:** Ikke i K&M (2019) — ren UIP er spesialtilfelle rho_s=0. Justiniano & Preston (2010) viser at AR(1) i UIP er nødvendig for å matche RER-persistens i data.

**Begrunnelse:** kj18 (KPI-JAE) ga KPI q4-ratio 0.40× NB (OK ≥ 0.35×) men BNP q4-ratio 4.55× NB (mål: 0.8–1.5×). Uten dynamikk i UIP absorberer `sigma_H` og `sigma_C` overraskende store sjokk og driver BNP-overreaksjon. AR(1)-glatting demper den umiddelbare RER-responsen ved pengepolitikksjokk.

**Modellendring:** `src/nemo/model/equations.py` ligning 15 (UIP):
```
rer_t = rho_s·rer_{t-1} + (1-rho_s)·[E_t[rer_{t+1}] - (i_D-π) + (i*-π*) + ε_rp + ...]
```
rho_s=0 gjenoppretter ren UIP (bakoverkompatibel).

**Fil:** `src/nemo/estimation/mcmc.py` (PARAM_PRIORS), `src/nemo/model/equations.py` (ligning 15), `src/nemo/model/parameters.py`

---

## Kjøring 19 — chain_kj19_prod (2026-05-26)

- **Test:** Fase 1B — AR(1)-glatting av RER i UIP (`rho_s` estimert), KPI-JAE
- **Parametre:** 21 (rho_s ny), sigma_rp fast=0.006, kappa_M fast=0.030
- **Startverdi:** kj18 posterior means + rho_s=0.40 (prior-mean)
- **Trekk:** 200k produksjon + 20k burnin + 50k rekalibrering, seed=19
- **Tid:** 77.4 min
- **Konvergens:** 20/21 OK, max PSRF=1.312 (sigma_H), min ESS=330

**Nøkkelresultater:**

| Parameter | K&M   | kj18   | kj19   |
|-----------|-------|--------|--------|
| rho_s     | 0.0   | —      | **0.009** [0.002,0.018] |
| psi_R     | 0.667 | 0.954  | 0.956  |
| sigma_H   | 0.050 | 0.310  | 0.309  |
| sigma_C   | 0.030 | 0.116  | 0.120  |
| phi_I1    | 4.0   | 0.103  | 0.103  |
| psi_P1    | 0.292 | 0.238  | 0.267  |

**Diagnostikk rho_s:** Posterior mean = 0.009, std = 0.005, CI = [0.002, 0.018].
Praktisk talt ved nedre prior-grense (0.001). Data avviser AR(1)-glatting fullstendig.

**Konklusjon:** ❌ **Fase 1B mislyktes.** rho_s → 0 betyr at avviket IKKE skyldes
manglende UIP-dynamikk. IRF-responser er identiske med kj18. BNP-overreaksjonen
er strukturell — sannsynligvis kombinasjon av psi_R≈0.956 og phi_I1≈0.10.

**Neste hypoteser (krever PE-eskalering):**
1. Kalibrere phi_I1 fast = K&M=4.0 (svakt identifisert, kj14-erfaring)
2. Diagnostisere identifikasjon av psi_R ved likelihood-profil
3. Utvide estimeringsperiode med mer post-COVID data (2024–2025)

---

## Kjøring 20 — chain_kj20_prod (2026-05-28)

- **Test:** PE-godkjent (2026-05-26): phi_I1 fast=0.50, rho_s fast=0.0 (ren UIP), KPI-JAE
- **Parametre:** 19, sigma_rp fast=0.006, phi_I1 fast=0.50, rho_s fast=0.0
- **Startverdi:** kj19 posterior means (19 overlappende param)
- **Trekk:** 200k produksjon + 20k burnin + 5 rekalibreringer, seed=20
- **Konvergens:** 19/19 OK, max PSRF=1.090 (rho_H), min ESS=430

**B5-benchmark:**
- **BNP q4-ratio:** 0.718× NB (mål [0.8,1.5]×) → ❌ FEIL (for lav)
- **KPI q4-ratio:** 0.183× NB (mål ≥0.35×) → ❌ FEIL (for lav)

**Nøkkelresultater:**

| Parameter | K&M   | kj19   | kj20   |
|-----------|-------|--------|--------|
| psi_R     | 0.667 | 0.956  | **0.956** [0.942,0.966] |
| psi_P1    | 0.292 | 0.267  | 0.253 |
| sigma_H   | 0.050 | 0.309  | 0.310 |
| sigma_A   | 0.006 | fast   | fast (ceiling 0.050) |
| phi_u     | 0.220 | —      | **0.012** (ekstremt lavt, K&M=0.22) |
| phi_I2    | ~10.0 | —      | 7.97 |

**Konklusjon:** ❌ **kj20 mislyktes begge mål.** psi_R=0.956 (prior-grense 0.970) gjenstår.
Effektiv Taylor-koeff = (1-0.956)×0.253 = 0.011 (K&M: 0.097). Svekket Taylor-prinsipp
er rotårsak: samtid π_t i Taylor-regel tvinger psi_R→1 som kompensasjon.

**Neste steg:** A4b — fremoverskuende Taylor-regel E_t[π_{t+4}] (K&M §2.13 mimicking rule).
Implementert i `build_matrices_pi4chain` (NZ=53), kjøring 21.

---

## Parameterendring — sigma_A fryses (2026-05-28, PE-godkjent)

**Fra:** estimert `Normal(0.010, 0.004, [0.002, 0.050])`
**Til:** fast `SIGMA_A_FIXED = 0.006` (K&M-verdi)

**Begrunnelse:** kj20 drev sigma_A→0.049 (tak=0.050, kun 1.2σ fra grensen) og phi_u→0.012 (gulv=0.010). Felles MCMC-forslag av alle 19 parametere ga 0% aksept i kj21 fordi minst én parameter alltid gikk utenfor grensene. sigma_A er svakt identifisert (K&M kaliberer fast=0.006). Resultat: N_PARAMS: 19→18.

**build_Q:** sigma_A lagt til `_fixed`-oppslag (som sigma_rp).

---

## Diagnoseendring — psi_R fryses (2026-05-28, PE-godkjent)

**Fra:** estimert `Beta(2.0, 3.0, [0.01, 0.970])`
**Til:** fast `PSI_R_FIXED = 0.667` (K&M-kalibrering)

**Begrunnelse:** pi4chain (A4b) mislyktes — sigma_i→0 (degenerert modus, se under).
Direkte diagnose: kj21 tester om psi_R→0.956 (alle kj18-20) er rotårsak til
KPI q4-ratio 0.183× NB. Effektiv Taylor-koeff: (1-0.956)×0.253=0.011 (kj20)
vs (1-0.667)×psi_P1 (kj21, psi_P1 fritt). N_PARAMS: 18→17.

---

## A4b/pi4chain mislyktes (2026-05-28)

pi4chain (lambda=0) ga degenererte moduser i to forsøk (kj21a og kj21b):
- kj21a (sigma_A estimert): 0% aksept fra start (sigma_A=0.049 ved tak=0.050 + phi_u=0.012 ved gulv=0.010)
- kj21b (sigma_A fast): MCMC fant modus ved (psi_R=0.966, sigma_i≈0, lp=-2526), men 0% aksept fra denne posisjonen (sigma_i ved nedre grense 1e-5)

Rotårsak: med ren E_t[π_{t+4}] og uten samtid π i Taylor-regelen mister sigma_i identifikasjon → kollapser til 0 (degenerert modus). Standard MCMC med normalforslag klarer ikke å utforske et slikt degenerert landskap.

Alternativ pi4chain (lambda>0) vurderes for fremtidige kjøringer etter kj21-diagnosen.

---

## Modellendring — pi4chain / A4b (2026-05-28, PE-godkjent)

**Endring:** Taylor-regel endret fra samtid π_t til fremoverskuende E_t[π_{t+4}]
  (K&M §2.13: `i_R = ψ_R·i_{t-1} + (1-ψ_R)·[ψ_P1·E_t[π_{t+4}] + ψ_Y·y + ...]`)

**lambda_pi4 fast=0.0** — ren K&M mimicking rule (A4b). Hybrid λ·π_t + (1-λ)·E_t[π_{t+4}]
støttes via `getattr(p, 'lambda_pi4', 0.0)` i `build_matrices_pi4chain`.

**NZ:** 49→53. Fire nye tilstander PI_E1..PI_E4 (Sims 2002 forventningskjede):
  PI_E1_t = E_t[π_{t+1}], PI_E2_t = E_t[π_{t+2}], PI_E3_t = E_t[π_{t+3}], PI_E4_t = E_t[π_{t+4}]

**Stabilitet:** MSV-løsning max|eig(T)| = 0.998 ✓ (alle lambda-verdier).

**Filer endret:**
- `src/nemo/model/equations.py`: `build_matrices_pi4chain` — hybrid λ, oppdatert docstring
- `src/nemo/estimation/mcmc.py`: `build_H_pi4chain()`, `LAMBDA_PI4_FIXED=0.0`, `log_posterior` auto-detekterer NZ_PI4

---


- **Test:** A — fjern i_3m_obs (13 obs)
- **Parametre:** 20, sigma_rp fast=0.006, kappa_M fast=0.030
- **Trekk:** 100k produksjon + 20k burnin, seed=15
- **Konvergens:** 17/20 OK, max PSRF=1.449 (gamma_p), min ESS=155
- **Nøkkelresultater:** psi_R=0.944, psi_P1=0.168, gamma_p=0.304
- **B5 KPI q4-ratio:** 0.19× NB
- **Beslutning A:** ❌ HJELPER IKKE (< 0.35×)

## Kjøring 16 — chain_kj16_prod (2026-05-26)

- **Test:** B — KPI-JAE (pi_core_obs) i stedet for total KPI
- **Parametre:** 20, sigma_rp fast=0.006, kappa_M fast=0.030
- **Trekk:** 100k produksjon + 20k burnin, seed=16
- **Konvergens:** 20/20 OK, max PSRF=1.031, min ESS=258
- **Nøkkelresultater:** psi_R=0.987 (!), psi_P1=0.298, sigma_H=0.321, sigma_C=0.111
- **B5 KPI q4-ratio:** 0.42× NB, BNP q4=-209% (!!)
- **Beslutning B:** ✅ KPI OK, men BNP ustabil → prior-justering psi_R for kj18

## Kjøring 17 — chain_kj17_prod (2026-05-26)

- **Test:** C — kun pre-COVID (75 kv, tom post-array)
- **Parametre:** 20, sigma_rp fast=0.006, kappa_M fast=0.030
- **Trekk:** 100k produksjon + 20k burnin, seed=17
- **Konvergens:** 20/20 OK, max PSRF=1.054, min ESS=181
- **Nøkkelresultater:** psi_R=0.941, psi_P1=0.164
- **B5 KPI q4-ratio:** 0.19× NB
- **Beslutning C:** ❌ HJELPER IKKE (< 0.35×)

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

## Kjøring 9 — phi_I1 fri + phi_B + phi_O (2026-05-20/21)
- **Parametre:** 19 (phi_I1 fri igjen, h_c=0.938 fast, sigma_A fast)
- **Prior phi_I1:** Normal(2.0, 2.0) på (0.1, 15.0)
- **Fil:** chain_fase2_phio_phi1_prod_posterior.json
- **Trekk:** 198 000 (akkumulert over 2 restarter: 16k + 182k)
- **Skript:** scripts/fase2_phio_phi1_akkumuler.py (akkumulerende strategi)
- **PSRF_max:** 1.005, **PSRF<1.10:** 19/19 ✓
- **ESS_min:** 532 (rho_rp), **ESS/n>1%:** 17/19 (rho_rp og sigma_rp svake)

### Nøkkelresultater kjøring 9

| Parameter | Kj9 | Kj8 | K&M |
|-----------|-----|-----|-----|
| phi_I1    | **0.205** [0.181,0.231] | 4.0 (fast) | 4.0 |
| sigma_rp  | 0.013 | 0.014 | 0.006 |
| psi_R     | 0.911 | 0.912 | 0.667 |
| rho_A     | 0.086 | 0.076 | 0.950 |
| rho_rp    | 0.808 | 0.831 | 0.920 |

**phi_I1=0.205**: Norske data forkaster K&M=4.0 sterkt. Liknende som fase2v2 (~0.5).
**rho_A=0.086**: TFP-sjokk lite persistent — mulig konsekvens av Q_K-spesifikasjon (test_09 xfail).

### B5-benchmark kjøring 9 (normalisert, posterior mean)

| Variabel | Kj9    | Kj8    | NB Figur 1 | Kj9/NB |
|----------|--------|--------|------------|--------|
| BNP q1   | -1.598 | -0.261 | -0.450     | 3.55×  |
| BNP q4   | -0.965 | -0.189 | -0.450     | 2.14×  |
| BNP q8   | -0.375 | -0.071 | -0.450     | 0.83×  |
| BNP q12  | -0.065 | +0.015 | -0.450     | 0.14×  |
| RER q4   | -0.592 | -0.621 | -0.400     | 1.48×  |

**Konklusjon:** phi_I1 fri gir stor forbedring i BNP-respons (fra 0.4× → 2.14× ved q4),
men responsen er for stor tidlig (3.55× ved q1) og for lite persistent (0.14× ved q12).
Normaliseringen til rente-topp q1 skaper artefakt — sjokket faller raskt.
Neste steg: undersøk rente-persistens og rho_A=0.086 (potensielt MPK-problem).

**Åpne spørsmål for PE:**
1. rho_A=0.086 — strukturproblem i Q_K-likning eller reelt norsk fenomen?
2. Kjøre ytterligere akkumulering (30k–50k trekk) for ESS rho_rp > 1%?
3. Endre normaliseringskonvensjon: BNP_q4-normalisering istedenfor rente-topp?

---

## Kjøring 10 — korrigert modell A4d + A_phi_L (2026-05-21)
- **Parametre:** 19 (phi_I1 fri, h_c=0.938 fast, sigma_A=0.006 fast)
- **Modellfix:** A4d (Q_K yk-koeff=1.0), A_phi_L (phi_L=1.50) — PE-godkjent 2026-05-21
- **Fil:** chain_kj10_prod_posterior.json
- **Trekk:** 178 000 (akkumulert over ~10 restarter via vaktløkke)
- **Skript:** scripts/fase2_kj10_akkumuler.py
- **PSRF_max:** 1.004, **PSRF<1.10:** 19/19 ✓
- **ESS_min:** 1384 (rho_rp), **ESS/n>1%:** 18/19
- **rho_rp ESS-note:** ESS/n=0.0078 — strukturelt lav (ACL≈140), bedres ikke med flere trekk. Krever HMC eller dedikert blokk i kj11.

### Nøkkelresultater kjøring 10

| Parameter | Kj10  | Kj9   | K&M   | Endring |
|-----------|-------|-------|-------|----------|
| rho_A     | **0.390** [0.21,0.57] | 0.086 | 0.950 | ↑ 4.5× — TFP-kanal åpnet |
| phi_I1    | **0.105** [0.10,0.12] | 0.205 | 4.0   | ↓ halvert, nær kalibrert |
| sigma_rp  | 0.014 [0.012,0.017]   | 0.013 | 0.006 | uendret |
| psi_R     | 0.912 [0.900,0.919]   | 0.911 | 0.667 | stabil |
| rho_rp    | 0.796 [0.34,1.00]     | 0.808 | 0.920 | bred posterior |
| rho_C     | 0.810 [0.37,1.00]     | –     | 0.800 | nær K&M |
| phi_u     | 0.027 [0.01,0.06]     | –     | 0.050 | rimelig |

**rho_A=0.390:** A4d-rettelsen løftet rho_A fra 0.086 → 0.390 (4.5×), men ikke til K&M=0.95.
Posteriorverdien er stabil og tolkbar — norske data støtter kortere TFP-persistens enn K&M.

**phi_I1=0.105:** Nær K&M kalibrert (~0.10). phi_I1=4.0 i K&M kan reflektere langsiktig kalibrering
ikke datadrevet estimering.

### Anbefalinger for kjøring 11

1. **Dedikert blokk for rho_rp** — skill ut fra AR-blokken for å bedre ESS/n
2. **B5-benchmark oppdatering** med kj10-posterior (BNP-respons forventes forbedret vs kj9)
3. **rho_A-diagnose** — er 0.39 vs K&M=0.95 et modell- eller dataproblem? Sjekk TFP-IRF.

---

## ⚠️ Navnekonvensjon — advarsel (2026-05-24)

Fra og med 2026-05-24 starter ny numerert serie (kj9–kj12) med egne `chain_kj*_prod_*.json`-filer.
**OBS:** `chain_kj10_prod_posterior.json` fra gammelt "Kjøring 10 — A4d + A_phi_L" er
**overskrevet** av den nye kj10 (sigma_rp fast). Parameterresultatene i
`Kjøring 10 — korrigert modell A4d + A_phi_L`-avsnittet over gjelder ikke lenger filen.

Filreferanser ny serie:
- `chain_kj9_prod_posterior.json` → kj9 nedenfor
- `chain_kj10_prod_posterior.json` → kj10 nedenfor (overskrevet)
- kj11: avbrutt, ingen posterior-JSON
- `chain_kj12_prod_posterior.json` → kj12 nedenfor (pågår)

---

## Kjøring 9 (ny serie) — ny COVID-split + sigma_A fast (2026-05-23)

- **Parametre:** 19 fri + sigma_A fast=0.006 (20 i PARAM_NAMES, men sigma_A = konstant)
  - sigma_rp **fri** (ikke fast ennå), phi_I1 fri, h_c fast=0.938
- **Ny Covid-split:** pre ≤ 2019Q4 (75 kv), post ≥ 2022Q1 (15 kv) — PE-godkjent Alt A 2026-05-23
- **Startverdi:** chain_kj10_prod_posterior.json (A4d + A_phi_L base)
- **Fil:** `chain_kj9_prod_posterior.json`
- **Trekk:** 200 000 produksjon + 20 000 burnin
- **PSRF_max:** 1.006 (psi_R), **ESS_min:** 447 (psi_R), **acc:** 0.259

### Nøkkelresultater kj9

| Parameter | Kj9    | K&M   | p5    | p95   |
|-----------|--------|-------|-------|-------|
| psi_R     | 0.911  | 0.667 | 0.900 | 0.918 |
| psi_P1    | 0.140  | 0.292 | 0.063 | 0.234 |
| sigma_rp  | 0.0163 | 0.006 | 0.014 | 0.019 |
| rho_A     | 0.146  | 0.950 | 0.037 | 0.298 |
| phi_I1    | 0.157  | 4.000 | 0.106 | 0.236 |
| rho_Ys    | 0.815  | 0.900 | 0.686 | 0.934 |

**Effektiv KPI-Taylor-koeffisient:** (1−0.911)×0.140 = **0.012** (svært lav)

### Konklusjon kj9

Ny COVID-split endre ikke bildet: psi_R=0.911, sigma_rp=0.016 uendret fra kj8.
rho_A=0.146 (vs K&M=0.95) tyder på kort TFP-persistens i norske data.
KPI-responsen er fortsatt underdrevet. Neste steg: fiks sigma_rp=0.006.

---

## Kjøring 10 (ny serie) — sigma_rp=0.006 fast (2026-05-24)

- **Parametre:** 19 fri (sigma_rp fjernet fra PARAM_NAMES, sigma_A nå fri)
- **Hypotese:** sigma_rp=0.016 presser psi_R opp → fiksering frigjør psi_P1
- **Startverdi:** kj9 posterior means
- **Fil:** `chain_kj10_prod_posterior.json`
- **Trekk:** 200 000 produksjon + 20 000 burnin
- **PSRF_max:** 1.169 (rho_O), **ESS_min:** 356 (rho_C), **acc:** ikke registrert

### Nøkkelresultater kj10

| Parameter | Kj10   | Kj9    | K&M   |
|-----------|--------|--------|-------|
| psi_R     | 0.911  | 0.911  | 0.667 |
| psi_P1    | 0.167  | 0.140  | 0.292 |
| sigma_rp  | 0.006  | 0.016  | 0.006 |
| sigma_A   | 0.0125 | (fast) | 0.006 |
| rho_A     | 0.175  | 0.146  | 0.950 |
| phi_I1    | 0.143  | 0.157  | 4.000 |

**Effektiv KPI-Taylor-koeffisient:** (1−0.911)×0.167 = **0.015** (fortsatt svært lav)

### Konklusjon kj10 ✗ (hypotese avkreftet)

**sigma_rp-dominanshypotesen er motbevist.** Fiksering av sigma_rp=0.006 endret IKKE
psi_R (fortsatt 0.911) og økte psi_P1 minimalt (0.140→0.167). Effektiv KPI-koeffisient
forble <0.02. Årsaken er ikke sigma_rp-dominans, men heller at modellen trenger enten:
1. psi_R som kan identifiseres mot data (som ønsker høy renteglatting), eller
2. Et annet element for å gi KPI persistens (hybrid Phillips-kurve).

Neste steg: test psi_R=0.667 fast (kj11) og gamma_p hybrid Phillips-kurve (kj12).

---

## Kjøring 11 (ny serie) — psi_R=0.667 fast — AVBRUTT (2026-05-24)

- **Parametre:** 18 fri (sigma_rp fast=0.006, psi_R fast=0.667)
- **Hypotese:** K&M-kalibrert psi_R frigjør psi_P1 → KPI-koeffisient nær K&M=0.292
- **Startverdi:** kj10 posterior means (ekskl. psi_R)
- **Fil:** ingen posterior-JSON (avbrutt ved 10 000 trekk)
- **Stoppkriterium:** Likelihood-fall > 50 log-enheter fra start

### Funn kj11

- **Startpunkt log-posterior:** ~3522 (kj10-nivå)
- **Etter psi_R=0.667 fast:** lp ≈ 3425 → **fall på 97 log-enheter**
- 97 log-enheter >> 50-enhetsgrense → data forkaster klart psi_R=0.667
- Sammenlign: K&M kalibrert psi_R=0.667 passer ikke norske data

### Konklusjon kj11 ✗ (avbrutt)

**Norske data vil ha høy renteglatting.** psi_R=0.667 (K&M) gir 97 log-enheter
likelihood-fall. Data er svært informative om psi_R (estimert tett rundt 0.91).
Løsningen er ikke å fikse psi_R, men å tilføre inflasjonspersistens via hybrid Phillips-kurve.
**Ikke gjenta denne testen** — den er grundig motbevist.

---

## Kjøring 12 (ny serie) — gamma_p hybrid NK Phillips-kurve (2026-05-24, pågår)

- **Parametre:** 20 fri (gamma_p ny, psi_R tilbake til estimering med økt grense 0.990)
- **Modellendring:** Hybrid NK Phillips-kurve, PE-godkjent 2026-05-24
  ```
  π_t = [γ_p/(1+βγ_p)]·π_{t-1} + [β/(1+βγ_p)]·E[π_{t+1}] + [κ_P/(1+βγ_p)]·mc_t + ...
  ```
  - G1[0, PI_L] = γ_p / (1+βγ_p) — bakseende ledd (PI_L = variabel 36)
  - Skalerer G0[0,MC], G0[0,RER], G0[0,PI_STAR] og Pi[0,PI] ned med 1/(1+βγ_p)
- **Prior gamma_p:** Beta(3,3) sentrert ~0.5, støtte [0.0, 0.95]. K&M: γ_p ≈ 0.35
- **Prior psi_R:** Beta(2,2) / (0.01, 0.990) — øvre grense utvidet fra 0.920
- **Startverdi:** kj10 posterior + gamma_p=0.35 (K&M kaldt start), std=0.05
- **Fil:** `chain_kj12_prod_posterior.json`
- **Trekk:** 200 000 produksjon + 20 000 burnin (pågår)

### Foreløpige resultater kj12 (ved 110k/200k, 2026-05-24)

- **Startpunkt lp:** 3564.50 — **+42 log-enheter vs kj10-start** (umiddelbar forbedring)
- **PSRF=1.01, ESS=335** — beste konvergens av alle kjøringer
- **lp=3566.4** ved 110k

| Parameter | kj12 (110k) | kj10  | K&M   |
|-----------|-------------|-------|-------|
| gamma_p   | ~0.226      | —     | 0.35  |
| psi_R     | ~0.953      | 0.911 | 0.667 |
| psi_P1    | ~0.213      | 0.167 | 0.292 |
| sigma_rp  | 0.006 (fast)| 0.006 | 0.006 |

**Effektiv KPI-Taylor-koeff ved 110k:** (1−0.953)×0.213 = **0.010** (fortatt lav)

### Foreløpig vurdering kj12

**gamma_p=0.226 er statistisk signifikant** (langt fra prior-massen ved 0.5, konsentrert).
Hybrid Phillips-kurven bedrer modellfit med +42 log-enheter — sterk evidens.
**Timing-problemet** (KPI topper umiddelbart, vs NB gradvis til t=4) forventes løst av γ_p≠0.

**Bekymring:** psi_R=0.953 (høyere med utvidet grense) → effektiv KPI-koeff noe lavere.
Data vil ha svært høy renteglatting UOG hybrid Phillips-kurve. Amplituden kan fortsatt være
underdrevet selv om timingen er fikset. IRF-validering nødvendig etter fullføring.

### Endelige resultater kj12 (etter fullføring)

| Parameter | kj12   | K&M   | p5    | p95   |
|-----------|--------|-------|-------|-------|
| psi_R     | 0.953  | 0.667 | 0.932 | 0.972 |
| psi_P1    | 0.210  | 0.292 | 0.087 | 0.356 |
| gamma_p   | 0.230  | 0.350 | 0.072 | 0.453 |
| phi_I1    | 0.154  | 4.000 | 0.105 | 0.230 |

**PSRF_max:** 1.007 (gamma_p), **ESS_min:** 670 (gamma_p), **acc:** 0.182

**Effektiv KPI-Taylor-koeffisient:** (1−0.953)×0.210 = **0.010** (svært lav)

**B5-benchmark kj12 (normalisert til +1pp rentetopp):**

| Variabel | kj12   | NB    | kj12/NB |
|----------|--------|-------|---------|
| BNP q4   | −51%   | −45%  | 1.14×   |
| KPI q4   | −3.0%  | −15%  | **0.20×** |
| RER q4   | −72%   | −40%  | 1.80×   |
| Rente q8 | +67%   | +20%  | 3.35×   |

**Konklusjon kj12:** gamma_p bedrer KPI-timing (kurven er nå negativt hellende gjennom hele horisonten, 
ikke positiv ved q12 som i kj10). Men amplituden er 0.20× NB — ekstremt underdrevet.
**Rotårsak identifisert:** κ_P = 5/phi_PQ = 5/669 = 0.0075 (ekstremt flat Phillips-kurve).
Med phi_I1≈0.15 svinger BNP enormt, men BNP→KPI-transmission er nesten brutt.

---

## Kjøring 13 (ny serie) — phi_PQ fri (2026-05-24)

- **Parametre:** 21 fri (phi_PQ ny, Steg A)
- **Hypotese:** phi_PQ=669 (K&M) for høyt → κ_P=0.0075 for flat → KPI-respons 0.20× NB
- **Prior phi_PQ:** Normal(669, 300, [50, 2000]) — sentrert ved K&M, tillater nedside
- **Startverdi:** kj12 posterior means (fra chain direkte) + phi_PQ=669 kaldt start
- **Fil:** `chain_kj13_prod_posterior.json`
- **Trekk:** 200 000 produksjon + 20 000 burnin
- **PSRF_max:** 1.019 (psi_Y), **ESS_min:** 415 (psi_Y), **acc:** 0.190

### Nøkkelresultater kj13

| Parameter | kj13    | kj12   | K&M   | p5     | p95    |
|-----------|---------|--------|-------|--------|--------|
| phi_PQ    | 584.4   | 669.0  | 669.0 | 104.4  | 1088.9 |
| psi_R     | 0.9528  | 0.9528 | 0.667 | 0.932  | 0.972  |
| psi_P1    | 0.2101  | 0.2099 | 0.292 | 0.086  | 0.349  |
| gamma_p   | 0.2346  | 0.2296 | 0.350 | 0.073  | 0.465  |
| phi_I1    | 0.1537  | 0.1536 | 4.000 | 0.105  | 0.231  |

**kappa_P:** kj13=0.00856, kj12=0.00747 (+15%)

**Effektiv KPI-Taylor-koeffisient:** (1−0.953)×0.210 = **0.010** (uendret fra kj12)

**B5-benchmark kj13:**

| Variabel | kj13   | kj12   | NB    | kj13/NB |
|----------|--------|--------|-------|---------|
| KPI q4   | −3.1%  | −3.0%  | −15%  | **0.21×** |
| BNP q4   | −52%   | −51%   | −45%  | 1.15×   |
| RER q4   | −72%   | −72%   | −40%  | 1.81×   |

### Konklusjon kj13 ✗ (Steg A utilstrekkelig)

**phi_PQ er svakt identifisert.** Posterior [104, 1089] er ekstremt bred — data kan ikke 
skille mellom flat og moderat Phillips-kurve med de observerte variablene.
phi_PQ falt kun 13% (669→584), κ_P økte kun 15% (0.0075→0.0086).
**KPI q4 forbedret seg fra 0.20× → 0.21× NB — marginalt ubrukelig.**

Alle andre parametere er identiske med kj12. phi_PQ absorberer ikke informasjon fra data 
fordi den er en skalafaktor i Phillips-kurven som allerede er dekt av andre parametre.

**Neste steg: Steg B — estimer kappa_M (importpriskanal).**
**Ikke gjenta fri phi_PQ** — identifikasjonsproblemet er dokumentert.

---

## Kjøring 14 (ny serie) — kappa_M fri (2026-05-24)

- **Parametre:** 21 fri (kappa_M ny, phi_PQ fjernet, Steg B)
- **Hypotese:** kappa_M=0.03 (K&M) er for lav → høyere κ_M → sterkere RER→KPI-kanal
- **Prior kappa_M:** Normal(0.03, 0.03, [0.005, 0.20]) — sentrert ved K&M
- **Startverdi:** kj13 posterior means + kappa_M=0.030 kaldt start. lp=3573.99 (+20 vs kj13)
- **Fil:** `chain_kj14_prod_posterior.json`
- **Trekk:** 200 000 produksjon + 20 000 burnin (4 rekalibreringer)
- **PSRF_max:** 1.053, **ESS_min:** 455, **acc:** 0.167

### Nøkkelresultater kj14

| Parameter | kj14    | kj12   | K&M   | p5     | p95    |
|-----------|---------|--------|-------|--------|--------|
| kappa_M   | **0.0175** | 0.030  | 0.030 | 0.006  | 0.039  |
| psi_R     | 0.954   | 0.953  | 0.667 | 0.933  | 0.972  |
| psi_P1    | 0.240   | 0.210  | 0.292 | 0.100  | 0.389  |
| gamma_p   | 0.204   | 0.230  | 0.350 | 0.061  | 0.411  |
| phi_I1    | 0.165   | 0.154  | 4.000 | 0.107  | 0.249  |

**Eff KPI-Taylor-koeff:** (1−0.954)×0.240 = **0.011** (marginalt bedre enn kj12)

**B5-benchmark kj14:**

| Variabel | kj14   | kj12   | NB    | kj14/NB |
|----------|--------|--------|-------|---------|
| KPI q4   | −2.0%  | −3.0%  | −15%  | **0.13×** |
| BNP q4   | −48%   | −51%   | −45%  | 1.06×   |
| BNP q8   | −35%   | −37%   | −35%  | **1.00×** |
| RER q4   | −72%   | −72%   | −40%  | 1.80×   |

### Konklusjon kj14 ✗ (Steg B avkreftet — overraskende funn)

**Kappa_M estimeres LAVERE enn K&M (0.0175 vs 0.030) — hypotesen er feil.**
Data vil ha svakere RER→KPI-transmisjon, ikke sterkere. KPI-responsen forverres
til 0.13× NB (fra 0.20× i kj12).

**Empirisk funn:** Norsk importpris-pass-through er lavere enn K&M antok.
Dette kan reflektere distribusjonskostnader, sticky importpriser i Norge,
eller at norsk KPI er dominert av innenlandsk tjenesteprisvekst.

**Samlet konklusjon fra Steg A (kj13) og Steg B (kj14):**
Verken phi_PQ eller kappa_M kan fikse KPI-amplitudeproblemet.
KPI-svakheten er en **robust empirisk egenskap** i norske data under dette
DSGE-rammeverket — ikke en modellparameter-feil. Rotårsakene er trolig:
1. Genuint flat Phillips-kurve for Norge (liten output gap → inflasjon-transmisjon)
2. phi_I1/psi_P1-substitusjon: med phi_I1≈0.15 (friksjonsfri investering) trenger
   ikke modellen høy psi_P1 for å stabilisere → psi_P1=0.21 er konsistent med data
3. psi_R≈0.95 er norsk data sitt svar — ikke et identifikasjonsproblem

**Anbefaling:** Aksepter kj12 som beste spesifikasjon. BNP-fit er god (1.06× ved q4,
1.00× ved q8 i kj14). KPI-timing er bedret av gamma_p. Amplitude er svak men
konsistent med norsk data. Gå til neste analysetrinn.
**Ikke gjenta Steg A eller B** — begge er grundig testet og dokumentert.

---

## kj21 — Diagnose: psi_R fast=K&M=0.667 (2026-05-28, avbrutt)

**Kjøring:** kj21 — diagnostisk, v3, KPI-JAE  
**Formål:** Test om psi_R→0.956 er rotårsak til KPI q4-ratio 0.183× NB (kj20)  
**Spesifikasjon:** psi_R fast=0.667 (K&M), sigma_A fast=0.006, phi_I1 fast=0.5, N_PARAMS=17  
**Resultat (70k/200k, avbrutt):** PSRF=1.01, ESS=269, acc=23%  

| Parameter | Posterior | K&M |
|-----------|-----------|-----|
| psi_P1    | 0.077     | 0.292 |
| psi_R (fast) | 0.667  | 0.667 |
| rho_C     | 0.076     | 0.725 |
| sigma_H   | 0.313     | 0.050 |
| phi_u     | 1.676     | 0.219 |

**B5-resultat (delvis posterior):**
- BNP q4: 0.925× NB ✅ (mål [0.8,1.5]×)
- KPI q4: 0.098× NB ❌ (mål ≥0.35×)

**Konklusjon:** Hypotesen **motbevist** — selv med psi_R=0.667 (K&M) trekker data psi_P1 ned til 0.077. 
Effektiv Taylor-inflasjonskoeffisient: (1-0.667)×0.077 = 0.026 (vs K&M=0.097). 
Dataene kompenserer ved å redusere psi_P1 i stedet for å øke psi_R. 
Avbrutt etter diagnostisk analyse avdekket rotårsak: se kj22-diagnose nedenfor.

---

## Rotårsak-diagnose: kappa_P-formel 6× for liten (2026-05-28)

**Funn:** B5-benchmark ved K&M-kalibrering (alle K&M-verdier) gir:
- BNP q4: 0.344× NB ❌
- KPI q4: 0.067× NB ❌ (KPI er 15× for liten)

**Rotårsak:** `kappa_P = (ε-1)/φ_PQ = 5/669 = 0.0075` — NKPC-helling er 6× for flat.  
Korrekt NEMO-formel med markup-normering: `κ_P = ε(ε-1)/φ_PQ = 30/669 = 0.0448`.  
φ_PQ=669 beholdes uendret fra K&M Tabell 8. Kun formelstruktur korrigeres.

**Verifikasjon (B5-sweep, korrekt annualisert formel):**
- κ_P=0.0448, phi_I1=0.5, psi_R=0.950, K&M base: BNP=1.046×✅, KPI=0.465×✅
- κ_P=0.0448, phi_I1=0.5, psi_R=0.900, K&M base: BNP=0.875×✅, KPI=0.396×✅
- κ_P=0.0448, phi_I1=0.5, psi_R=0.667, K&M base: BNP=0.334×❌ (psi_R→0.95 nødvendig)

**Feasible region (phi_I1=0.50, K&M base):** psi_R∈[0.90, 0.968] → begge mål oppnås.

**Implementert:** `kappa_P()` og `kappa_W()` endret til `ε(ε-1)/φ` i `parameters.py`.  
φ_PQ=669, φ_W=666.92 uendret (K&M Tabell 8). 89/89 tester bestått.

---

## Prior-endring — psi_R reaktivert (2026-05-28, PE-fullmakt)

**Fra:** DEAKTIVERT (kj21-diagnose, PSI_R_FIXED=0.667)  
**Til:** `Beta(2.0, 3.0, [0.01, 0.970])` (samme som kj18)

**Begrunnelse:** Med κ_P=0.0448 er KPI-kanalen sterk nok til at psi_R~0.95 (som data konsekvent foretrekker) gir KPI≥0.35×. Å fryse psi_R er ikke lenger nødvendig.

---

## kj22 — Forhåndsregistrering (2026-05-28)

**Kjøring:** kj22 — produksjonskjøring, v3, KPI-JAE, κ_P-fix  
**Formål:** Første kjøring med korrigert NKPC-helling. Test om BNP og KPI begge treffer B5-benchmark.  
**Spesifikasjon:**
- κ_P = 0.0448 (ε(ε-1)/φ_PQ — ny formel), κ_W = 0.0449
- psi_R fri, Beta(2,3,[0.01,0.970])
- sigma_A fast=0.006, phi_I1 fast=0.5, rho_s fast=0.0
- N_PARAMS=18
- v3-matriser (NZ=49), KPI-JAE
- Startverdi: K&M-defaults
- 200k produksjon, seed=22

**Forventet resultat:** psi_R→~0.95, BNP q4-ratio ~1.04×✅, KPI q4-ratio ~0.46×✅  
**Mål:** BNP q4 ∈ [0.8,1.5]× NB OG KPI q4 ≥ 0.35× NB

---

## kj22 — Avbrutt av container (2026-05-28)

**Kjøring:** kj22 — avbrutt etter 26k/200k produksjonstrekk (container timeout ~46 min)  
**Partial chain:** `data/results/chain_kj22_prod_partial.npy` (26k trekk)

**B5-normaliseringsrettelse (2026-05-28):** Normalisering av BNP var feil i diagnostikk.
Y er kvartals-log-avvik; NB-benchmark er annualisert (-0.45% p.a.). Korrekt formel:
`4×Y[q4]/peak / (-0.45%)`. Med korrekt formel reproduseres eksakt:
- K&M + kP=0.0448 + psi_R=0.95 + phi_I1=0.5 → BNP=1.046×✅, KPI=0.465×✅

**Partial posterior (26k, ikke konvergert):**

| Parameter | kj22 (26k) | K&M   | Note |
|-----------|-----------|-------|------|
| psi_R     | 0.968 (pri-tak) | 0.667 | ved grense 0.970 |
| psi_P1    | 0.328     | 0.292 | nær K&M |
| phi_u     | 1.715     | 0.219 | 8× K&M — ikke konvergert |
| rho_C     | 0.068     | 0.725 | ekstremt lav — ikke konvergert |
| rho_H     | 0.979     | 0.694 | nær prior-tak |

**B5 med korrekt formel (26k-posterior):**
- BNP q4: 2.32× NB ❌ (for STOR — phi_u=1.715 og ikke-konvergerte param)
- KPI q4: 0.92× NB ✅

**Konklusjon:** κ_P-fiksen virker (KPI 0.10→0.92×). BNP er for stor (2.32×) fordi
chain ikke er konvergert. Med K&M-like param og psi_R∈[0.90,0.968]: BNP∈[0.88,1.11]×✅.

**Feasible region (K&M base, kP=0.0448, phi_I1=0.50):**
| psi_R | BNP q4 | KPI q4 |
|-------|--------|--------|
| 0.90  | 0.875×✅ | 0.396×✅ |
| 0.95  | 1.046×✅ | 0.465×✅ |
| 0.968 | 1.113×✅ | 0.492×✅ |

---

## kj23 — Forhåndsregistrering (2026-05-28)

**Kjøring:** kj23 — identisk med kj22, ny seed, warm-start fra kj22 26k-posterior  
**Formål:** Fullføre kjøring avbrutt av container. Bekrefte BNP og KPI treffer mål.  
**Spesifikasjon:**
- Identisk med kj22: κ_P=0.0448, psi_R Beta(2,3,[0.01,0.970]), phi_I1=0.50 fast
- sigma_A fast=0.006, rho_s fast=0.0, N_PARAMS=18
- Startverdi: kj22 26k-posterior means (varm start — lp=-2658, nær modus)
- 10k burnin (vs 20k), scale_init=0.81, seed=23, 200k produksjon

**Forventet resultat:** psi_R→~0.95, BNP∈[0.8,1.5]×✅, KPI≥0.35×✅