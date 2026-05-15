# Oppgave A — Likningstransparentgransking

**Spor:** A (Likningstransparens)
**Ansvarlig:** DSGE-økonom
**Fase:** 0.5 (aktiv)
**Prioritet:** Medium (parallelt med C-sporene)

---

## Bakgrunn

`src/nemo/model/equations.py` inneholder tre versjoner av likningssystemet:
- `build_matrices()` — v1, basisversjon
- `build_matrices_v2()` — med MC, Q_K, INV-rettelser
- `build_matrices_v3()` — med boligprislikning, mimicking rule, h_c-oppdatering

Hvert ledd skal kunne refereres til en ligning i Kravik & Mimir (2019) eller
NEMO komplett dokumentasjon (2019). Oppgaven er å gjøre denne sporingslinjen
eksplisitt og avdekke eventuelle avvik.

---

## Steg

### A1 — Gjennomgang av v1 mot K&M (2019)

For hver blokk (A–F) i `build_matrices()`:
1. Finn tilhørende K&M-ligning (seksjonsnummer og likning)
2. Sjekk koeffisienter og fortegn
3. Dokumenter eventuelle avvik eller forenklinger

### A2 — Gjennomgang av v2-rettelsene

Verifiser de tre rettelsene fra v2:
- Fix 1: `MC = σ̃·y - (1+φ_L/(1-α_K))·a - α_K/(1-α_K)·k_lag`
  → K&M seksjon 2.3, ligning for marginal kostnad
- Fix 2: `Q_K` med r_K-avkastningsledd (α_K · mc)
  → K&M seksjon 2.5, kapitalverdi-ligning
- Fix 3: `INV = (1/φ_I1)·q_K` med fremoverskuende justeringskostnader
  → K&M seksjon 2.5, investeringsdynamikk

### A3 — Gjennomgang av v3-tilleggene

Verifiser:
- Boligprislikning med b_sa/lambda_sa (Gelain et al. 2018, referert i K&M Tabell 8)
- Mimicking rule (K&M seksjon 2.13)
- h_c-oppdatering i Euler-likningene

### A4 — Spesifikke mistenkelige punkter

**A4a — Bankligning (linje ~26 i kode, ligning 26 i G0):**
```python
G0[26, NB] = 1.0
G0[26, I_R] = -p.phi_o
G0[26, B_NW] = -p.phi_o
G0[26, NB] += p.phi_c   # ← dette legger phi_c til G0[26,NB] som allerede er 1.0
```
Sjekk: Er `1.0 + phi_c` riktig koeffisient for nb i sin egen ligning?
Referanse: Gerali et al. (2010) for bankkapitaldynamikk.

**A4b — Mimicking rule (ligning 20):**
Koden bruker `PI_L` (lagget inflasjon) for ψ_P1-leddet:
```python
G1[20, PI_L] = (1.0 - psi_R) * psi_P1
```
K&M seksjon 2.13 spesifiserer **fremoverskuende** inflasjon E[π_{t+4}].
Spørsmål: Er bakoverblikkende erstatning en bevisst forenkling eller en feil?

**A4c — EPS_PHI_H symmetri (ligning 22–23):**
```python
G0[22, EPS_PHI_H] = -1.0   # i_L_W reduseres
G0[23, EPS_PHI_H] = -1.0   # i_L_NW reduseres
```
Sjekk: Er det riktig at et positivt LTV-sjokk (strammere LTV) **reduserer** utlånsrentene?
Forventet: strammere LTV → høyere risikopremie → høyere renter.

### A5 — Steady-state konsistenssjekk

Verifiser ressursbetingelsen numerisk:
```python
from nemo.model.parameters import Parameters as p
assert abs(p.CY + p.IY + p.GY + p.XY - p.MY - 1.0) < 0.01
assert abs(p.IHY - p.delta_H * ...) < 0.01
```

---

## Akseptansekriterier

- [ ] Hvert ledd i v3 har K&M-referanse (seksjon og ligning) i kodekommentar
- [ ] A4a: Bankligning bekreftet korrekt eller rettelse foreslått med PE-godkjenning
- [ ] A4b: Mimicking rule-valget (bakover vs. fremover) dokumentert med begrunnelse
- [ ] A4c: LTV-sjokk-retning bekreftet eller korrigert
- [ ] A5: Steady-state-sjekk passerer (kan legges inn i testpakken)

---

## Avhengigheter

- Krever: `docs/references/nemo_complete_documentation_2019.pdf`
  (last ned med `scripts/fetch_references.sh`)
- Endringer i `equations.py` krever:
  1. Commit-melding med K&M-referanse
  2. Alle 15 IRF-krav fortsatt bestått
  3. PE-godkjenning hvis endringen er materiell
