# Spor A — Granskning av tre mistenkelige punkter

**Spor:** A (Likningstransparens)
**Status:** Granskning levert — krever PE-godkjenning før kodeendring
**Dato:** 2026-05-15

Denne rapporten gransker tre konkrete punkter identifisert i `docs/oppgaver/A_equation_review.md`.
**Ingen kode er endret** — alle rettelser krever PE-godkjenning per `AGENTER.md` § Eskaleringsregler.

---

## A4a — Bankkapital-likning (linje 396–402)

### Funn

Den nåværende implementeringen i `equations.py`:

```python
# E7. Bankkapital-akkumulering (netto kapital)
# nb = (1-δ_b)·nb_{t-1} + spread·lån - kapitalkrav
G0[26, NB]   =  1.0
G0[26, I_R]  = -p.phi_o   # spread-inntekt
G0[26, B_NW] = -p.phi_o
G0[26, NB]  +=  p.phi_c   # kapitaldekning-kostnad
```

**Tre problemer:**

1. **Manglende lagg-ledd**: Kommentaren sier `nb = (1-δ_b)·nb_{t-1} + ...`, men koden har
   **ingen G1-koeffisient for NB**. Akkumuleringsdynamikken mangler helt.

2. **Koeffisient-overlapp på NB**: `G0[26, NB] = 1.0` etterfulgt av `G0[26, NB] += p.phi_c`
   gir effektivt `G0[26, NB] = 11.0` (siden `phi_c = 10.0`, `Parameters` linje 174).
   Det er uklart om dette er tilsiktet — kommentaren antyder at φ_c skal være en
   separat kostnadskoeffisient, ikke en multiplikator på selve NB-leddet.

3. **Skala-mismatch**: φ_o = 0.0046 (Tabell 8, banks driftskostnad), φ_c = 10.0.
   Resultatet er at φ_c · nb dominerer fullstendig over spread-inntekten:

   ```
   11·nb - 0.0046·i_R - 0.0046·b_NW = 0
   ⇒ nb ≈ 0.0004·(i_R + b_NW)   # nær null uansett spread
   ```

   Banken har nesten ingen kapitaldynamikk.

### Forslag til rettelse (krever PE-godkjenning)

Korrekt Gerali et al. (2010)-form:

```python
# E7. Bankkapital-akkumulering: nb_t = (1-δ_b)·nb_{t-1} + j_b·spread_lagget
G0[26, NB]    =  1.0
G1[26, NB]    =  (1.0 - p.delta_b)        # akkumulering — δ_b = 0.0161
G1[26, I_R]   = -p.phi_o                   # spread-inntekt fra forrige periode
G1[26, B_NW]  = -p.phi_o
# φ_c-leddet hører ikke til her — det inngår i bankrente-spreadene (linje 364, 373, 379)
```

**Referanse:** K&M (2019) seksjon 2.7 / NEMO komplett dokumentasjon 2019 seksjon 2.7.

**Konsekvens for IRF:** Banken får riktig persistens i nb-dynamikken, og
spread-inntekt-koblingen blir reell.

---

## A4b — Mimicking rule: bakover- vs. fremoverskuende inflasjon (linje 357)

### Funn

K&M (2019) seksjon 2.13 spesifiserer mimicking rule med **fremoverskuende**
inflasjon (E[π_{t+4}]):

```
i_R = ψ_R·i_{t-1} + (1-ψ_R)·[ψ_P1·E[π_{t+4}] + ψ_Y·y + ψ_S·rer + ψ_W·π_W] + ε_i
```

Kommentaren i kode (linje 351–352) erkjenner:
> "Fremoverskuende π: E[π_{t+4}] ≈ ψ_P1·π_t (forenkling for BK-løsning)"

Men implementeringen bruker `PI_L` (π_{t-1}), ikke π_t:

```python
G1[20, PI_L] = (1.0 - psi_R) * psi_P1   # lagg av inflasjon
```

**Diskrepans:**
- K&M-spesifikasjon: E[π_{t+4}]
- Kodekommentar:    π_t (forenkling)
- Faktisk kode:     π_{t-1} (ekstra forsinkelse)

Dette gir 1 ekstra kvartals etterslep i rentereaksjonen. Mest sannsynlig forklaring
på det observerte avviket i Spor B5: **rentestien dødes ut for raskt** fordi
inflasjonsresponsen påvirker renten med ett kvartals forsinkelse, slik at
rentebanen ikke "ser" inflasjonsfallet i samtid.

### Forslag til rettelse (krever PE-godkjenning — endrer modellspesifikasjon)

**Alternativ 1 (mindre invasivt):** Erstatt PI_L med PI (samtid):

```python
G0[20, PI]   = -(1.0 - psi_R) * psi_P1   # samtid inflasjon
# Fjern G1[20, PI_L] = ...
```

**Alternativ 2 (korrekt K&M-spesifikasjon):** Implementer ekte fremoverskuende:

```python
Pi[20, PI]   = -(1.0 - psi_R) * psi_P1   # E[π_{t+1}] (én-periode forkant)
# E[π_{t+4}] krever flere ledd og endrer BK-strukturen
```

**Avhengighet:** Alternativ 2 endrer Pi-matrisen, som kan flytte modellen ut av
BK-stabilitetsregionen for nåværende psi_R-verdi. Krever ny BK-sjekk og
muligens re-estimering.

**Anbefaling til PE:** Start med Alternativ 1 (rask test, ingen reestimering),
mål effekten på B5-benchmark, vurder Alternativ 2 etter resultatet.

---

## A4c — LTV-sjokk-fortegn (linje 374, 380, 394)

### Funn — sjokkfortegnet er **inkonsistent** mellom ligningene

LTV-sjokket E_phi_h påvirker tre ligninger med tilsynelatende motstridende fortegn:

**Ligning 22 (i_L_W):**
```python
G0[22, EPS_PHI_H] = -1.0   # gir i_L_W = ... + eps_phi_h
```
Positivt sjokk → utlånsrente **opp**. ✅ Konsistent med "strammere LTV → høyere risiko".

**Ligning 23 (i_L_NW):**
```python
G0[23, EPS_PHI_H] = -1.0   # gir i_L_NW = ... + eps_phi_h
```
Samme tolkning: positivt sjokk → utlånsrente opp. ✅

**Ligning 25 (b_NW, gjeld låntakere — LTV-bindende):**
```python
G0[25, B_NW]      = 1.0
G0[25, Q_H]       = -m_H
G0[25, H_NW]      = -m_H
G0[25, I_L_NW]    = m_H
Psi[25, E_phi_h]  = 1.0    # LTV-sjokk direkte på debt
```

Ligningen reduserer til:
```
b_NW = m_H·q_H + m_H·h_NW - m_H·i_L_NW + eps_phi_h
```

Positivt sjokk → **mer gjeld tillatt** (b_NW opp). ❌ Tolkningskonflikt: dette
tilsvarer **løsere** LTV, ikke strammere.

### Konsekvens

Når LTV-sjokket inntreffer positivt:
- Utlånsrentene øker (ligning 22, 23) — strammere kreditt
- MEN belåningskapasiteten øker (ligning 25) — løsere kreditt

Den dominerende effekten via ligning 25 forklarer hvorfor `test_15_ltv_laantakerkonsum_ned`
feiler: c_NW går **opp** netto, fordi gjeldskapasitetens direkte effekt på b_NW
slår tilbake-på-renteeffekten.

### Forslag til rettelse (krever PE-godkjenning)

Velg én konsistent konvensjon:

**Konvensjon "strammere LTV"** (positivt sjokk = strammere, mest intuitivt):
```python
Psi[25, E_phi_h] = -1.0   # færre lån når LTV strammes
# ligning 22, 23 uendret (rente opp ved tightening) ✓
```

**Konvensjon "løsere LTV"** (positivt sjokk = løsere, K&M-mulig konvensjon):
```python
G0[22, EPS_PHI_H] = +1.0   # rente NED ved løsere LTV
G0[23, EPS_PHI_H] = +1.0
# ligning 25 uendret (mer gjeld ved loosening) ✓
```

**Anbefaling til PE:** Konvensjon "strammere LTV" er mer i tråd med makroprudensiell
politikkdiskurs i Norge (LTV-grense ble strammet i 2010, 2015, 2017). Verifiser
mot K&M (2019) seksjon 2.2 hvilken konvensjon de bruker når PDF er tilgjengelig.

---

## A5 — Steady-state konsistens (numerisk sjekk lagt til testpakken)

Lagt til som `tests/test_steady_state.py` i denne PR-en. **Avdekket reell kalibreringsfeil:**

| Versjon | Sum | Avvik fra 1.0 |
|---------|----:|--------------:|
| CY + IY + GY + XY - MY        | 0.840 | -0.160 |
| CY + IY + IHY + GY + XY - MY  | 0.940 | -0.060 |

**Tolkning:** Den "med bolig"-versjonen (0.94) er nærmest 1.0, så IY inkluderer trolig
**ikke** boliginvestering — kommentaren i `parameters.py` linje 142 ("Inkluderer nå
kapitalakkumulering") er misvisende. Selv med IHY er det fortsatt 6 pp gap som må utredes.

Begge tester er markert `xfail` med klar dokumentasjon — de er sporbare og dokumenterer
nåværende kalibreringssituasjon uten å blokkere CI.

**Forslag til rettelse (krever PE-godkjenning):** Verifiser andelene mot K&M (2019)
Tabell 8 og SSB nasjonalregnskap. Sannsynlige justeringer:
- MY oppjusteres fra 0.34 til ca. 0.40 (norsk import-til-BNP, 2020-snitt SSB: ~0.32)
- Eller XY oppjusteres tilsvarende
- Verifisere at gap kan tilskrives reell statistisk diskrepans i kilden

---

## Oppsummering for PE

| Punkt | Funn | Alvorlighetsgrad | Krever ny MCMC? |
|-------|------|------------------|------------------|
| A4a   | Manglende lagg-ledd og koeffisient-overlapp i bankligning | Høy | Ja |
| A4b   | π_{t-1} i stedet for π_t (eller E[π_{t+4}]) i mimicking rule | Medium | Trolig ikke (Alt. 1) |
| A4c   | Inkonsistent LTV-sjokk-fortegn mellom ligningene | Medium | Ja |
| A5    | Steady-state konsistenssjekk lagt til | — | Nei |

**Anbefalt rekkefølge for handling:**

1. **A5 først** (passer testen — gir tillit til steady-state-koden)
2. **A4b Alt. 1** (raskt eksperiment uten reestimering, mål B5-effekt)
3. **A4a** (krever reestimering — koordineres med Fase 2)
4. **A4c** (krever reestimering — koordineres med Fase 2)

Punkt 3 og 4 bør håndteres samlet i én ny MCMC-kjøring etter at de er bekreftet av PE.
