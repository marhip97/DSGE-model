# PE-eskalering — Fase 3: Oljepriskanal (sigma_rp-problem)

**Avsender:** STAT/DSGE  
**Mottaker:** Prosjekteier (PE)  
**Dato:** 2026-06-03  
**Forankring:** Begrensning 1 (BNP 6×, RER 29× NB-benchmark)

---

## Sammendrag

Modellen har `phi_O = 0.15` (K&M Tabell 8) — en direkte olje→RER-kanal i
UIP-likningen. Kanalen eksisterer, men er kalibrert fast og er ikke sterk nok
til å absorbere observerte oljepris-drevne svingninger i NOK. Data estimerer
derfor `sigma_rp = 0.017` (vs. K&M 0.006) — risikopremiesjokket fungerer som
en residual-absorber for alle uforklarte valutabevegelser, inkludert oljepris.

**Anbefalt tiltak:** Frigjør `phi_O` for estimering + legg oljepris til
observasjonssettet. Forventet effekt: sigma_rp faller mot K&M 0.006, og
BNP/RER-IRF reduseres mot NB-benchmark.

---

## Nåværende tilstand — hva finnes

### UIP-likningen (rad 15, `build_matrices_v3`)

```
rer_t = rho_s·rer_{t-1} + (1-rho_s)·[
    E_t[rer_{t+1}]
  − (i_D,t − π_t)
  + (i*_t − π*_t)
  + ε_rp,t          ← sigma_rp = 0.017 (estimert, for stor)
  + phi_O·pO_t      ← kalibrert fast = 0.15 (K&M Tabell 8)
  + phi_B·b_NW,t    ← kalibrert fast = 0.0016 (K&M Tabell 8)
]
```

`pO_t` er allerede en AR(1)-tilstand (state 28, driven av E_O-sjokket).
`phi_O = 0.15` betyr: +10% oljepris → +1.5% NOK-appresiering (lavere RER).

### Problemet

kj46-posterior viser `rho_O = 0.244` (K&M: 0.874). Med `rho_O = 0.244`
og `phi_O = 0.15`:
- Oljeprissjokk er svært kortvarige (0.24-persistente)
- Effekten på RER er liten og kortvarig
- Alt uforklart NOK-volatilitet must-absorb via `sigma_rp`

**Diagnose:** Enten er `phi_O = 0.15` for lav (data trenger ~0.4–0.6), eller
`rho_O = 0.244` er misidentifisert (reel rho_O for Brentpris ≈ 0.85–0.95),
eller begge deler.

---

## Tre alternativer (eskalert for PE-beslutning)

### Alt. A — Estimer phi_O fritt (enklest, anbefales)

**Hva:** Legg `phi_O` til PARAM_PRIORS med informativ prior.
Prior: `phi_O ~ Beta(2.0, 3.5, [0.05, 0.80])` → mean ≈ 0.22, 95%-CI [0.08, 0.55].

K&M Tabell 8 kalibration var 0.15. Prior sentreres litt over med vide haler
slik at data kan velge høyere verdi om nødvendig.

**Hva endres:**
- `parameters.py`: `phi_O = 0.15 # (CAL)` → `phi_O = 0.15 # (EST — frigjort Fase 3)`
- `mcmc.py`: legg til `'phi_O': ('beta', 2.0, 3.5, 0.05, 0.80)` i PARAM_PRIORS
- 20 → 21 estimerte parametre (eller 20 om rho_s kalibreres fast per PE_eskalering_fase2_sampler)

**Forventet effekt:**
- phi_O konvergerer mot 0.30–0.50 (litteraturanslag for olje-eksporterende land)
- sigma_rp kan falle mot 0.008–0.012 (lavere, men sannsynligvis fortsatt over K&M)
- BNP- og RER-IRF reduseres noe mot NB-benchmark

**Kostnader:** ~1 time implementering + 2 timer MCMC (kj48).

**Risiko:** Prior-avhengighet om phi_O og sigma_rp er sterkt negativt korrelerte.

---

### Alt. B — Legg oljepris til observasjonssettet (strukturelt riktig)

**Hva:** Legg Brent råolje (realpris) til observasjonsvektoren `Y_obs`.
pO (state 28) identifiseres da direkte fra data — rho_O og sigma_O forankres mot
faktisk oljepris-serie fra FRED/SSB (Fase 1).

**Observasjonslikning:** `log(P_oil,t / P_oil,t-1) = pO_t + e_obs`
Lagt til som rad 15 i H-matrisen.

**Hva endres:**
- `data/processed/`: ny serie `oil_price_real.csv` fra FRED (DCOILBRENTEU deflatert)
- `mcmc.py`: `N_OBS: 14 → 15`, oppdatert `build_H()` med ny observasjonslikning
- `kalman.py`: oppdatert `Sv` (15×15 diagonal kovarians) med oljeprisbråk
- Krever **PE-godkjenning** (ny variabel i observasjonssettet — eskaleringsliste)

**Forventet effekt:**
- rho_O identifiseres mot Brent-empirisk persistens (~0.85–0.95)
- sigma_O kalibreres mot observert oljeprisvolatilitet
- sigma_rp sannsynligvis redusert til ~0.008–0.010

**Kostnader:** ~1 dag implementering + 2 timer MCMC (kj48).

**Risiko:** Ny variabel kan skape identifikasjonsproblemer; N_OBS-endring krever
grundig testing av Kalman-filter.

---

### Alt. C — Estimér rho_O med informativ prior (minst invasiv)

**Hva:** Gi `rho_O` en mer informativ prior sentrert på empirisk Brent-persistens.
Nåværende prior: `Beta(2.0, 0.5, 0.01, 0.9995)` → mean ≈ 0.80.
kj46-posterior: `rho_O = 0.244` — langt fra prior mean, tyder på at likelihood
trekker aggressivt mot lav persistens pga. manglende forankring mot faktisk oljeprisdata.

**Ny prior:** `Beta(6.0, 1.5, 0.50, 0.9995)` → mean ≈ 0.80 med smal fordeling
(95%-CI: [0.70, 0.93]) — tvinger rho_O mot empirisk verdi.

**Hva endres:**
- `mcmc.py`: én linje — ny prior for rho_O
- Krever PE-godkjenning (prior-stramming — eskaleringsliste)

**Forventet effekt:**
- rho_O til ~0.75–0.90 (mot K&M 0.874)
- phi_O·pO bidrar mer til RER-dynamikk over tid
- sigma_rp kan falle noe, men trolig moderat effekt alene

**Kostnader:** ~15 min + 2 timer MCMC (kj48).

**Risiko:** Modellen kanskje faktisk trenger lav rho_O — stram prior kan gi
likelihood-strid og dårligere RMSE.

---

## Tilrådning

**Anbefalt rekkefølge: Alt. C → Alt. A → (Alt. B betinget)**

### Steg 1 — Alt. C (kj47 eller kj48, ~15 min)

Strammer rho_O-prior mot empirisk Brent-persistens. Rimeligst test —
avklarer om lav rho_O er ekte identifikasjonsproblem eller prior-artefakt.

### Steg 2 — Alt. A (kj48, ~1 time)

Frigjør phi_O for estimering kombinert med strammet rho_O.
Høyere phi_O + høyere rho_O gir sterkere og mer persistent olje→RER-kanal
→ sigma_rp kan falle.

### Steg 3 — Alt. B (betinget, krever PE)

Kun om Alt. A ikke løser sigma_rp-problemet. Legger oljepris til observasjonssettet
som "ankerfeste" — strukturelt korrekt løsning men mer invasiv.

---

## Implikasjon for begrensning 1

Dersom Alt. A hever phi_O til ~0.40 og rho_O til ~0.85:
```
phi_O · rho_O^4 ≈ 0.40 × 0.52 ≈ 0.21  (kv4-bidrag)
```
vs. nåværende `0.15 × 0.244^4 ≈ 0.0002` (neglisjerbar).

Oljeprissjokkene tar over noe av RER-forklaringen fra risikopremie.
Estimert sigma_rp vil sannsynligvis falle, men det er usikkert om det
når K&M-nivå 0.006 — dette avhenger av data.

---

## PE-beslutning som trengs

☐ **Godkjenn Alt. A** — estimer phi_O fritt (prior Beta(2.0, 3.5, [0.05, 0.80]))  
☐ **Godkjenn Alt. C** — stram rho_O-prior til Beta(6.0, 1.5, [0.50, 0.9995])  
☐ **Godkjenn Alt. B** — ny observabel (oljepris); krever separat PE-godkjenning  
☐ **Kombiner A+C** — estimer phi_O fritt OG stram rho_O-prior  
☐ **Avvent** — implementer sampler (kj47) først, ta oljepriskanal i kj48/kj49

---

## Vedlegg: Relevante parametre i nåværende posterior (kj46)

| Parameter | K&M | kj46 posterior | Avvik | Kommentar |
|-----------|-----|---------------|-------|-----------|
| phi_O | 0.150 | 0.150 (fast) | — | Oljepris→RER-koeff, ikke estimert |
| rho_O | 0.874 | **0.244** | 3.6× | Svært lav persistens — misidentifisert? |
| sigma_rp | 0.006 | **0.017** | 2.8× | Absorberer alt uforklart NOK-volatilitet |
| sigma_O | 0.079 | **0.151** | 1.9× | For stor oljeprisvolatilitet i modellen |

Merk: phi_O og rho_O er nå konsistent (fast og lavt), men denne kombinasjonen
gir en oljepriskanal med neglisjerbar akkumulert effekt på RER. Dataene
kompenserer med høy sigma_rp.
