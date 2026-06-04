# [PL/DSGE] Transmisjonsdiagnose — RER/UIP-kanalen mot NB (2026-06-04)

**Rolletag:** `[PL]` koordinering, `[DSGE]` UIP-tolkning, `[NUM]` IRF-kjøringer.
**Status:** 🚧 Diagnose ferdig — **strukturelt valg eskaleres til PE før implementering.**
**Bakgrunn:** GEORG-læringssteget (se `GEORG_laeringssteg_plan.md` §11) viste at
NB-avviket er **transmisjonsdrevet**, konsentrert i RER. Denne diagnosen lokaliserer
problemet i UIP/RER-blokken og legger fram det strukturelle valget.

**Metode:** Rene IRF-eksperimenter på kj41-referanseposterior
(`chain_kj41_prod_posterior.json`). Ingen reestimering (krever PE-godkjenning).
Alle tall normalisert til styringsrente-topp = 1. Benchmark: NB Figur 1
(`scripts/nb_multikvartal_score.py`).

---

## 1. Funn: RER-kanalen treffer ikke NB — og kjente parametere lukker ikke gapet

NB Figur 1 (pengepolitikksjokk): RER **−1.50** (impact) → −1.00 (q4) → −0.50 (q8)
→ **−0.20 (q12)**. Altså et stort utslag som henger igjen (monoton, treg
appresiering). Vår modell (kj41): RER −1.04 (impact) → −0.24 (q8) → **+0.12 (q12)**
— for lite utslag, og **snur til positivt** (overshoot) ved lang horisont.

### 1a. UIP-glatting `rho_s` (Justiniano & Preston 2010) — eksisterer, men avveier

kj41 estimerte `rho_s ≈ 0.003` (≈ av). Sweep (mimicking rule):

| rho_s | RER q1 | RER q8 | RER q12 | RMSE(16pt) |
|-------|--------|--------|---------|------------|
| 0.00 (kj41) | −1.04 | −0.24 | +0.12 | 0.295 |
| 0.70 | −0.31 | −0.43 | −0.09 | 0.387 |
| 0.80 | −0.20 | −0.44 | −0.17 | 0.415 |
| 0.90 | −0.10 | −0.33 | −0.22 | 0.456 |

**Konklusjon:** `rho_s` bytter impact mot persistens. Høy `rho_s` fikser
q12-fortegnet (−0.17 @ 0.80, nær NB −0.20) men dreper impact (−0.20 vs NB −1.50).
**Én glatteparameter kan ikke gi både stort utslag og treg hale.** Aggregat-RMSE
forverres monotont.

### 1b. GEORG-regelen hjelper ikke RER

GEORG + rho_s-sweep: RER-impact blir *mindre* (−0.71 @ rho_s=0) fordi GEORGs
impact-rente er lavere. RER snur fortsatt positivt. Bekrefter: RER-gapet er
**ikke** politikkregel.

### 1c. Olje-valuta-kanal `phi_O` — irrelevant for pengepolitikksjokk

`phi_O ∈ {0, 0.15, 0.5, 1.0}` gir **identisk** RER-IRF (oljepris PO beveger seg
ikke under et pengepolitikksjokk). Olje-FX-kanalen virker kun på oljesjokk, ikke
på pengepolitikk-transmisjonen. **Avskrevet for dette avviket.**

---

## 2. Diagnose: strukturell mangel i UIP/RER-blokken

Patologien er at RER **overshooter til positivt** (depresierer forbi SS) når
rentedifferansen lukkes, mens NBs RER henger appresiert. Modellen mangler en
mekanisme som holder RER appresiert etter en innstramming. Tre kandidat-årsaker
(jf. `CLAUDE.md` sigma_rp-fallgrube og Spor C3/C6):

1. **Statisk UIP uten endogen risikopremie.** Risikopremien (`EPS_RP`) er ren
   AR(1)-eksogen; den reagerer ikke på politikkstanden. NBs RER-persistens kan
   kreve en premie som responderer endogent (finansiell akselerator på valuta).
2. **Ufullstendig/treg valutapass-through mangler nivåledd.** UIP er ren
   forventnings-arbitrasje uten friksjon; et nivå-/vaneledd (habit i RER) ville
   gi treg tilbakevending uten å drepe impact.
3. **`sigma_rp`-dominans (kjent).** FEVD: risikopremiesjokk forklarer 88 % av
   RER-varians. Strukturelt symptom på (1)/(2) — dokumentert som begrensning i
   `fase05_begrensningsdokument.md`.

**`sigma_rp`-fastpunktstesten (Spor C3) er fortsatt ukjørt** — den krever
reestimering (PE-godkjenning) og ville skille (1) fra et identifikasjonsproblem.

---

## 3. Strukturelt valg — eskaleres til PE

Alle reelle fikser endrer UIP-likningen i `equations.py` og/eller krever
reestimering. Disse er på eskaleringslisten (AGENTER.md):

| Alternativ | Endring | Eskaleringspunkt | Reversibel? |
|-----------|---------|------------------|-------------|
| **A. Endogen risikopremie** | UIP-premie = f(rentedifferanse/stand) + AR(1). Ny tilstand. | NZ-endring + reestimering | Ny bygger ved siden av v3 (ja) |
| **B. Habit/nivåledd i RER** | Legg `rer_{t-1}`-vane med eget vekt-ledd (utover rho_s). Ev. ny tilstand. | equations.py + reestimering | Ny bygger (ja) |
| **C. sigma_rp-fastpunktstest først (Spor C3)** | Fest sigma_rp=0.006, reestimer øvrige. Diagnostisk, ingen strukturendring. | Reestimering (~2t MCMC) | Helt (ja) |
| **D. Aksepter begrensning** | Ingen endring; dokumentér RER som kjent begrensning, gå videre til Fase 3/4. | Ingen | — |

**PL-anbefaling:** **C før A/B.** Spor C3 (sigma_rp-fastpunkt) er den billigste
neste testen som faktisk skiller hypotesene: faller likelihood drastisk →
modellen *trenger* høy sigma_rp (strukturproblem → A/B berettiget). Er den
omtrent uendret → sigma_rp er absorberende (identifikasjon → A/B kan gjøre
vondt verre uten flere observabler, jf. C6). C krever kun PE-godkjenning for én
reestimering, ikke strukturendring.

---

## 4. Akseptansekriterier (for valgt spor)

- [ ] PE velger A / B / C / D.
- [ ] Hvis A/B: ny bygger ved siden av v3 (v3 urørt), K&M-sidereferanse i commit,
      `test_irf_signs.py` 15/15, `max|eig(T)|<1`, exitstrategi.
- [ ] Hvis C: forhåndsregistrert i `mcmc_log.md` før kjøring; resultat i
      `posterior_vN.json`; PSRF<1.10 rapportert.
- [ ] Funn oppdaterer `fase05_begrensningsdokument.md` (begrensning RER/sigma_rp).

---

*Neste handling: PE velger strukturelt spor (§3). Diagnosen er rent
analysearbeid og krever ingen godkjenning; alle videre steg gjør det.*
