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

### 3.0 KORREKSJON (2026-06-04): sigma_rp er allerede fast og irrelevant for IRF-gapet

Ved nærmere ettersyn (empirisk verifisert):

1. **`sigma_rp` er allerede fiksert til 0.006** (K&M) i `mcmc.py`
   (`SIGMA_RP_FIXED=0.006`, PE-godkjent kj10 2026-05-24). kj41s `PARAM_NAMES`
   inkluderer **ikke** sigma_rp. Det opprinnelige C3-eksperimentet (0.017→0.006)
   er altså **allerede gjennomført** — kj41 *er* «sigma_rp fast = K&M».
2. **`sigma_rp` påvirker ikke pengepolitikk-IRF-en i det hele tatt.** RER-IRF for
   et pengepolitikksjokk er **identisk** for sigma_rp ∈ {0.006, 0.017, 0.05}.
   Grunn: sigma_rp skalerer kun risikopremiesjokket (`E_rp`); under et
   pengepolitikksjokk er `E_rp=0`, så sigma_rp inngår aldri i `R[:,E_i]`-banen.
   FEVD-andelen (88 % av RER-varians) gjelder *ubetinget varians* drevet av
   risikopremiesjokk — et **annet objekt** enn IRF-gapet mot NB Figur 1.

**Følge:** En ny «fest sigma_rp=0.006 og reestimer» (opprinnelig spor C) ville
bare reprodusere kj41 og kan **ikke** lukke det diagnostiserte
monetær-IRF-RER-gapet. Spor C er derfor strøket som tiltak mot dette avviket.

### 3.1 Reelle alternativer

| Alternativ | Endring | Eskaleringspunkt | Adresserer monetær RER-IRF? |
|-----------|---------|------------------|------------------------------|
| **A. Endogen risikopremie** | UIP-premie reagerer på rentestand/-differanse (+AR(1)). Ny tilstand. v3 urørt. | NZ-endring + reestimering | **Ja** — kan holde RER appresiert mens renten er høy (uten å drepe impact) |
| **B. Habit/nivåledd i RER** | `rer_{t-1}`-vane utover rho_s. | equations.py + reestimering | Delvis — samme avveiing som rho_s (bakoverledd → impact↓) |
| **C′. Fri sigma_rp (varians-test)** | Frigjør sigma_rp, se om data vil ha den høy. Diagnostisk for FEVD, **ikke** for IRF-gapet. | Reestimering (~2t) | Nei (IRF uendret); kun varians/FEVD-diagnose |
| **D. Aksepter begrensning** | Dokumentér RER som kjent begrensning, gå til Fase 3/4. | Ingen | — |

**PL-anbefaling (revidert):** **A.** Det monetære RER-gapet er strukturelt i
UIP og uavhengig av sigma_rp. `rho_s`-sweepen viste at et rent bakoverledd (≈ B)
avveier impact mot persistens. En endogen risikopremie (A) som stiger med
rentestanden kan gi *både* stort impact og treg hale, og er teoretisk forankret
(Mæhlum 2025, Staff Memo 3/2025, UIP og valutakurs). A bygges som ny bygger ved
siden av v3 (samme mønster som GEORG), med exitstrategi og K&M/Mæhlum-referanse.

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
