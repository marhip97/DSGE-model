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

## 5. Resultat — Alt A implementert og kalibreringstestet (2026-06-04)

PE valgte **A: endogen risikopremie** (2026-06-04). Implementert som ny bygger
`build_matrices_rpendo` ved siden av v3_forward (v3/v3_forward **urørt**):

- Ny tilstand `RP_ENDO` (NZ_RPENDO=51): `RP_ENDO_t = ρ_pe·RP_ENDO_{t-1} + κ_pe·(i_D − i*)`.
- UIP rad 15 utvidet: `rer_t = … − (1−ρ_s)·RP_ENDO_t`.
- Fremoverskuende Taylor (fixed-point) løst på det utvidede 51-systemet → sammenliknbar med kj41.
- Parametere `kappa_rp_endo`, `rho_rp_endo` i `parameters.py`. Exit: κ_pe=0 → eksakt v3_forward (verifisert, atol=1e-8).
- `tests/test_rpendo.py`: 7 tester (dim, RP_ENDO-lov, UIP-kobling, BK, IRF-fortegn, appresiering, exit). Full suite: **113 passed, 3 xfailed**.

### Kalibreringstest mot NB (kj41-transmisjon, IKKE reestimert)

RER-IRF (normalisert, pengepolitikksjokk):

| (κ_pe, ρ_pe) | RER q1 | q4 | q8 | q12 | RMSE(16pt) vs NB |
|--------------|--------|----|----|-----|------------------|
| (0, –) = kj41 | −1.04 | −0.72 | −0.24 | +0.12 | 0.295 |
| (0.2, 0.7) | −1.25 | −1.13 | −0.48 | +0.11 | **0.263** |
| (0.4, 0.0) | −1.45 | −0.99 | −0.30 | +0.19 | 0.272 |
| NB Figur 1 | −1.50 | −1.00 | −0.50 | −0.20 | — |

**Funn:** En **moderat** endogen premie (κ≈0.2–0.4) løfter RER-impact mot NBs
−1.50 og forbedrer q4/q8 vesentlig. **16-punkts NB-RMSE faller fra 0.295 til
0.263** (≈11 %), BK-stabil i hele området. Den dypeste resten — RER-fortegnet
ved q12 (NB −0.20, vår fortsatt svakt positiv) — reduseres men forsvinner ikke.
Mekanismen virker altså i riktig retning og er teoriforankret, men er foreløpig
**håndkalibrert** mot NB, ikke estimert.

### Neste steg — krever PE-godkjenning (reestimering)

Den håndkalibrerte testen viser at mekanismen er lovende. For å avgjøre om
**data** støtter en endogen premie (og hvilke κ_pe/ρ_pe), kreves reestimering
med `build_matrices_rpendo` og κ_pe/ρ_pe frie — ~2t MCMC, forhåndsregistreres i
`mcmc_log.md`. **Eskaleringspunkt:** reestimering (AGENTER.md pkt. 10) + utvidet
parameterrom. Ikke startet uten eksplisitt godkjenning.

---

*Status: Alt A implementert og kalibreringsverifisert. Reestimering avventer
PE-godkjenning.*

---

## 6. Reestimeringsresultat — kj50 (192k trekk, stoppet av PE 2026-06-04)

PE godkjente reestimering og stoppet kjøringen ved 192k/200k.

**Konvergens:** PSRF_max=1.0022 (✓), ESS_min=877, ESS/n=0.0046 (> kj41s 0.003,
< krav 0.02 — rho-klusteret er fortsatt ESS-flaskehals).

**Estimat (godt identifisert):**
- `kappa_rp_endo` = **0.043 ± 0.016** (ESS=1718) — positiv, ~2.6 sd fra 0. Data
  støtter en endogen premie, men ~5× mindre enn håndkalibrert κ≈0.20.
- `rho_rp_endo` = **0.919 ± 0.018** (ESS=1191) — høy, godt identifisert.
- `psi_R` = **0.9895** (kj41: 0.949) → presset mot prior-tak 0.99.

**NB-fit (16pt RMSE = 0.374; kj41: 0.295):**

| Var | q1 | q4 | q8 | q12 | NB q1/q4/q8/q12 |
|-----|----|----|----|-----|------------------|
| I_R | +1.00 | +0.96 | +0.91 | +0.86 | 1.00/0.55/0.10/−0.15 |
| RER | −1.19 | −1.18 | −0.78 | **−0.25** | −1.50/−1.00/−0.50/−0.20 |
| PI  | −0.15 | −0.27 | −0.28 | −0.20 | −0.03/−0.14/−0.22/−0.22 |
| Y   | −0.48 | −0.63 | −0.51 | −0.33 | −0.12/−0.47/−0.40/−0.25 |

**Konklusjon:** Den endogene premien er **datastøttet** (liten κ, høy ρ) og
**løser de to diagnostiserte patologiene**: RER-q12-fortegnsskiftet (kj41 +0.12 →
kj50 −0.25 ≈ NB −0.20) og inflasjonspersistensen (PI nå nær NB ved q8/q12). Men
frigjøring av UIP-kanalen presser `psi_R` til taket 0.99 → renten overpersisterer
(I_R.q12 +0.86 vs NB −0.15) → aggregert NB-RMSE forverres (0.374 vs 0.295). Dette
er **begrensning 7** (UIP-kanal ↔ psi_R-korrelasjon) som reaktiveres, samme
avveiing som ved phi_O. Figur: `data/results/kj50_vs_nb.png`.

**Anbefalt oppfølging (eskaleringspunkt, ikke startet):** Bryte
psi_R↔premie-korrelasjonen — f.eks. (i) reestimere med psi_R kalibrert fast =
0.949 (kj41) og kun κ_pe/ρ_pe frie, for å isolere premiens bidrag uten
renteglatting-drift, eller (ii) informativ/strammere psi_R-prior. Krever
PE-godkjenning.

---

## 7. Oppfølging kj51 — psi_R pinnet (PE-godkjent, 200k trekk)

psi_R pinnet = 0.949 via dogmatisk prior `Normal(0.949, 0.0005)` (prior_overrides),
øvrig = kj50. Konvergens: PSRF_max=1.120 (rho_rp, marginalt over), ESS_min=204.

**Estimat:**
- `psi_R` = 0.9494 (pinnet ✓)
- `kappa_rp_endo` = **0.094 ± 0.094** — std = mean → **svakt identifisert** (mot kj50s
  tette 0.043 ± 0.016). Premien er ikke lenger pinnet ned når psi_R holdes fast.
- `rho_rp_endo` = 0.389 ± 0.218 (kollapset fra kj50s 0.92).
- `rho_rp` (eksogen risikopremie-AR1) = **0.910** (opp fra kj41/kj50 ~0.15).

**NB-fit:** 16pt-RMSE = **0.281** (BESTE; kj41 0.295, kj50 0.374). Men:
- I_R bedre (psi_R holdt nede): +1.00/+0.81/+0.62/+0.47.
- RER **q12 = +0.14** — fortegnsskiftet TAPT igjen (kj50 hadde −0.25). PI tilbake til lav persistens.

**Hovedfunn — observasjonsekvivalens (Spor C6 / begrensning 7):**
Persistensen *migrerer* mellom tre substituerbare kanaler avhengig av hva som er
fritt: (a) renteglatting psi_R→0.99 (kj50), (b) endogen premie ρ_pe=0.92 (kj50),
(c) eksogen risikopremie ρ_rp=0.91 (kj51). Avgjørende: (c) inngår **ikke** i
pengepolitikk-IRF-en (risikopremiesjokk=0 under et rentesjokk), så kj51 vinner på
likelihood/aggregat-RMSE men **mister** den monetære RER-q12-fikset som kj50 ga.

**Konklusjon:** Den endogene risikopremien, renteglattingen og den eksogene
risikopremie-persistensen er **ikke separat identifiserbare** med dagens 14
observabler. Dette er en strukturell identifikasjonsgrense (observasjonsekvivalens),
ikke løsbar ved parameterpinning. Den NB-treffende RER-banen krever et
**høy-persistens endogent premieledd** (kj50), men data foretrekker å legge
persistensen i den eksogene premien når den får lov — og de to er empirisk
uskillelige uten flere observabler.

**Anbefaling (eskaleringspunkt):** Løs identifikasjonen med **flere
observasjonsserier** som skiller kanalene — f.eks. NIBOR-OIS/pengemarkedspremie
som egen observabel, valuta-terminpremie, eller en kreditspread (jf. Spor C6).
Alternativt: aksepter RER-q12 som dokumentert begrensning og gå til Fase 3/4 med
kj41 som referanse. Begge krever PE-beslutning.

---

## 8. kj52 — i_3m anker pengemarkedspremien (PE-godkjent, 200k trekk)

Utnytter en EKSISTERENDE serie: `i_3m_obs` (NIBOR 3M) re-mappet fra redundant
I_R-observasjon til `i_3m = i_R + EPS_PREM`. **Kritisk funn underveis:**
premie-sjokket `E_prem` var inaktivt (Q=0) i alle tidligere kjøringer → `EPS_PREM`
var en død tilstand. Aktivert ved å estimere `sigma_prem` (N=22). psi_R holdt fri.

**Konvergens (beste av rpendo):** PSRF=1.0041, ESS_min=750.

**Resultat:**
- `sigma_prem` = 0.0002 (ESS=1931) — pengemarkedspremien nå aktiv og identifisert.
  log-posterior −3114 (kj50 −3306, kj51 −3537): observablen tilfører reell info.
- `kappa_rp_endo` = 0.042 ± 0.015 (ESS=2169) — robust, ~ kj50.
- `psi_R` = 0.9895 — **driver fortsatt til taket.** Ankeret brøt IKKE driften.
- 16pt-RMSE = 0.376. RER best av alle (−1.08/−1.01/−0.68/−0.29 vs NB; q4 nær
  eksakt, q12 negativ), men I_R overpersisterer (psi_R=0.99) → dominerer RMSE.

**Tolkning:** Pengemarkedspremien (`EPS_PREM`/`i_3m`) er et **annet objekt** enn
FX-risikopremien / renteglatting-floken. i_3m løste premie-identifikasjonen (en
egen, tidligere død kanal — verdifullt i seg selv) og bekreftet at den endogene
FX-premien er robust datastøttet, men brøt **ikke** psi_R↔FX-premie-ekvivalensen.

**Samlet konklusjon (kj50–52):** Den endogene FX-risikopremien er reell og
datastøttet, og fikser RER-patologiene (kj52 RER er utmerket). Men den er
observasjonsekvivalent med renteglatting (psi_R→0.99) gitt dagens observabler;
verken psi_R-pinning (kj51) eller pengemarkedsrente-anker (kj52) bryter floken.
**Å bryte den krever en FX-spesifikk observabel** (valuta-terminpremie /
cross-currency basis), ikke pengemarkedsrenten. Figur: `data/results/kj52_vs_nb.png`.

## 9. Avslutning — FX-sporet lukket (PE-beslutning 2026-06-04)

Vi vurderte en FX-spesifikk observabel for å bryte floken. Diagnose av en
**konstruert UIP-proxy** fra eksisterende data (`i_R_obs − ds_obs_{t+1}`): for
svak som anker — std 0.025 dominert av valutakurs-prognosefeil (mot rente-skala
~0.005), AR(1)=0.14 (nær støy), ingen observert `i*` (utenlandsk rente er en død
tilstand), og enhetsskalerings-tvetydighet (i_R_obs ×4 vs ds_obs ×1 i H). En ren
FX-serie (cross-currency basis / terminpremie) er markedsdata, ikke tilgjengelig
via SSB/NB/FRED-API-ene og blokkert fra sky-IP.

**PE besluttet 2026-06-04: avslutt FX-sporet.** Identifikasjonsgrensen aksepteres
som dokumentert begrensning (`fase05_begrensningsdokument.md` § 8). **kj41 forblir
referanseestimat** (RMSE=0.277). PARAM_PRIORS tilbakestilt til kj41-default (N=19);
premie-sjokket (`E_prem`) deaktivert igjen i `build_Q`. **Fase 3 (analyseverktøy)
kan starte.**

**Bevart for fremtidig reaktivering (med FX-serie):** `build_matrices_rpendo`,
`build_H_rpendo`/`build_H_rpendo_i3m`, parametrene `kappa_rp_endo`/`rho_rp_endo`
(parameters.py), `tests/test_rpendo.py`, og kj50–52-skriptene. Alle med exit
κ=0 → v3_forward.

*Merk: kj52-commit/datafiler ble midlertidig borte da sky-containeren ble
gjenvunnet, men lå på remote og er gjenopprettet (chain_kj52_prod_posterior.json,
kj52_vs_nb.png).*
