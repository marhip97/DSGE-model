# [PL] GEORG — læringssteg: planleggingsdokument

**Rolletag:** `[PL]` (koordinering) med faglige bidrag fra `[DSGE]` (regelform,
dekning), `[NUM]` (BK-stabilitet, forventnings-maskineri) og `[STAT]`
(IRF-sammenligning).
**Dato:** 2026-06-04
**Status:** 🚧 Planforslag — **avventer PE-godkjenning før implementering.**
**Kilde:** Almlid, Haltia & Robstad (2025), *Mapping Optimal Policy into a Rule
in NEMO: GEORG*, Staff Memo 15/2025 (`docs/staff-memo-2025-15-georg.txt`).

---

## 0. Hvorfor GEORG — det strategiske spørsmålet

Fase 2 er avsluttet (kj41, RMSE=0.277). Modellen treffer **ikke** NB Memo
3/2024 Figur 1 (IRF av pengepolitikksjokk). Men den figuren viser IRF fra NBs
**tapsfunksjonsbaserte optimale politikk**, mens vår modell bruker en
AR(1)-mimicking rule (rad 20 / `I_R` i `build_matrices_v3*`). Vi vet derfor
**ikke** om avviket skyldes:

1. **Politikkregelen** (vår mimicking rule ≠ NBs optimale politikk), eller
2. **Transmisjonen** (vår modells likninger propagerer sjokket annerledes).

GEORG ("Ganske Enkel Optimal ReGel") er NBs egen enkle regel som via
IRF-matching reproduserer nettopp den tapsfunksjonsbaserte optimale politikken
i NEMO. Implementerer vi GEORG i vår modell og beregner IRF for et
pengepolitikksjokk, **isolerer vi spørsmålet**:

- Hvis GEORG nærmer seg NB-figuren → avviket var drevet av **politikkregelen**.
- Hvis avviket består med GEORG → avviket er drevet av **transmisjonen**.

Dette er et **rent læringssteg**. GEORG estimeres ikke mot data; den brukes til
IRF-sammenligning. Konklusjonen styrer prioriteringen i neste fase (se §9).

---

## 1. GEORG-regelens form (Staff Memo 15/2025, §2)

### Regelen (lign. 1 og 3)

Styringsrenten i kvartal *t*:

```
r_t = r̄_t + ω_r·(r_{t-1} − r̄_t) + (1 − ω_r)·X_t + Z_t            (1)
```

der `ω_r ∈ [0,1)` er renteglatting, og det sammensatte leddet `X_t` samler de
fremoverskuende makroindikatorene:

```
X_t = E_t[ ω_π·π̂_{t+1} + ω_y·ŷ_{t+1} + ω_ϕ·ϕ̂_{t+1}
           + ω_S·Ŝ_{t+1} + ω_rf·r̂^f_{t+1} + ω_μ·μ̂_t ]            (3)
```

Alle variabler er **gap** (avvik fra steady-state, `^`). Pengepolitikksjokket
`Z_t` følger en AR(1)-prosess (lign. 2):

```
Z_{t+1} = λ_Z·Z_t + ε_{t+1},   |λ_Z| < 1,   λ_Z = 0.75            (2)
```

λ_Z = 0.75 er satt konsistent med persistensen i pengepolitikksjokket i NEMO
med tapsfunksjon.

### De seks indikatorene (alle som gap)

| Symbol | Beskrivelse | Definisjon i memoet |
|--------|-------------|---------------------|
| π̂_{t+1} | Inflasjon (KPI-JAE) | 1-kv. frem-anslag av **fire-kvartalers** KPI-JAE-vekst |
| ŷ_{t+1} | Innenlandsk outputgap | y − ȳ |
| ϕ̂_{t+1} | Lønnskostnadsvekst-gap | **fire-kvartalers** vekst i enhetslønnskostnad (lønn justert for produktivitetstrend) |
| Ŝ_{t+1} | Valutakursvekst-gap | **åtte-kvartalers** vekst i nominell I-44: (x_t − x_{t-8})/x_{t-8}·100 |
| r̂^f_{t+1} | Utenlandsk rente-gap | r^f − r̄^f |
| μ̂_t | Pengemarkedspremie-gap (samtid, ikke t+1) | 3m NIBOR − forventet styringsrente, − μ̄ |

### Koeffisienter (Tabell 2/4, IRF-matchet via ABC-algoritme)

| Param | ω_r | ω_π | ω_y | ω_ϕ | ω_S | ω_rf | ω_μ |
|-------|-----|-----|-----|-----|-----|------|-----|
| Verdi | 0.74 | 1.17 | 1.27 | 1.25 | 0.13 | 0.25 | −1.00 |

Pluss λ_Z = 0.75 for sjokk-persistensen.

**Merknad om annualisering (`[NUM]`/`[STAT]`):** Tabell 4 er oppgitt som
*annualiserte* koeffisienter, og GEORGs rente er annualisert. Vår modell har
kvartalsrater i tilstandsvektoren (jf. `build_H` som multipliserer `PI` og
`I_R` med 4 for å annualisere i observasjonslikningen). Implementeringen må
sikre konsistent annualisering mellom indikator-gap og rente-gap, ellers blir
nivået på renterresponsen feil. Dette avklares eksplisitt i §3.

---

## 2. Dekningsanalyse — hvilke indikatorer finnes i dag?

Gjennomgang av tilstandsvektoren (`equations.py`, NZ-blokker) og
observasjonssettet (`mcmc.py`, `build_H`). **Viktig avgrensning:** GEORG erstatter
kun **politikkregelen** (rad 20 / `I_R`). For læringssteget (IRF-sammenligning)
trengs **ingen nye observasjonsvariabler** — eskaleringspunkt «ny
observasjonsvariabel» utløses derfor **ikke** her. Spørsmålet er om hver
indikator kan uttrykkes som en lineærkombinasjon av eksisterende tilstander
(evt. via forventnings-maskineriet), eller om den krever **nye tilstander**
(→ NZ-endring → eskalering).

| # | Indikator | Finnes? | Eksisterende tilstand | Minste endring | Eskalering? |
|---|-----------|---------|------------------------|----------------|-------------|
| 1 | π̂ (inflasjon) | **Delvis** | `PI` (0), `pi_lag` (36) | Fire-kv.-vekst krever π_{t-2}, π_{t-3} + 1-kv.-forventning E_t[π_{t+1}]. Forventningsleddet dekkes av forward-maskineriet (T-potenser, jf. `v3_forward`); 4-kv.-summen krever 2 ekstra inflasjonslagg-tilstander. | **Ja** (nye lagg-tilstander → NZ) |
| 2 | ŷ (outputgap) | **Ja** | `Y` (9) | E_t[y_{t+1}] = e_Y·T·z_t via forward-maskineri. Ingen ny tilstand. | Nei |
| 3 | ϕ̂ (lønnskostnadsvekst) | **Nei (hull)** | `piW` (4), `w` (5), `a` (37), `l` (10), `y` (9) | Enhetslønnskostnad-vekst = nominell lønnsvekst − produktivitetstrend. Fire-kv.-vekst krever akkumulering. Konstruerbar fra eksisterende states, men trenger akkumulator-/lagg-tilstander. | **Ja** (nye tilstander → NZ) |
| 4 | Ŝ (8-kv. valutakursvekst) | **Nei (hull)** | `s` (19, 1-kv. nominell kursendring) | 8-kv.-vekst = Σ_{j=0..7} s_{t−j}. Krever 8-kvartalers vindu → ~7 nye lagg-tilstander (eller akkumulator + s_{t-8}-lagg). **Dominerende dimensjonskostnad.** | **Ja** (nye tilstander → NZ) |
| 5 | r̂^f (utenlandsk rente) | **Ja** | `I_STAR` (44) | E_t[i*_{t+1}] = e_{I_STAR}·T·z_t via forward-maskineri. Ingen ny tilstand. | Nei |
| 6 | μ̂ (pengemarkedspremie) | **Ja** | `EPS_PREM` (46) | Samtidig gap, direkte kobling. Ingen ny tilstand. | Nei |
| — | Z_t (AR(1)-sjokk, λ_Z=0.75) | **Nei (hull)** | `E_i` (sjokk 7) er i.i.d. i dag (`Psi[20,E_i]=1.0`, ingen AR(1)-state) | Ny AR(1)-tilstand for persistent pengepolitikksjokk. Minimal (én tilstand, som øvrige AR(1)). | **Ja** (ny tilstand → NZ) |

### Oppsummering av hull

- **Indikator 2, 5, 6** finnes — dekkes av eksisterende tilstander (+ forward-maskineriet for t+1-leddene).
- **Indikator 1** krever 2 ekstra inflasjonslagg (for 4-kv.-vekst).
- **Indikator 3** (lønnskostnad) krever akkumulator-/lagg-tilstander for 4-kv.-vekst.
- **Indikator 4** (8-kv. valutakursvekst) er det største hullet (~7 nye tilstander).
- **Z_t** krever én ny AR(1)-tilstand (persistent sjokk).

**Konsekvens:** GEORG kan ikke implementeres uten å øke tilstandsromdimensjonen.
Dette er på **eskaleringslisten** (AGENTER.md §«Eskaleringsregler» pkt. 1:
modellens dimensjon NZ/NE; CLAUDE.md «Hva som ikke skal endres uten
godkjenning»). **Se §8 for det formelle eskaleringspunktet.**

### Minimerings-strategi (reduserer ny dimensjon)

Forventningsleddene `E_t[X_{t+1}]` (for π, y, r^f) trenger **ikke** nye
jump-/hjelpetilstander: de kan uttrykkes som lineærkombinasjoner av nåværende
tilstand via T-potenser i en fixed-point-løsning — nøyaktig slik
`build_matrices_v3_forward` allerede beregner E_t[π_{t+4}] = e_PI·T⁴·z_t. De
genuint nye tilstandene er derfor kun **akkumulatorene/laggene** for:

- 8-kv. valutakursvekst (Ŝ) — størst,
- 4-kv. lønnskostnadsvekst (ϕ̂),
- 4-kv. inflasjonsvekst (to ekstra π-lagg), og
- AR(1) pengepolitikksjokk Z_t.

Et anslag er **NZ_GEORG ≈ 51 + 10–12 ≈ 61–63** avhengig av hvor kompakt
8-kv.-vinduet implementeres. Eksakt tall fastsettes i implementeringsfasen og
rapporteres til PE. (Til sammenligning: `v3_forward` NZ=50, `v3_plt` NZ=51,
`altB` NZ=51, `pi4chain` NZ=53, `v4` NZ=56.)

---

## 3. Implementeringsplan (avventer PE-godkjenning)

### Prinsipp: ny bygger ved siden av eksisterende — ingen endring av v3

`build_matrices_georg()` legges til som en **ny** bygger i `equations.py`, helt
analogt med mønsteret for `build_matrices_v3_plt`, `build_matrices_altB` og
`build_matrices_pi4chain`. **`build_matrices_v3`, `build_matrices_v3_forward`
og mimicking rule (rad 20-koeffisientene i v3) endres ikke.**

### Skisse

```
def build_matrices_georg(p=None, theta_H=0.05, lambda_pi4=None,
                          n_iter=30, tol=1e-8, use_georg=True):
    """
    NEMO med GEORG-politikkregel (Staff Memo 15/2025) ved siden av v3.

    1. Hent (NZ=50)-matriser fra build_matrices_v3_forward (gjenbruker
       fixed-point forventnings-maskineriet for E_t[·_{t+1}]-leddene).
    2. Utvid til NZ_GEORG med nye akkumulator-/lagg-tilstander:
         - S8-vindu (8-kv. valutakursvekst),
         - ULC4-akkumulator (4-kv. lønnskostnadsvekst),
         - 2 ekstra π-lagg (4-kv. inflasjon),
         - Z_MP (AR(1) pengepolitikksjokk, λ_Z=0.75).
    3. Erstatt rad 20 (I_R) med GEORG (lign. 1+3):
         i_R_t = ω_r·i_R_{t-1} + (1-ω_r)·[ω_π·E[π̂4_{t+1}] + ω_y·E[ŷ_{t+1}]
                  + ω_ϕ·E[ϕ̂_{t+1}] + ω_S·E[Ŝ_{t+1}] + ω_rf·E[r̂^f_{t+1}]
                  + ω_μ·μ̂_t] + Z_t
       Forventningsledd via T-potenser (fixed-point, som v3_forward).
    4. Koble Z_MP inn i rad 20 og sett Psi[Z_MP, E_i]=1, G1[Z_MP,Z_MP]=λ_Z.
    """
```

### Gjenbruk av forventnings-maskineriet (`[NUM]`)

`build_matrices_v3_forward` har allerede en fixed-point-løkke som regner
`E_t[π_{t+4}] = e_PI·T⁴·z_t` og legger det inn i rad 20 som en
lineærkombinasjon av alle tilstander. **Samme mekanisme gjenbrukes** for
GEORGs `E_t[π̂4_{t+1}]`, `E_t[ŷ_{t+1}]` og `E_t[r̂^f_{t+1}]` (T¹-potens i
stedet for T⁴, samt T-uttrykk for 4-kv.-inflasjonssummen). Alternativt kan
`pi4chain`-kjeden brukes for inflasjonsleddet hvis fixed-point gir
konvergensproblemer. Begge er allerede validert i kodebasen.

### Annualisering (`[NUM]`)

GEORG-koeffisientene (Tabell 4) er annualiserte. Vår tilstand `I_R` og `PI` er
kvartalsrater (annualiseres ×4 i `build_H`). Implementeringen bruker
gap-tilstander direkte og kalibrerer ω slik at rente-respons og
indikator-respons er på samme (annualiserte) skala. Konkret testes dette i
`test_georg`: GEORG-IRF for et +1pp (annualisert) styringsrentesjokk skal ha
korrekt fortegn og rimelig magnitude før IRF-sammenligning.

### Exitstrategi (`[NUM]`)

To nivåer, etter mønster fra `psi_PL=0`- og `psi_R2=0`-exit:

1. **`use_georg=False`** → bygger returnerer `build_matrices_v3_forward`-output
   uendret (de nye GEORG-tilstandene blir «dead states», jf. P_STAR_GAP).
   Dette er kjent referanseatferd (mimicking rule) og brukes i exit-testen.
2. **ω-vektor → 0** (alle seks indikatorvekter null, behold kun ω_r) → regelen
   reduseres til ren renteglatting i_R_t = ω_r·i_R_{t-1} + Z_t, som er en
   degenerert, men veldefinert grensetilfelle for sanity-sjekk.

### Steady-state-forenkling (`[DSGE]`)

Modellen er lineærisert i gap (steady-states = 0). GEORGs nivåer i Tabell 1
(r̄=3.05%, r̄^f=2.65%, μ̄=0.25pp, ϕ̄=2.4%) trengs **ikke** for IRF i
lineær modell — de faller ut i avviksform. Dette forenkler implementeringen
betydelig: vi mapper kun ω-koeffisientene på gap-tilstandene.

---

## 4. Læringsmål — IRF-protokoll (det viktigste)

Når `build_matrices_georg` er på plass og BK-stabil:

1. **Beregn IRF for pengepolitikksjokk** (E_i, +1pp annualisert
   styringsrente-topp, normalisert som NB-figuren) med GEORG-regelen, for
   π, BNP, boligpris, RER, reallønn, styringsrente over 16 kvartaler.
2. **Sammenlign (a): GEORG vs NB Memo 3/2024 Figur 1.** Bruk samme
   tabellformat og normalisering som Spor B5
   (`docs/oppgaver/B5_nb_benchmark.md`): topp-magnitude, topp-tidspunkt,
   halveringstid per variabel.
3. **Sammenlign (b): GEORG vs vår mimicking-rule-IRF** (kj41-referanse,
   `build_matrices_v3_forward`).
4. **Svar eksplisitt på hovedspørsmålet:**
   - **Nærmer GEORG seg NB-figuren** (særlig I_R.q12, jf. begrensning 6, og
     persistens i Y/PI)? → Avviket var drevet av **POLITIKKREGELEN**.
   - **Består avviket med GEORG**? → Avviket er drevet av **TRANSMISJONEN**.

Resultatet rapporteres til PE med tabell + figur og en entydig konklusjon.

---

## 5. Tester (etter mønster fra `tests/test_plt_kanal.py`)

Ny fil `tests/test_georg.py`:

- `test_matrisedimensjoner` — `build_matrices_georg` returnerer
  (NZ_GEORG×NZ_GEORG), (·×NE=13)-matriser med korrekte former.
- `test_bk_stabilitet` — BK-stabil med GEORG-koeffisientene fra Tabell 4
  (`max|eig(T)| < 1.0`).
- `test_irf_fortegn` — pengepolitikksjokk (+) gir BNP(−) og π(−) (de mest
  robuste kvalitative kravene; jf. `__main__.py` sjekkene), evt. flere fra
  `test_irf_signs.py`.
- `test_exit_use_georg_false` — `use_georg=False` gir eksakt samme
  ikke-GEORG-tilstandsbane som `build_matrices_v3_forward` (atol=1e-8), à la
  `test_exit_psi_PL_0`.
- `test_georg_indikator_kobling` — rad 20 har korrekte ω-koeffisienter på de
  respektive (forventnings)leddene; Z_MP har λ_Z=0.75 i G1 og Psi[·,E_i]=1.
- `test_observasjon_uendret` — `build_H`-kompatibel form (nye GEORG-tilstander
  gir null-kolonner i H), jf. `test_build_H_plt_form`.

**Krav:** Alle eksisterende tester må fortsatt passere (GEORG er additiv; v3
urørt).

---

## 6. Leveranser

1. Dette planleggingsdokumentet (✅ denne filen, versjonskontrollert).
2. **Etter PE-godkjenning:** `build_matrices_georg()` i `equations.py` +
   `tests/test_georg.py` + IRF-sammenligningsrapport (§4) + oppdatering av
   `PROSJEKTPLAN.md` (Fase 3-kontekst / GEORG-læringssteg).

---

## 7. Akseptansekriterier

- [ ] GEORG-regelens form, indikatorer, koeffisienter og λ_Z er korrekt
      dokumentert mot Staff Memo 15/2025 (§1 ✅).
- [ ] Dekningsanalysen identifiserer alle hull med minste endring og eksplisitt
      eskaleringsmarkering (§2 ✅).
- [ ] `build_matrices_georg` bygger ved siden av v3 uten å endre v3/v3_forward
      eller mimicking rule.
- [ ] BK-stabil med Tabell 4-koeffisienter (`max|eig(T)| < 1`).
- [ ] Exitstrategi (`use_georg=False`) reproduserer v3_forward eksakt.
- [ ] IRF-protokollen (§4) gir et entydig svar: **politikkregel vs
      transmisjon**.
- [ ] Alle nye og eksisterende tester passerer.
- [ ] PE har godkjent NZ-utvidelsen (§8) **før** implementering starter.

---

## 8. PE-eskaleringspunkter (eksplisitt)

> **STOPP-punkt før implementering.** Følgende krever PE-godkjenning iht.
> AGENTER.md §«Eskaleringsregler» og CLAUDE.md «Hva som ikke skal endres uten
> godkjenning»:

1. **Modellens dimensjon (NZ).** GEORG krever nye tilstander for 8-kv.
   valutakursvekst (~7), 4-kv. lønnskostnadsvekst, 4-kv. inflasjon (2 lagg) og
   AR(1) pengepolitikksjokk Z_t. Anslått NZ_GEORG ≈ 61–63. **Eskaleringspunkt
   pkt. 1 (modellens dimensjon).**
   - *Presedens:* `v3_plt`, `altB`, `pi4chain`, `v4` ble alle innført som egne
     byggere med egen dimensjon etter PE-godkjenning. GEORG følger samme
     mønster (ny bygger, v3 urørt).

2. **Scope.** GEORG er et nytt analysespor (Fase 3-kontekst). Bekreft at dette
   er innenfor gjeldende prioritering. **Eskaleringspunkt pkt. 8
   (scope-utvidelse).**

3. **Ikke utløst, men dokumentert:** Læringssteget krever **ingen** ny
   observasjonsvariabel og **ingen** prior-/estimeringsendring (GEORG
   estimeres ikke mot data — koeffisientene er gitt av Tabell 4). Hvis vi
   senere ønsker å *re-estimere* GEORG-koeffisientene via IRF-matching mot vår
   egen modell, blir det et separat eskaleringspunkt.

**Spørsmål til PE:**
- Godkjenner du NZ-utvidelsen for en ny `build_matrices_georg`-bygger (v3
  urørt), med eksakt dimensjon rapportert tilbake før innsjekk?
- Skal 8-kv. valutakursvekst implementeres fullt (mest trofast mot GEORG, men
  ~7 nye tilstander), eller godtar du en kompakt tilnærming (f.eks. akkumulator
  + ett 8-kv.-lagg) for å holde dimensjonen nede?

---

## 9. Utfallsmatrise — hva betyr hvert GEORG-resultat for neste fase?

| Utfall (IRF-sammenligning) | Tolkning | Konsekvens for neste fase |
|----------------------------|----------|----------------------------|
| **GEORG ≈ NB Figur 1** (og GEORG ≠ vår mimicking-IRF) | Avviket var drevet av **politikkregelen**. Vår transmisjon er i orden; mimicking rule var feil objekt å sammenligne med. | Bytt referanse-politikk: gå mot **tapsfunksjonsbasert optimal politikk** (vektene `lambda_dr=0.40`, `lambda_y=0.30`, `lambda_lr=0.02` finnes i `parameters.py`). GEORG blir det operative regel-uttrykket. Begrensning 6 (I_R.q12) revurderes — kan være artefakt av mimicking rule. |
| **GEORG ≈ vår mimicking-IRF** (og begge ≠ NB) | Avviket består uavhengig av regel → drevet av **transmisjonen**. | Prioriter **modell-fidelitet**: likningsaudit mot complete documentation, oljesektor/GPFG-kanal mot `sigma_rp`-dominans, datagrunnlag (KPI-JAE, oljepris som observabel). Politikkregel er ikke flaskehalsen. |
| **GEORG delvis nærmere NB** (noen variabler ja, andre nei) | Blandet: regel forklarer noen kanaler (typisk I_R-bane/persistens), transmisjon forklarer andre (typisk RER-magnitude, jf. `sigma_rp`). | Todelt neste fase: (a) adoptér GEORG/optimal politikk for rente-banen, (b) fiks transmisjonskanalene der avviket består (mest sannsynlig UIP/`sigma_rp`). |
| **GEORG ustabil / BK feiler** med Tabell 4-koeffisienter | Vår transmisjon tåler ikke NBs optimale-politikk-koeffisienter → strukturell forskjell mellom vår modell og NBs NEMO. | Sterkt signal om transmisjonsproblem. Eskaler funn til PE; vurder likningsaudit før videre politikk-arbeid. |

---

## 10. Etter GEORG (kort skisse — startes ikke nå)

NEMO-fidelitet, prioritert etter GEORG-konklusjonen:

1. **Tapsfunksjonsbasert optimal politikk** (vekter i `parameters.py`:
   `lambda_dr=0.40`, `lambda_y=0.30`, `lambda_lr=0.02`).
2. **Oljesektor/GPFG-kanal** mot `sigma_rp`-dominansen (FEVD: 22% BNP, 88% RER).
3. **Likningsaudit** mot complete documentation (`nemo_complete_documentation_2019.pdf`).
4. **Datagrunnlag:** KPI-JAE (allerede i pipeline), oljepris som observabel.
5. **Blokksampling** for ESS (rho-klusteret er fortsatt ESS-flaskehals, ESS/n≈0.003–0.0055 < krav 0.02).

---

*PE godkjente 2026-06-04 (Godkjenn+implementer, full 8-kv.-window).
Implementering fullført — se §11.*

---

## 11. Resultat — implementering og IRF-funn (2026-06-04)

### Implementering

- `build_matrices_georg()` lagt til i `equations.py` ved siden av v3 (v3,
  v3_forward og mimicking rule **urørt**). NZ_GEORG = **64** (14 nye tilstander:
  π_{t-2}, 2×πW-lagg, 3×a-lagg, 7×s-lagg for full 8-kv. window, AR(1)-Z).
- GEORG-koeffisienter (Tabell 4) + λ_Z=0.75 lagt til `parameters.py`.
- `build_H_georg()` i `mcmc.py`. `tests/test_georg.py`: 9 tester, alle grønne.
  Full suite: **106 passed, 3 xfailed** (uendret).
- BK-stabil med Tabell 4-koeffisientene: max|eig(T)| = **0.9890**.
- Exitstrategi verifisert: `use_georg=False` reproduserer v3_forward-kjernen
  eksakt (atol=1e-8).

### IRF for pengepolitikksjokk — GEORG vs vår mimicking rule

Normalisert til styringsrente-topp = 1 (jf. Spor B5-konvensjon):

| Variabel | q0 (G\|M) | q4 (G\|M) | q8 (G\|M) | q12 (G\|M) | Topp GEORG |
|----------|-----------|-----------|-----------|------------|------------|
| I_R  | +0.68\|+1.00 | +0.77\|+0.14 | +0.25\|+0.02 | +0.17\|+0.02 | +1.00 @ q2 |
| π    | −0.06\|−0.08 | −0.07\|−0.01 | +0.00\|+0.01 | +0.02\|+0.01 | −0.09 @ q2 |
| Y    | −0.31\|−0.45 | −0.44\|−0.07 | −0.05\|+0.04 | +0.04\|+0.05 | −0.55 @ q2 |
| q_H  | −0.74\|−1.08 | −1.84\|−0.48 | −0.13\|+0.33 | +0.74\|+0.48 | −2.02 @ q3 |
| RER  | −0.73\|−1.08 | −0.61\|+0.01 | +0.21\|+0.20 | +0.36\|+0.19 | −1.02 @ q1 |
| w    | +0.03\|+0.05 | +0.10\|+0.03 | −0.06\|−0.08 | −0.26\|−0.15 | −0.48 @ q19 |

(G = GEORG, M = mimicking rule / `build_matrices_v3_forward`, kj41-kalibrering.)

### Konklusjon — politikkregel vs. transmisjon (BLANDET utfall)

**Begge spiller en rolle — utfallet er det «blandede» scenariet i §9:**

1. **Politikkregelen forklarer rentebanens FORM.** GEORG gir en **pukkelformet,
   persistent** rentebane (topp ved q2, fortsatt +0.77 ved q4, +0.25 ved q8),
   mens mimicking rule topper umiddelbart (q0) og faller raskt (+0.14 ved q4).
   Den pukkelformede, gradualistiske banen er nettopp signaturen til NBs
   fremoverskuende optimale politikk. Vår mimicking rule var altså for
   front-lastet — **en betydelig del av NB-avviket var drevet av regelen.**
   Output- og boligpris-responsen blir tilsvarende større og mer forsinket,
   konsistent med en mer persistent rentebane.

2. **Transmisjonen forklarer rest-avviket ved lang horisont.** Begrensning 6
   (I_R.q12): NB Memo 3/2024 Figur 1 viser at renten snur svakt *negativt*
   (~−0.15) ved q12. GEORG gir I_R.q12 = **+0.17** (mimicking +0.02). GEORG
   flytter altså i riktig retning (mer persistens) men **reproduserer ikke
   fortegnsskiftet** ved lang horisont. Denne rest-avviket — og den vedvarende
   RER-responsen — består uavhengig av regelen og peker mot **transmisjonen**
   (mest sannsynlig UIP/`sigma_rp` og manglende mean-reversion-kanal).

### Implikasjon for neste fase

Todelt (jf. §9, «GEORG delvis nærmere NB»):

- **(a) Adoptér GEORG/optimal-politikk-banen** for rente-responsen — den
  pukkelformede banen er en reell forbedring mot NB og bør være referanse
  fremfor AR(1)-mimicking rule. Begrensning 6 revurderes: den er *delvis*
  artefakt av mimicking rule (forma), men *ikke fullt* (fortegnsskiftet består).
- **(b) Fiks transmisjonskanalene** der avviket består: I_R.q12-fortegnsskiftet
  og RER-magnitude → prioriter UIP/`sigma_rp`-diagnosen (oljesektor/GPFG-kanal)
  og likningsaudit mot complete documentation.

**Forbehold:** Sammenligningen mot NB Figur 1 er kvalitativ (figuren finnes ikke
som rådata i repoet; q12≈−0.15 er hentet fra B5-/begrensningsdokumentet).
GEORG-koeffisientene er NBs IRF-matchede verdier mot *deres* NEMO; en framtidig
re-matching mot *vår* modell ville være et separat (eskalerings)steg.
