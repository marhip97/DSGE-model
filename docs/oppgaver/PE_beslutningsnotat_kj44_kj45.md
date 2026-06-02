# PE-beslutningsnotat — psi_R-persistens og I_R.q12 etter kj44/kj45

**Avsender:** STAT/DSGE
**Mottaker:** Prosjekteier (PE)
**Dato:** 2026-06-02
**Status:** Krever PE-beslutning (valg mellom to veier)

---

## Sammendrag

To Fase 2-kjøringer har avklart to åpne spørsmål rundt pengepolitikk-IRF-en:

1. **kj44** (logit-reparametrisering): psi_R presser **genuint** mot 0.99 — det er ikke
   en numerisk samplingsartefakt. Modellen *trenger* nær-unit-root renteglatting.
2. **kj45** (AR(2) Taylor-regel): et andregrads rentelagg `psi_R2` ble estimert til
   ~0 og **forkastet av data**. Mean-reversion via autoregressiv struktur er ikke mulig.

Konsekvens: I_R.q12-feilfortegnet (begrensning 6) lar seg ikke fikse innenfor mimicking
rule-rammeverket. PE må velge om vi (A) implementerer en strukturell PLT-kanal, eller
(B) aksepterer begrensningen og lukker den autoregressive sporet.

---

## Dokumentert bevisgrunnlag

### Nøkkelresultater kj41 → kj45

| Kjøring | Endring | psi_R | psi_R2 | ESS/n | RMSE | I_R.q12 (NB: −0.15) |
|---------|---------|-------|--------|-------|------|---------------------|
| kj41 | best-fit baseline | 0.9490 | — | 0.0033 | **0.2771** | +0.85 |
| kj44 | logit-reparam psi_R | 0.9894 | — | 0.0054 | 0.3642 | +0.851 |
| kj45 | AR(2) psi_R2 fri | 0.9894 | **−0.0003** | 0.0033 | 0.3633 | +0.848 |

Alle tre konvergerte godt (PSRF ≤ 1.01). ESS/n forblir under kravet 0.02.

### Funn 1 — psi_R-grensen er genuin (kj44)

Logit-transformen mapper (0.50, 0.99) → (−∞, ∞) og fjerner grenserefleksjon.
Posterioret i logit-rom har en lang høyrehale (max=11.4) **uten indre modus** og
presser mot 0.99 med sd=0.0004. Dette beviser at likelihood-ryggen fortsetter inn
i det forbudte området over 0.99 — modellen vil ha enda høyere persistens.

### Funn 2 — AR(2) mean-reversion forkastes (kj45)

psi_R2 fikk prior sentrert på −0.10 (mean-reversion) og startet på −0.05, men
likelihood drev den til 0.0 (øvre grense, sd=0.0003). Andregrads-lagget er en død
tilstand: modellen oppfører seg eksakt som AR(1). I_R.q12 forble +0.85.

### Tolkning

Renten i NEMO følger ε_i-sjokket med geometrisk forfall (psi_R≈0.99) og kan ikke
reversere fortegn fordi det ikke finnes noen reverserende kraft i likningssettet.
NB Memo 3/2024 viser at NB normaliserer renten under nøytralnivå etter ~3 år
(I_R.q12=−0.15). Det krever en mekanisme som «husker» det akkumulerte renteoverskuddet
— ikke bare rentenivået i forrige kvartal.

---

## To veier videre

### Vei A — Prisnivåmål (PLT) i Taylor-regelen

Renten reagerer på akkumulert prisnivå-gap (ikke bare inflasjon):

```
i_t = psi_R·i_{t-1} + (1-psi_R)·[psi_P·π_t + psi_PL·(p_t − p*_t) + psi_Y·y_t] + ε_i
```

Prisnivå-gap `(p_t − p*_t)` er en integrert tilstand som gir genuin mean-reversion:
etter et innstrammende sjokk faller prisnivået under mål, og psi_PL > 0 trekker
da renten ned under nøytral — nettopp reverseringen NB viser.

| | |
|---|---|
| **Fordel** | Eneste gjenværende kandidat for I_R.q12 < 0; teoretisk velfundert (Woodford) |
| **Kostnad** | Ny tilstandsvariabel (NZ 50 → 51), én ny parameter psi_PL, ny PE-runde |
| **Risiko** | PLT kan endre andre IRF-er; krever full IRF-tegn-revalidering (15 krav) |
| **Innsats** | ~1 kjøring + validering. Eksitmulighet: psi_PL=0 → ren inflasjonsmål |

### Vei B — Aksepter begrensning 6, lukk autoregressivt spor

Behold kj41 som best-fit. Dokumentér I_R.q12-feilfortegnet som en kjent, forklart
begrensning. Fokusér Fase 2 på de tre formålene modellen gjør godt:

| | |
|---|---|
| **Fordel** | Ingen ny modellrisiko; modellen er allerede validert for FEVD/HD/sjokkanalyse |
| **Kostnad** | Pengepolitikk-IRF utover ~8 kvartaler er ikke kvantitativt pålitelig |
| **Risiko** | Ingen ny — begrensningen er allerede dokumentert |
| **Innsats** | Kun dokumentasjon; frigjør tid til ESS-forbedring (blokksampling/HMC) |

---

## Anbefaling

**Vei B på kort sikt, Vei A som separat eksperiment ved behov.**

Begrunnelse: kj41 er allerede en gyldig produksjonsmodell for sjokkanalyse, FEVD og
historisk dekomposisjon (de tre dokumenterte formålene). I_R.q12-feilfortegnet påvirker
kun langsiktig rente-normalisering i pengepolitikk-IRF, som allerede er merket som
overvurdert ~6× (begrensning 1). Å åpne en PLT-kanal nå introduserer modellrisiko og
krever full revalidering uten å forbedre de primære bruksområdene.

Det mer presserende åpne punktet er **ESS/n < 0.02** (begrensning 2), som gjelder
konfidensbånd for *alle* parametre. Blokksampling eller HMC (C8-diagnose) gir mer verdi
for produksjonskvaliteten enn en PLT-utvidelse.

**Spørsmål til PE:**
1. Velg Vei A (implementer PLT, NZ→51) eller Vei B (aksepter begrensning 6)?
2. Hvis B: skal neste Fase 2-innsats være ESS-forbedring (blokksampling/HMC)?
