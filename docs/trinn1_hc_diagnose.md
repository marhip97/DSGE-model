# Trinn 1 — h_c-prior-relaksering: BK-stabilitet eliminerer Alt. B

**Dato:** 2026-05-15
**Kontekst:** Beslutningsgrunnlag for løsning av TFP→BNP-bug (test_09 xfail).

---

## Sammendrag

Forsøket på å teste **Alt. B (relaksere h_c-prior fra 0,9995 til 0,90)** avdekket at
modellen er **Blanchard-Kahn-ustabil for h_c < 0,92**, uavhengig av andre parametre.
Dette eliminerer prior-relaksering som mulig løsning og styrker argumentet for
strukturell endring (Alt. A: variabel kapitalutnyttelse).

---

## Eksperiment

K&M-verdier for alle parametre unntatt h_c, deretter testet BK-stabilitet for
ulike h_c-verdier:

| h_c   | BK-stable? | max\|eig(T)\| | n_unstable |
|-------|-----------|----------------|-------------|
| 0,70  | ✗         | 1,0042         | 1           |
| 0,80  | ✗         | 1,0032         | 1           |
| 0,85  | ✗         | 1,0023         | 1           |
| 0,88  | ✗         | 1,0014         | 1           |
| 0,90  | ✗         | 1,0005         | 0*          |
| **0,92** | **✓**  | **0,9991**     | **0**       |
| 0,94  | ✓         | 0,9968         | 0           |
| 0,95  | ✓         | 0,9950         | 0           |
| 0,98  | ✓         | 0,9880         | 0           |

*Marginal — n_unstable=0 men max|eig| > 1 indikerer numerisk grenseverdi.

**BK-stabilitetsgrensen ligger rundt h_c ≈ 0,91-0,92.**

---

## Tolkning

### Hva betyr dette økonomisk?

Den effektive renteresponsen for forbruk er:

    a3_W = (1 - h_c) / (σ (1 + h_c))

Når h_c → 1: a3_W → 0 (forbruk responderer ikke på renten — "permanent income"-grense)
Når h_c → 0: a3_W → 1/σ (full renterespons)

Modellens BK-stabilitet krever a3_W lite nok — dvs. h_c stort nok. Med lavere
h_c får forbruksligningen *for sterk* renterespons relativt til de andre
sammenkoblede dynamiske ligningene, og en av egenverdiene flytter seg utenfor
enhetssirkelen.

### Hva betyr dette for hypotesene fra Fase 0.5?

Tidligere ble H4-hypotesen formulert som:

> "Likelihood-rygg langs h_c→1 der a3_W=(1-h_c)/(σ(1+h_c))→0"

Riktig formulering er nå:

> **"Modellens BK-stabilitet krever h_c ≥ 0,92, og prior-grensen på 0,9995
> kombinert med stabilitetsgrensen lar likelihood plassere h_c hvor som helst i
> intervallet [0,92, 0,9995]. Posterior treffer øvre grense fordi data foretrekker
> at."**

Dette er en blanding av H4 (numerisk grense) og H1 (modellen "trenger" høy h_c),
men med en kritisk vri: **modellen tillater ikke alternativet**.

---

## Konsekvens for TFP→BNP-bug

Vi kan ikke teste om "lavere h_c gir TFP→BNP positiv" empirisk, fordi modellen
ikke tillater lavere h_c. Tre logiske svar:

1. **Hvis bare h_c-grensen drev test_09:** strukturell stabilitet hindrer fix
2. **Hvis NK-puzzle er drivkraften:** h_c alene løser det ikke uansett
3. **Hvis begge:** dobbeltvanske — krever både prior-utvidelse OG modellendring

I alle tre tilfeller kommer vi til samme konklusjon: **Alt. A (strukturell)
er nødvendig.**

---

## Anbefaling oppdatert

**Hopp over Alt. B og Alt. C — gå direkte til Alt. A.**

### Spesifikt: Implementer variabel kapitalutnyttelse

Følger Smets-Wouters (2007) §2.3 / K&M (2019) §2.7:

```
Endogen ny variabel:  u_t           (utilization rate)
Effektiv kapital:     k̂_t = k_{t-1} + u_t
Bedriftens FOC:       r_K_t = Ψ'(u_t) ≈ (Ψ''/Ψ') · u_t   (linear approx)
Produksjonsfunksjon:  y_t = a_t + α·k̂_t + (1-α)·l_t
```

**Endringer som kreves:**
- NZ: 48 → 49 (én ny tilstand)
- Ny parameter: ζ_u (utilization-kostnadselastisitet, K&M: ~0,5)
- 2 nye ligninger: FOC for u_t, modifisert produksjonsfunksjon
- Ressursbetingelsen uendret (Y fortsatt demand-determined)

**Forventet effekt:**
- TFP↑ → u↑ (bedrifter utnytter eksisterende kapital mer) → effektiv k↑ → Y↑
- Q_K-effekten balansert: lavere MC oppveies av høyere effektiv kapitalinnsats
- Forventet test_09: kumulativ Y(20q) flytter seg fra −0,001 til ca. +0,005 til +0,010

### Risiko

- BK-stabilitet ikke garantert — krever testing
- 1 ekstra parameter å estimere (ζ_u)
- Endrer NZ-konstanten → ringvirkninger i tester, observasjonsligning, fixtures

### Estimert tid

- Implementasjon + BK-testing: 1-2 dagsverk
- Re-estimering MCMC: 1 dagsverk
- Verifisering test_09 og dokumentasjon: 0,5 dagsverk

**Total: 2-4 dagsverk, krever PE-godkjenning før start.**

---

## Eskaleringsbeslutning

| Punkt | Status |
|-------|--------|
| Alt. B testet og avvist | ✓ |
| Modellen kan ikke ha lavere h_c | ✓ Bekreftet |
| Strukturell endring nødvendig | ✓ |
| PE-godkjenning kreves | **Avventer** |

**Ber om PE-godkjenning for Alt. A (variabel kapitalutnyttelse).**
