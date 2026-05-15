# NEMO — Modellbeskrivelse

Implementasjon av NEMO (Norwegian Economy MOdel) basert på
Kravik & Mimir (2019) "Navigating with NEMO", Staff Memo 5/2019.

Modellen er et mellomstort DSGE-system for norsk åpen økonomi med
boligmarked, heterogene husholdninger (Ricardian / ikke-Ricardian)
og finansiell friksjon (LTV-begrensning).

---

## Tilstandsrom

Modellen er formulert som et lineært tilstandsrom:

```
Γ₀ z_t = Γ₁ z_{t-1} + Ψ ε_t + Π η_t
```

løst ved Blanchard-Kahn til:

```
z_t = T z_{t-1} + R ε_t
```

| Dimensjon | Verdi |
|-----------|-------|
| NZ — antall tilstandsvariabler | 48 |
| NE — antall sjokk | 13 |

---

## Tilstandsvariabler (NZ = 48)

### Endogene variabler (indeks 0–36)

| Indeks | Symbol | Beskrivelse |
|--------|--------|-------------|
| 0  | PI       | Konsumprisinflasjon (kvartalsvekst) |
| 1  | C_W      | Konsum, Ricardian (lønnstaker) |
| 2  | C_NW     | Konsum, ikke-Ricardian (låntaker) |
| 3  | C        | Aggregert konsum |
| 4  | PIW      | Lønnsinflasjon |
| 5  | W        | Reallønn |
| 6  | Q_H      | Boligpris (realterm) |
| 7  | H_W      | Boligbeholdning, Ricardian |
| 8  | H_NW     | Boligbeholdning, ikke-Ricardian |
| 9  | Y        | BNP-gap |
| 10 | L        | Sysselsetting (timer) |
| 11 | K        | Kapitalstock |
| 12 | INV      | Næringslivinvestering |
| 13 | MC       | Marginal kostnad |
| 14 | Q_K      | Tobins Q (kapitalpris) |
| 15 | RER      | Reell valutakurs (REER-gap) |
| 16 | X        | Eksport |
| 17 | M        | Import |
| 18 | PM       | Importpris |
| 19 | S        | Nominell valutakurs (depresieringsrate) |
| 20 | I_R      | Styringsrente (Norges Bank) |
| 21 | I_D      | Innskuddsrente |
| 22 | I_L_W    | Utlånsrente, Ricardian |
| 23 | I_L_NW   | Utlånsrente, ikke-Ricardian |
| 24 | B_W      | Gjeld, Ricardian |
| 25 | B_NW     | Gjeld, ikke-Ricardian (LTV-begrenset) |
| 26 | NB       | Netto utenlandsgjeld |
| 27 | G        | Offentlig etterspørsel |
| 28 | PO       | Oljepris (realterm) |
| 29 | K_L      | Laggede kapital (K_{t-1}) |
| 30 | INV_L    | Lagget investering (INV_{t-1}) |
| 31 | H_W_L    | Lagget bolig Ricardian (H_W_{t-1}) |
| 32 | H_NW_L   | Lagget bolig ikke-Ricardian (H_NW_{t-1}) |
| 33 | I_R_L    | Lagget styringsrente (I_R_{t-1}) |
| 34 | RER_L    | Lagget valutakurs (RER_{t-1}) |
| 35 | W_L      | Lagget reallønn (W_{t-1}) |
| 36 | PI_L     | Lagget inflasjon (PI_{t-1}) |

### Eksogene sjokkprosesser (indeks 37–47)

| Indeks | Symbol | Sjokk | AR-parameter |
|--------|--------|-------|--------------|
| 37 | A         | TFP-sjokk | rho_A |
| 38 | EPS_C     | Preferansesjokk (konsum) | rho_C |
| 39 | EPS_H     | Boligpreferanse-sjokk | rho_H |
| 40 | EPS_G     | Offentlig etterspørselssjokk | — (kalibrert) |
| 41 | YS        | Utenlandsk etterspørselssjokk | rho_Ys |
| 42 | EPS_RP    | Risikopremiesjokk | rho_rp |
| 43 | PI_STAR   | Utenlandsk inflasjon | — |
| 44 | I_STAR    | Utenlandsk rente | — |
| 45 | EPS_PHI_H | LTV-sjokk (boligkravssjokk) | — |
| 46 | EPS_PREM  | Premiesjokk (rentespread) | — |
| 47 | EPS_I_ADJ | Investerings-justeringskostnad-sjokk | — |

---

## Sjokk (NE = 13)

| Indeks | Navn | Psi-kolonne | sigma-parameter |
|--------|------|-------------|-----------------|
| 0  | E_A     | TFP           | sigma_A (fast=0.006) |
| 1  | E_C     | Konsum        | sigma_C |
| 2  | E_H     | Bolig         | sigma_H |
| 3  | E_G     | Offentlig     | — |
| 4  | E_O     | Oljepris      | sigma_O |
| 5  | E_I     | Inv.adj.      | — |
| 6  | E_Ys    | Utenl. BNP    | sigma_Ys |
| 7  | E_rp    | Risikopremie  | sigma_rp |
| 8  | E_P     | Kostprissjokk | sigma_P |
| 9  | E_phi_h | LTV-sjokk     | — |
| 10 | E_prem  | Premiesjokk   | — |
| 11 | E_i     | Pengepol.     | sigma_i |
| 12 | E_I_adj | Inv.adj.2     | — |

---

## Estimerte parametre (N = 17)

`sigma_A` kalibreres fast til K&M-verdi 0.006 (svakt identifisert).

| Parameter | K&M (2019) | Posterior mean | Beskrivelse |
|-----------|-----------|----------------|-------------|
| rho_A     | 0.804 | 0.738 | AR(1) TFP |
| rho_C     | 0.725 | 0.810 | AR(1) konsum-preferanse |
| rho_O     | 0.874 | 0.409 | AR(1) oljepris |
| rho_Ys    | 0.783 | 0.818 | AR(1) utenlandsk BNP |
| rho_rp    | 0.737 | 0.769 | AR(1) risikopremie |
| rho_H     | 0.694 | 0.793 | AR(1) boligpreferanse |
| sigma_C   | 0.030 | 0.013 | Std.avv. konsum-sjokk |
| sigma_O   | 0.079 | 0.108 | Std.avv. oljeprissjokk |
| sigma_Ys  | 0.011 | 0.006 | Std.avv. utenl. BNP-sjokk |
| sigma_rp  | 0.006 | 0.016 | Std.avv. risikopremie-sjokk ⚠ |
| sigma_i   | 0.0003 | 0.0003 | Std.avv. pengepol.-sjokk |
| sigma_P   | 0.003 | 0.006 | Std.avv. kostprissjokk |
| sigma_H   | 0.050 | 0.154 | Std.avv. boligsjokk |
| psi_R     | 0.666 | 0.963 | Renteglatting ⚠ |
| psi_P1    | 0.292 | 0.291 | Inflasjonsvekt i Taylor |
| psi_Y     | 0.242 | 0.263 | BNP-gap-vekt i Taylor |
| h_c       | 0.938 | 0.992 | Konsumvane (habit) ⚠ |

⚠ = avviker vesentlig fra K&M, se `docs/oppgaver/A_funn_rapport.md` og C2-rapport.

---

## Modellfiler

| Fil | Innhold |
|-----|---------|
| `src/nemo/model/equations.py` | `build_matrices_v3()` — G0, G1, Ψ, Π |
| `src/nemo/model/parameters.py` | Kalibrerte parametre (steady-state) |
| `src/nemo/solver/blanchard_kahn.py` | BK-løser og IRF-beregning |
| `src/nemo/estimation/mcmc.py` | Adaptiv RWMH, Kalman-filter |
| `src/nemo/estimation/kalman.py` | Kalman-filter (separat modul) |

---

## Nøkkelegenskaper

### Observasjonslikning

14 observerbare serier kobles til tilstandsrommet via `H`-matrisen:

```
y_t = H z_t + v_t,  v_t ~ N(0, Sv)
```

Serier: Δy, Δc, ΔInv, Δx, Δm, π, Δw, i_R, i_3m, Δs, ΔpO, ΔyS, Δh, Δb

### COVID-hull

Likelihood splittes i to blokker:
- Pre-COVID: ≤ 2019Q4 (75 kv.)
- Post-COVID: ≥ 2022Q1 (15 kv.)

Kalman-filteret reinitialiseres mellom blokkene (8 kvartaler hull).

### Blanchard-Kahn

Krav: `max|eig(T)| < 1.0` (BK-stabilitet).

Aktiv versjon (`build_matrices_v3`) er stabil med posterior mean.

---

## Kjente svakheter (Fase 0.5)

Se `docs/oppgaver/A_funn_rapport.md` for full listeundersøkelse.

1. **sigma_rp overestimert (0.016 vs. K&M 0.006)** — sannsynlig H3: manglende UIP-dynamikk
2. **h_c = 0.992 ved priorbegrensning** — H4: likelihood-rygg langs h_c→1 (a3_W → 0)
3. **psi_R = 0.963 ved priorbegrensning** — H4 og/eller H3 (se C2-rapport)
4. **Systemic lag-state bug** — 5 likninger bruker G1 på lag-tilstander (2-periode lag). Fikset for ligning 20 (mimicking rule, Spor A4b). Rader 5, 7, 8, 11, 12 gjenstår (krever PE-godkjenning).
5. **Ressursbetingelse ikke oppfylt** — CY+IY+GY+XY−MY = 0.84, avvik −0.16 (Spor A5)
6. **ESS/n ≈ 0.006** — svært treig MCMC-blanding. Blokksampling sigma_C/h_c (r=−0.811) anbefales i Fase 2.

---

## Referanser

- Kravik & Mimir (2019) "Navigating with NEMO", Staff Memo 5/2019, Norges Bank
- Norges Bank Memo 3/2024 — IRF-benchmark (Figur 1)
