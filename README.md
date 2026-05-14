# NEMO — DSGE-modell for norsk økonomi

Bayesiansk estimert DSGE-modell for norsk økonomi, inspirert av
Kravik & Mimir (2019). 48 tilstandsvariable, 13 strukturelle sjokk,
estimert på 14 observerte serier fra 2001Q1.

## Hva modellen kan

- **Sjokkanalyse**: IRF og FEVD for åtte hovedsjokk (TFP, konsum,
  prismarkup, oljepris, ettersp., risikopremie, pengepolitikk, bolig)
- **Historisk dekomposisjon**: hvilke sjokk har drevet BNP, inflasjon,
  styringsrente, valutakurs og boligpris fra 2001 til i dag
- **Sjokk-betinget prognose**: 16 kvartaler med 50 % og 90 % bånd
- **Realtid nowcast**: månedlig oppdatering når KPI / styringsrente
  publiseres mellom kvartalene

## Installasjon

```bash
git clone <repo>
cd nemo
pip install -e .[dev]
```

Krever Python 3.11+.

## Bruk

### Komplett pipeline

```bash
python -m nemo run --refresh-data --estimate --analyse
```

### Steg for steg

```bash
# 1. Hent ferske data fra SSB, Norges Bank, FRED
python -m nemo data refresh

# 2. Verifiser at modellen løser (kalibrert)
python -m nemo solve

# 3. Estimer (~2 timer, 200k MCMC-trekk)
python -m nemo estimate --draws 200000 --burnin 20000

# 4. Generer IRF, FEVD, prognose
python -m nemo analyse \
    --posterior data/results/posterior_v4.json \
    --data data/processed/nemo_data.csv

# 5. Bygg dashboard
python -m nemo dashboard build
```

### Tester

```bash
pytest tests/
```

## Mappestruktur

```
src/nemo/
├── model/        Likningssystem, parametere, steady state
├── solver/       Blanchard-Kahn / QZ-løser
├── data/         API-klienter (SSB, NB, FRED) + pipeline
├── estimation/   Kalman-filter, prior, MCMC, diagnostikk
├── analysis/     IRF, FEVD, dekomposisjon, prognose, nowcast
└── dashboard/    HTML-rapport

data/
├── raw/          Cached API-responser (i .gitignore)
├── processed/    Endelig observasjonssett (committed)
└── results/      Posterior, IRF, FEVD (committed)

tests/            Pytest-tester
notebooks/        Utforskende analyse
docs/             MODEL.md, DATA.md, ESTIMATION.md
```

## Datakilder

| Variabel               | Kilde       | Frekvens |
|------------------------|-------------|----------|
| BNP fastland, konsum, investering, eksport, import | SSB 09189 | Kvartal |
| KPI / KPI-JAE          | SSB 03013 / 10235 | Måned (→ kvartal) |
| Sysselsetting (AKU)    | SSB 05111   | Kvartal |
| Lønnsindeks            | SSB 09786   | Kvartal |
| Boligprisindeks        | SSB 07241   | Kvartal |
| Styringsrente, NIBOR 3M | Norges Bank | Daglig (→ kvartal) |
| NOK/EUR                | Norges Bank | Daglig (→ kvartal) |
| K2 husholdninger       | Norges Bank | Måned (→ kvartal) |
| Brent oljepris         | FRED        | Måned (→ kvartal) |
| BNP handelspartnere    | FRED / IMF  | Kvartal |

## Modellstruktur

Tilstandsvektor (48):

- Husholdninger (sparere W, låntakere NW): konsum, bolig, lønn
- Produksjon: BNP, sysselsetting, kapital, investering, Tobin's Q
- Valuta og handel: RER, eksport, import, importpris
- Finansiell sektor: styringsrente, innskudd/utlånsrenter,
  husholdningsgjeld (LTV-bindende), bankkapital
- Offentlig sektor: offentlig konsum, oljepris (AR(1))
- 8 eksogene AR(1)-prosesser

Pengepolitikk modelleres som "mimicking rule" (NEMO seksjon 2.13) med
estimerte koeffisienter, ikke en standard Taylor-regel.

## Estimering

- Adaptiv Random-Walk Metropolis-Hastings
- 20 000 burn-in + 200 000 produksjonstrekk
- COVID-hull: likelihood splittes i pre-2020Q1 og post-2022Q1
- 17 parametere; `sigma_A` kalibreres fast (svakt identifisert)
- Konvergens: PSRF < 1.10, ESS/n > 0.02

## Lisens

Se `LICENSE`.

## Referanser

Kravik, E. M., & Mimir, Y. (2019). *Macroeconomic models for monetary
policy: A critical review from a finance perspective.* Norges Bank
Staff Memo 5/2019.
