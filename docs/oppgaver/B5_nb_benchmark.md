# Oppgave B5 — NB Memo 3/2024 Benchmark

**Spor:** B (Numerisk verifikasjon)
**Ansvarlig:** NUM, med DSGE-bidrag på tolkning
**Fase:** 0.5 (aktiv)
**Prioritet:** Høy — rask diagnose av om posterior er rimelig

---

## Bakgrunn

NB Memo 3/2024 Figur 1 viser impulssvar (IRF) fra et pengepolitikkssjokk i
Norges Banks operative NEMO-modell. Vi ønsker å sammenligne vår implementerings
respons mot NB-figuren for å avdekke systematiske avvik.

Kjent mistanke: `sigma_rp = 0.016` (mot K&M 0.006) gir for stor RER-respons.

---

## Steg

### Steg 1 — Punkt-estimat

1. Last `data/results/chain_v3_v2_posterior.json`
2. Bruk posterior-mean for alle 17 estimerte parametre
3. Bygg `build_matrices_v3(Pt)` og løs med `bk_solve`
4. Kjør pengepolitikkssjokk: `E_i = 0.0025` (25 bp)
5. Normaliser slik at styringsrenten (I_R) topper på +1 pp (skaler alle IRF tilsvarende)
6. Plot 5 variabler over 20 kvartaler: Y, PI, I_R, RER, Q_H

### Steg 2 — Usikkerhetsbånd (ny)

**Problem:** `chain_v3_v2_posterior.json` inneholder kun summary-statistikk (mean/std/pctl),
ikke de fulle 200 000 trekkene.

**Løsning:** Bruk Gaussian-approksimering basert på posterior mean og std.
Trekk 500 pseudo-samples:

```python
rng = np.random.default_rng(2024)
samples = rng.multivariate_normal(posterior_mean, np.diag(posterior_std**2), size=500)
```

For hvert sample: bygg modell, løs, beregn IRF. Beregn median + 5/95-prosentilbånd.

**Merk:** Gaussian-approksimering er forenklet (ignorerer skjevhet og korrelasjon mellom
parametre). Fulle posterior-korrelasjoner krever ny MCMC-kjøring (PE-godkjenning).

### Steg 3 — Avvik-rapport

Sammenlign kvantitativt med NB Memo 3/2024 Figur 1:

| Variabel | Vår modell (topp/bunn) | NB Memo | Avvik |
|----------|------------------------|---------|-------|
| BNP-gap  | ?                      | ?       | ?     |
| KPI      | ?                      | ?       | ?     |
| Rente    | +1 pp (norm.)          | +1 pp   | 0     |
| RER      | ?                      | ?       | ?     |
| Boligpris| ?                      | ?       | ?     |

Spesifikk hypotese å teste:
- **Hypotese RER**: RER-topp er for stor fordi sigma_rp-overestimering
  forsterker UIP-kanalen. Test: kalibrert sigma_rp=0.006 vs. estimert 0.016
  i RER-IRF ved ellers samme posterior.

---

## Akseptansekriterier

- [ ] IRF-plot lagret i `data/results/B5_nb_benchmark.png`
- [ ] Avvik-tabell dokumentert med konkrete tall
- [ ] Hypotese om RER-avvik og sigma_rp enten bekreftet eller avkreftet
- [ ] Funn rapportert til PL for videre eskalering til PE

---

## Avhengigheter

- Ingen ny MCMC-kjøring nødvendig (bruker eksisterende posterior summary)
- Krever: `data/results/chain_v3_v2_posterior.json`
- Referanse: `docs/references/nb_memo_3_2024_haandbok.pdf`
