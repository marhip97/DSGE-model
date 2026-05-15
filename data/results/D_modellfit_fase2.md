# D — Modell-fit-analyse: NEMO v3 vs. faktiske data

**Estimeringsperiode:** 2001Q2–2019Q4 + 2022Q1–2025Q3  
**COVID utelatt:** 2020Q1–2021Q4 (8 kvartaler)

To mål:
- **Filtrert** (H@z_{t|t}): modellens forklaring etter å ha sett data t.o.m. t — mål på in-sample tilpasning
- **Predikert** (H@z_{t|t-1}): ett-stegs fremover, uten å se data i t — strengere mål på prediksjonskraft

## Fit-statistikk per serie

| Serie | Beskrivelse | Filt. R² | Filt. korr | Pred. R² | Pred. korr |
|-------|-------------|----------|------------|----------|------------|
| `dy_obs` | BNP-vekst (Δy) | 0.282 ⚠ | 0.631 | -2.459 ✗ | 0.172 |
| `dc_obs` | Konsum-vekst (Δc) | 0.521 | 0.732 | -0.073 ✗ | -0.054 |
| `dinv_obs` | Invest.-vekst (ΔInv) | 0.813 ✓ | 0.930 | -0.725 ✗ | 0.151 |
| `dx_obs` | Eksport-vekst (Δx) | 0.213 ⚠ | 0.464 | -0.162 ✗ | -0.088 |
| `dm_obs` | Import-vekst (Δm) | 0.504 | 0.711 | -0.199 ✗ | 0.357 |
| `pi_obs` | KPI-inflasjon (π) | 0.933 ✓ | 0.992 | 0.007 ⚠ | 0.087 |
| `dw_obs` | Lønnsvekst (Δw) | 0.924 ✓ | 0.963 | 0.720 ✓ | 0.864 |
| `i_R_obs` | Styringsrente (i_R) | 0.998 ✓ | 0.999 | 0.883 ✓ | 0.940 |
| `i_3m_obs` | 3m pengemarked (i_3m) | 0.997 ✓ | 0.999 | 0.874 ✓ | 0.935 |
| `ds_obs` | Valutakurs (Δs) | 0.666 | 0.870 | -0.010 ✗ | -0.071 |
| `dpO_obs` | Oljepris (ΔpO) | 0.939 ✓ | 0.997 | 0.018 ⚠ | 0.184 |
| `dyS_obs` | Utenl. BNP (ΔyS) | 0.819 ✓ | 0.932 | 0.645 | 0.852 |
| `dh_obs` | Boligpris (Δh) | 0.901 ✓ | 0.983 | 0.629 | 0.899 |
| `db_obs` | Gjeld (Δb) | 0.982 ✓ | 0.994 | 0.739 ✓ | 0.873 |

**Gj.snitt filtrert R²:** 0.749  
**Gj.snitt predikert R²:** 0.063  
**Gj.snitt filtrert korrelasjon:** 0.871


## Tolkning

Filtrert R² > 0.7 (✓) = god in-sample tilpasning.  
Filtrert R² < 0.3 (⚠) eller negativ (✗) = modellen klarer ikke å forklare variansen i serien, selv med full tilgang til data t.o.m. t.  
Predikert R² er alltid lavere enn filtrert — det er ventet.

> **Strukturell merknad:** K&M (2019) estimerte på data t.o.m. ~2019. Vi inkluderer 15 kvartal post-COVID (2022–2025) med Norges Banks rentehevingssyklus (0→4,5 %). Svak fit på realvariabler kan reflektere reelle strukturelle brudd etter 2020, ikke kun modellfeil.

