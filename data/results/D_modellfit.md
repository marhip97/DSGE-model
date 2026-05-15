# D — Modell-fit-analyse: NEMO v3 vs. faktiske data

**Estimeringsperiode:** 2001Q2–2019Q4 + 2022Q1–2025Q3  
**COVID utelatt:** 2020Q1–2021Q4 (8 kvartaler)

To mål:
- **Filtrert** (H@z_{t|t}): modellens forklaring etter å ha sett data t.o.m. t — mål på in-sample tilpasning
- **Predikert** (H@z_{t|t-1}): ett-stegs fremover, uten å se data i t — strengere mål på prediksjonskraft

## Fit-statistikk per serie

| Serie | Beskrivelse | Filt. R² | Filt. korr | Pred. R² | Pred. korr |
|-------|-------------|----------|------------|----------|------------|
| `dy_obs` | BNP-vekst (Δy) | 0.224 ⚠ | 0.567 | -0.915 ✗ | 0.151 |
| `dc_obs` | Konsum-vekst (Δc) | 0.507 | 0.718 | -0.129 ✗ | -0.185 |
| `dinv_obs` | Invest.-vekst (ΔInv) | 0.012 ⚠ | 0.285 | 0.007 ⚠ | 0.152 |
| `dx_obs` | Eksport-vekst (Δx) | 0.239 ⚠ | 0.491 | -0.167 ✗ | -0.086 |
| `dm_obs` | Import-vekst (Δm) | 0.448 | 0.678 | 0.086 ⚠ | 0.335 |
| `pi_obs` | KPI-inflasjon (π) | 0.662 | 0.877 | 0.006 ⚠ | 0.084 |
| `dw_obs` | Lønnsvekst (Δw) | 0.890 ✓ | 0.945 | 0.519 | 0.763 |
| `i_R_obs` | Styringsrente (i_R) | 0.998 ✓ | 0.999 | 0.882 ✓ | 0.939 |
| `i_3m_obs` | 3m pengemarked (i_3m) | 0.997 ✓ | 0.999 | 0.873 ✓ | 0.934 |
| `ds_obs` | Valutakurs (Δs) | 0.657 | 0.868 | -0.009 ✗ | -0.062 |
| `dpO_obs` | Oljepris (ΔpO) | 0.970 ✓ | 0.998 | 0.016 ⚠ | 0.187 |
| `dyS_obs` | Utenl. BNP (ΔyS) | 0.829 ✓ | 0.933 | 0.656 | 0.856 |
| `dh_obs` | Boligpris (Δh) | 0.476 | 0.725 | 0.320 | 0.633 |
| `db_obs` | Gjeld (Δb) | 0.662 | 0.819 | 0.394 | 0.642 |

**Gj.snitt filtrert R²:** 0.612  
**Gj.snitt predikert R²:** 0.181  
**Gj.snitt filtrert korrelasjon:** 0.779


## Tolkning

Filtrert R² > 0.7 (✓) = god in-sample tilpasning.  
Filtrert R² < 0.3 (⚠) eller negativ (✗) = modellen klarer ikke å forklare variansen i serien, selv med full tilgang til data t.o.m. t.  
Predikert R² er alltid lavere enn filtrert — det er ventet.

> **Strukturell merknad:** K&M (2019) estimerte på data t.o.m. ~2019. Vi inkluderer 15 kvartal post-COVID (2022–2025) med Norges Banks rentehevingssyklus (0→4,5 %). Svak fit på realvariabler kan reflektere reelle strukturelle brudd etter 2020, ikke kun modellfeil.

