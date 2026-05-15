# D — Modell-fit-analyse: NEMO v3 vs. faktiske data

**Estimeringsperiode:** 2001Q2–2019Q4 + 2022Q1–2025Q3  
**COVID utelatt:** 2020Q1–2021Q4 (8 kvartaler)

To mål:
- **Filtrert** (H@z_{t|t}): modellens forklaring etter å ha sett data t.o.m. t — mål på in-sample tilpasning
- **Predikert** (H@z_{t|t-1}): ett-stegs fremover, uten å se data i t — strengere mål på prediksjonskraft

## Fit-statistikk per serie

| Serie | Beskrivelse | Filt. R² | Filt. korr | Pred. R² | Pred. korr |
|-------|-------------|----------|------------|----------|------------|
| `dy_obs` | BNP-vekst (Δy) | 0.294 ⚠ | 0.639 | -1.170 ✗ | 0.176 |
| `dc_obs` | Konsum-vekst (Δc) | 0.538 | 0.746 | -0.049 ✗ | -0.026 |
| `dinv_obs` | Invest.-vekst (ΔInv) | 0.348 | 0.667 | 0.007 ⚠ | 0.164 |
| `dx_obs` | Eksport-vekst (Δx) | 0.220 ⚠ | 0.471 | -0.164 ✗ | -0.087 |
| `dm_obs` | Import-vekst (Δm) | 0.503 | 0.717 | 0.077 ⚠ | 0.355 |
| `pi_obs` | KPI-inflasjon (π) | 0.910 ✓ | 0.983 | 0.005 ⚠ | 0.086 |
| `dw_obs` | Lønnsvekst (Δw) | 0.918 ✓ | 0.959 | 0.758 ✓ | 0.882 |
| `i_R_obs` | Styringsrente (i_R) | 0.998 ✓ | 0.999 | 0.883 ✓ | 0.940 |
| `i_3m_obs` | 3m pengemarked (i_3m) | 0.997 ✓ | 0.999 | 0.874 ✓ | 0.935 |
| `ds_obs` | Valutakurs (Δs) | 0.678 | 0.879 | -0.008 ✗ | -0.061 |
| `dpO_obs` | Oljepris (ΔpO) | 0.939 ✓ | 0.997 | 0.018 ⚠ | 0.184 |
| `dyS_obs` | Utenl. BNP (ΔyS) | 0.826 ✓ | 0.934 | 0.654 | 0.856 |
| `dh_obs` | Boligpris (Δh) | 0.902 ✓ | 0.983 | 0.632 | 0.904 |
| `db_obs` | Gjeld (Δb) | 0.982 ✓ | 0.994 | 0.742 ✓ | 0.876 |

**Gj.snitt filtrert R²:** 0.718  
**Gj.snitt predikert R²:** 0.233  
**Gj.snitt filtrert korrelasjon:** 0.855


## Tolkning

Filtrert R² > 0.7 (✓) = god in-sample tilpasning.  
Filtrert R² < 0.3 (⚠) eller negativ (✗) = modellen klarer ikke å forklare variansen i serien, selv med full tilgang til data t.o.m. t.  
Predikert R² er alltid lavere enn filtrert — det er ventet.

> **Strukturell merknad:** K&M (2019) estimerte på data t.o.m. ~2019. Vi inkluderer 15 kvartal post-COVID (2022–2025) med Norges Banks rentehevingssyklus (0→4,5 %). Svak fit på realvariabler kan reflektere reelle strukturelle brudd etter 2020, ikke kun modellfeil.

