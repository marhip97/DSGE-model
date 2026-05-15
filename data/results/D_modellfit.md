# D — Modell-fit-analyse: NEMO v3 vs. faktiske data

**Estimeringsperiode:** 2001Q2–2019Q4 + 2022Q1–2025Q3  
**COVID utelatt:** 2020Q1–2021Q4 (8 kvartaler)

To mål:
- **Filtrert** (H@z_{t|t}): modellens forklaring etter å ha sett data t.o.m. t — mål på in-sample tilpasning
- **Predikert** (H@z_{t|t-1}): ett-stegs fremover, uten å se data i t — strengere mål på prediksjonskraft

## Fit-statistikk per serie

| Serie | Beskrivelse | Filt. R² | Filt. korr | Pred. R² | Pred. korr |
|-------|-------------|----------|------------|----------|------------|
| `dy_obs` | BNP-vekst (Δy) | 0.239 ⚠ | 0.599 | -1.228 ✗ | 0.158 |
| `dc_obs` | Konsum-vekst (Δc) | 0.506 | 0.728 | -0.118 ✗ | -0.222 |
| `dinv_obs` | Invest.-vekst (ΔInv) | 0.123 ⚠ | 0.557 | 0.007 ⚠ | 0.094 |
| `dx_obs` | Eksport-vekst (Δx) | 0.241 ⚠ | 0.493 | -0.165 ✗ | -0.085 |
| `dm_obs` | Import-vekst (Δm) | 0.463 | 0.690 | 0.072 ⚠ | 0.335 |
| `pi_obs` | KPI-inflasjon (π) | 0.662 | 0.877 | 0.005 ⚠ | 0.076 |
| `dw_obs` | Lønnsvekst (Δw) | 0.909 ✓ | 0.956 | 0.697 | 0.859 |
| `i_R_obs` | Styringsrente (i_R) | 0.998 ✓ | 0.999 | 0.883 ✓ | 0.940 |
| `i_3m_obs` | 3m pengemarked (i_3m) | 0.997 ✓ | 0.999 | 0.874 ✓ | 0.935 |
| `ds_obs` | Valutakurs (Δs) | 0.672 | 0.875 | -0.010 ✗ | -0.071 |
| `dpO_obs` | Oljepris (ΔpO) | 0.971 ✓ | 0.998 | 0.017 ⚠ | 0.187 |
| `dyS_obs` | Utenl. BNP (ΔyS) | 0.825 ✓ | 0.933 | 0.654 | 0.856 |
| `dh_obs` | Boligpris (Δh) | 0.466 | 0.717 | 0.311 | 0.616 |
| `db_obs` | Gjeld (Δb) | 0.688 | 0.833 | 0.431 | 0.665 |

**Gj.snitt filtrert R²:** 0.626  
**Gj.snitt predikert R²:** 0.174  
**Gj.snitt filtrert korrelasjon:** 0.804


## Tolkning

Filtrert R² > 0.7 (✓) = god in-sample tilpasning.  
Filtrert R² < 0.3 (⚠) eller negativ (✗) = modellen klarer ikke å forklare variansen i serien, selv med full tilgang til data t.o.m. t.  
Predikert R² er alltid lavere enn filtrert — det er ventet.

> **Strukturell merknad:** K&M (2019) estimerte på data t.o.m. ~2019. Vi inkluderer 15 kvartal post-COVID (2022–2025) med Norges Banks rentehevingssyklus (0→4,5 %). Svak fit på realvariabler kan reflektere reelle strukturelle brudd etter 2020, ikke kun modellfeil.

