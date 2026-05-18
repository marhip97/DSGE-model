# Fase 2 reparam — Diagnostikkrapport

**Dato:** 2026-05-16  
**Chain:** chain_fase2_reparam_prod (200k trekk, use_reparam=True)  

## Spørsmål 1: Er reparam-modusen genuint bedre enn K&M?

| Startpunkt | log-posterior |
|-----------|--------------|
| K&M-verdier | 3126.04 |
| Fase 2v2 posterior | 3564.26 (Δ=+438.2) |
| Reparam posterior mean | 3568.62 (Δ=+442.6) |

**Konklusjon:** Reparam-modusen er Δlp≈+443 bedre enn K&M. Dataene støtter **ikke** K&M-kalibreringen. Dette er et genuint empirisk funn, ikke en sampler-artefakt.

## Spørsmål 2: Datadrevne avvik fra K&M

| Parameter | K&M | Reparam-modus | Δlp (modus − K&M) | Tolkning |
|-----------|-----|--------------|-------------------|----------|
| rho_A | 0.804 | 0.0608 | +17.7 | Teknologisjokk nær IID i norske data; lav persistens |
| rho_O | 0.874 | 0.3906 | +10.7 | Oljeprissjokk raskere revertering enn K&M |
| phi_I1 | 4.000 | 0.5020 | +97.8 | Lave inv.justeringskostn. — data foretrekker nedre grense |
| phi_u | 0.219 | 0.6734 | +44.3 | Høy kapitalutnyttelsesfleksibilitet i norsk økonomi |
| sigma_H | 0.050 | 0.1539 | +93.5 | Større boligprisvolatilitet enn K&M-kalibrering |

## Spørsmål 3: Årsak til lav ESS

ESS/n er 0.001–0.003 for alle parametre — langt under 0.02-kravet.
Årsaken er **ikke** feil modus, men høy autokorrelasjon:

### Tregest parametre (høyest IAT)

| Parameter | IAT | ESS | ESS/n (%) |
|-----------|-----|-----|-----------|
| rho_Ys | 887 | 226 | 0.113% |
| rho_A | 792 | 252 | 0.126% |
| rho_C | 779 | 257 | 0.128% |
| psi_Y | 656 | 305 | 0.152% |
| psi_P1 | 523 | 382 | 0.191% |

### Årsak til høy autokorrelasjon

Posterioret har steile vegger langs rho_C, psi_P1 og psi_Y (alle ved eller nær hhv. beta-prior øvre grense / bred støtte). Komponentvis RWMH i 20 dimensjoner med sterke korrelasjoner gir korte effektive steg.

## Anbefalinger

1. **Forhåndsregistrer rho_A-prior-stramming** (Beta(2, 2) → posterior konsentreres, ESS bedres for rho_A-dimensjonen).
2. **Fiks phi_I1 til 4.0** eller stram prior kraftig (data er nær nedre grense; parameteren er svakt identifisert).
3. **HMC** (eskaleringsliste) vil dramatisk bedre ESS ved å bruke gradient-informasjon — anbefalt hvis RWMH ikke når ESS/n>0.02.
4. **Blokksampling** for (rho_C, psi_P1, psi_Y) kan redusere IAT uten HMC.

Funn rapportert til PE for videre beslutning.