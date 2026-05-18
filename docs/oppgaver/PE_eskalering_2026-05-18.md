# PE-eskalering 2026-05-18 — Modellfix og prior-innstramming før re-estimering

**Avsender:** Claude Code (ARK/DSGE/STAT)
**Mottaker:** Prosjekteier (PE)
**Status:** Krever godkjenning før re-estimering
**Branch:** `claude/review-and-plan-OKioj`

---

## Sammendrag

Spor A grundig diagnose har avdekket fire likningsfeil i `build_matrices_v3()` og
en sannsynlig prior-misspesifisering for `psi_R`. De fire feilene er allerede
fikset på branchen (commits inkl. testpakke som passerer 54/5xfailed). Re-estimering
kan ikke kjøres uten PE-godkjenning, og bør bundles med en prior-justering for å
unngå at posterior på nytt treffer grensen.

**Tre PE-beslutninger trengs:**
1. Godkjenne de fire likningsfiksingene (allerede committet, lett å revertere)
2. Godkjenne ny MCMC-kjøring (~2 timer) med fikset modell
3. Godkjenne strammere prior på `psi_R` før kjøring

---

## 1. Likningsfiksing (allerede implementert på branch)

| ID  | Likning | Feil før | Etter |
|-----|---------|----------|-------|
| A4a | Bankkapital (lign 26) | `G0[26,NB]=11.0`, ingen akkumulering | `G0[26,NB]=1.0`, `G1[26,NB]=(1−δ_b)`, Gerali et al. (2010) |
| A4c | LTV-sjokk (lign 25) | `Psi[25,E_phi_h]=+1.0` (strammere LTV → mer gjeld, feil fortegn) | `Psi[25,E_phi_h]=-1.0` |
| CEE | Investerings-Q (lign 12, v3) | `G0[12,Q_K]=-1/φ_I1` (2× for stor respons) | `G0[12,Q_K]=-1/(φ_I1·(1+β))`, Christiano-Eichenbaum-Evans (2005) |
| A5  | BNP-likning (lign 9) | `G0[9,INV]=-IY`, kun kapitalinv. → CY+IY+GY+XY−MY=0.90 (ubalansert) | `G0[9,INV]=-(IY+IHY)` → 1.00 ✓ |

**Verifikasjon:**
- BK-stabilitet `max|eig(T)| < 1` for v3 ✓
- 54 tester passerer, 5 xfailed (v1/v2 deprecated, marginalt ustabile etter A4a)
- IRF-fortegn-tester passerer

**Konsekvens for likelihood:** A4a endrer state-transition dynamisk → ny MCMC kreves.

---

## 2. Re-estimering (krever PE-godkjenning)

**Plan:**
- 5-blokk RWMH som forrige produksjonskjøring
- 160 000 trekk × 2 kjeder (samme oppsett som `chain_fase2_phi1fix_prod`)
- `phi_I1=4.0` fast (PE-godkjent), `sigma_A=0.006` fast
- Output: `data/results/chain_fase2_postfix_posterior.json` + `.npy`
- Estimert tid: ~2 timer

**Akseptansekriterier (lagres i `data/results/mcmc_log.md`):**
- PSRF_max < 1.10 alle 19 parametre
- ESS/n > 0.02 for minst 17 av 19
- B5-benchmark BNP-ratio mot NB Memo 3/2024: forbedring fra 0.13 mot ≥0.50

---

## 3. Prior-innstramming for `psi_R` (krever PE-godkjenning)

### Diagnose

| Kilde | psi_R / ω_r | Kommentar |
|-------|-------------|-----------|
| K&M (2019) | 0.6663 | Historisk estimat 2001–2016 |
| NB Staff Memo 2025-15 (GEORG) | 0.74 | Optimal tapsfunksjon-vekting |
| Vår produksjonsposterior | 0.964 | Treffer prior-grensen 0.990 |

Vår posterior er 44 % høyere enn K&M-estimatet. Hovedhypotese: pre-A4a manglet
modellen reell bank-dynamikk, så `psi_R` absorberte historisk rentepersistens som
egentlig skulle vært dynamikk i bankkapitalen. GEORG ω_r=0.74 korroborerer at
optimal smoothing er vesentlig lavere enn 0.964.

### Forslag til ny prior

```python
# Før (mcmc.py)
('beta', 2.0, 0.5, 0.01, 0.990)   # tykk hale mot 1.0, sentrum nær 0.8

# Etter
('beta', 2.0, 2.0, 0.01, 0.85)    # symmetrisk, topp ~0.5, cut-off 0.85
```

**Begrunnelse:**
- Øvre grense 0.85 > K&M+1σ og > GEORG (0.74) — data har rom for høyere
- Blokkerer patologiske verdier > 0.85 som tidligere har blitt forklart av bug
- Hvis posterior likevel kryper mot 0.85 etter fiksing: separat diagnose

### Alternativ (mer konservativ)

Hold dagens prior, men kjør først ny MCMC med fikset modell. Hvis posterior fortsatt
treffer 0.990, dokumenter dette og kom tilbake med prior-forslag. **Anbefales ikke**
fordi det dobler MCMC-budsjett (2+2 timer) når diagnosen allerede peker entydig.

---

## 4. Anbefalt rekkefølge

1. PE godkjenner punkt 1 (likningsfix — lavest risk, allerede implementert)
2. PE godkjenner punkt 3 (psi_R prior) eller velger alternativ
3. PE godkjenner punkt 2 (re-estimering) — kjøres etter steg 1+2 er klare
4. Resultat dokumenteres i `data/results/mcmc_log.md` og oppdatert B5-rapport

---

## 5. Åpne spørsmål til PE

- **`phi_I1`-status:** Data foretrekker `phi_I1 ≈ 0.5` ved fri estimering, men K&M-verdi
  4.0 er nåværende fix. Skal vi un-fix samtidig, eller holde det til separat runde?
- **`h_c`-prior:** Posterior treffer alltid 0.9995. Samme bug-kompensasjonsmønster?
  Foreslår å vente til vi ser post-fix-posterior før vi rører den.
- **Rapportering til styringsgruppe:** Skal funnene meldes som «modellfix før Fase 1»
  eller som del av Fase 0.5 sluttrapport?
