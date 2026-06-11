# [QA] Kvalitetssikring av NEMO-dashbordet (2026-06)

Gjennomgang av alle grafer og datastørrelser i GitHub Pages-dashbordet
(`docs/index.html`), generert fra `data/results/analyse_resultater.json` via
`src/nemo/dashboard/build.py` og malen `src/nemo/dashboard/templates/dashboard.html`.

## Metode

- Krysssjekket alle seks faner (Oversikt, IRF, FEVD, Historikk, Prognose, Diagnostikk)
  mot kildedata og dokumentasjon.
- Verifiserte modellbygget mot det arkiverte referanseresultatet
  `data/results/kj41_fevd.json` (produsert av `scripts/kj41_fevd_hd.py`).
- Bekreftet enheter mot rådata i `data/processed/nemo_data_kpi_jae.csv` og
  demean-verdier i `data/processed/nemo_demean_kpi_jae.json`.
- Kjørte hele testsuiten (128 tester, inkl. 15 IRF-fortegnskrav).

## Hovedfunn og status

| # | Alvor | Funn | Status |
|---|-------|------|--------|
| A | Kritisk | Dashbordet merket alt som **kj41** (psi_R=0,949), men `analyse_resultater.json` var generert fra **chain_v3_v2** (psi_R≈0,96) i PR #21. Pipelinen kunne ikke regenerere fra kj41: `load_posterior`/`PARAM_NAMES` krevde `sigma_rp`/`h_c` i posterioren, som kj41 fikserer. | **Rettet** |
| B | Kritisk | Nivåseriene var skalert feil: dataene er demeanede **andeler**, men pipelinen mistet ×100 i refaktoren (PR #63). Styringsrenten ville vist «0,0 %» i stedet for «4,0 %». | **Rettet** |
| C | Kritisk | Tomme haleskvartaler (2026K1–K2, NaN i gammel data) ga «–» på alle KPI-kort, null aktive sjokk og betinget bane = ubetinget. | **Rettet** |
| D | Høy | Diagnostikk-kortet viste `n_states = 48`, men modellen er `NZ = 50`; footeren hardkodet 50. | **Rettet** |
| E | Middels | `sigma_rp` vist som 0,014 (motstrid med CLAUDE.md 0,016). I kj41 er `sigma_rp` fast = 0,006 (K&M), og dominerer ikke lenger valutakursvariansen. | **Rettet** |
| F | Middels | Oversikt Figur 4 merket «Importveid kronekurs (I-44)», men serien er kvartalsvis endring i realvalutakursen (`ds_obs`). | **Rettet** |
| G | Middels | Oversikt Figur 1 merket «BNP-gap», men serien er observert BNP-**vekst**; forklaringsteksten påsto «modellfiltrerte» gap-verdier. | **Rettet** |
| H | Middels | `n_shocks = 13` vist, men kun 8 sjokk dekomponeres i IRF/FEVD/HD. | **Rettet** |
| I | Lav | KPI-inflasjon merket som generell KPI, men estimeringen og filteret bruker KPI-JAE (`pi_core_obs`). | **Rettet** |
| J | OK | FEVD summerer til 100 %; 15 IRF-fortegnskrav holder; `hist_decomp`-justering korrekt; `max|eig| = 0,9815 < 1`. | Ingen tiltak |
| K | Gjenstår (PE) | `src/nemo/analysis/analyse.py` (gammel monolitt, `__main__`-stien) har samme skalerings- og kj41-inkompatibilitet. Anbefaler å konsolidere på `nemo.analysis.run` eller oppdatere analyse.py tilsvarende. | Åpen |
| L | Forventet | Boligpris (`dh_obs`) mangler de ferskeste kvartalene (publiseringsetterslep) → enkelte haleskverdier er `null`. Håndteres nå robust i KPI/grafer. | Ingen tiltak |

## Rettelser i koden

**`src/nemo/analysis/irf.py`**
- `load_posterior` drives nå av posteriorens egne nøkler (håndterer vilkårlige estimerte
  parametersett). `sigma_A` og `sigma_rp` settes fast = 0,006.
- `build_estimated_model` bygger kj41-modellen med `build_matrices_v3_forward`,
  `phi_PQ = 150`, `lambda_pi4 = 0`. Reproduserer `kj41_fevd.json` eksakt (Y- og
  RER-dekomponering stemmer til desimalen).

**`src/nemo/analysis/run.py`**
- PI-observasjonen bruker `pi_core_obs` (konsistent med kj41-estimeringen).
- Trimmer haleskvartaler der hele observasjonsraden mangler (forecast-origo).
- `meta.n_states` rapporterer faktisk `T.shape[0]`.
- Nivåseriene skaleres til prosent (`_PCT = 100`; styringsrenten i tillegg ×4).

**`src/nemo/dashboard/templates/dashboard.html`**
- KPI-kort: bruker siste ikke-null verdi; etiketter «BNP-vekst», «Realvalutakurs».
- Figur 1/4 og forklaringstekst rettet (observerte serier, riktige enheter).
- FEVD-merknad og Diagnostikk-tabell rad 1 oppdatert (kj41: `sigma_rp = 0,006`,
  valutakursvariansen drives av oljepris).
- Diagnostikk-kort: «Sjokk dekomponert / totalt» = 8 / 13.
- Prognose: skjuler betinget bane og Figur 16 hvis ingen sjokk er aktive (forsvar i dybden).

## Verifisering

- `meta`: `eig_max = 0,981521`, `n_states = 50`, `n_shocks = 13`, `last_obs_date = 2025-12-31`.
- FEVD (kj41) reproduserer `kj41_fevd.json`: BNP q8 Konsum 72,6 %, Oljepris 19,2 %;
  RER q8 Oljepris 53,5 %, Bolig 25,5 %.
- IRF-fortegn: pengepolitisk innstramming gir BNP-gap < 0, inflasjon < 0, rente > 0.
- KPI-kort viser nå: styringsrente 4,0 %, KPI-JAE 1,3 %, BNP-vekst og realvalutakurs i prosent.
- Aktive sjokk er ikke-null; betinget prognose avviker fra ubetinget.
- `pytest`: 125 passert, 3 xfailed.

## Besluttet

1. **Oversikt viser observerte serier** (ikke modellens gap-estimater). Begrunnelse:
   det er konsistent med Historikk- og Prognose-fanene, som også bruker observerte
   nivåserier, mens modellens gap-/strukturhistorie dekkes i IRF-, FEVD- og
   Historikk-fanene. Observerte data + modellanslag er dessuten den gjenkjennelige
   standardpresentasjonen (jf. Norges Bank og SSB). Etikettene er rettet deretter.

## Anbefalte oppfølginger (PE)

2. Rydd opp i `analyse.py` vs `analysis/run.py` (dobbel pipeline, funn K).
3. Vurder å automatisere regenerering av `analyse_resultater.json` i deploy-workflowen,
   slik at dashbordet ikke kan drive fra hverandre fra modellen igjen.
