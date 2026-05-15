"""
[STAT] Spor C8 — MCMC Mixing og ESS-diagnose.

Kjører en kort MCMC-kjøring (~20 000 produksjonstrekkninger) for å
diagnostisere blandingsproblemer i v3-kjeden:

  - ESS_min = 662 av 200 000 trekk → ESS/n = 0.0033 (6× under kravet 0.02)
  - h_c = 0.989 og psi_R = 0.960 traff priorbegrensning

Leveranser:
  1. ACF per parameter (lag 1..100) → identifiserer tregeste param
  2. IAT og ESS per parameter
  3. Posteriorkorrelasjonsmatrise → avdekker blokkeringsstrukturer
  4. Anbefaling for Fase 2-sampler

Output:
  data/results/C8_mcmc_diagnostics.json  — numerisk diagnose
  data/results/C8_acf_rapport.md         — markdown-rapport

NB: Bruker posterior mean fra chain_v3_v2_posterior.json som startpunkt.
    Kort kjøring er nok for ACF-diagnose; full kjøring krever PE-godkjenning.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from nemo.estimation.mcmc import (
    PARAM_NAMES, N_PARAMS, KM, PARAM_PRIORS,
    log_posterior, build_H, build_Sv,
    compute_ess, compute_psrf,
    adaptive_mcmc_with_monitoring,
    OBS_NAMES,
)

logger = logging.getLogger(__name__)

N_BURNIN    = 5_000
N_PROD      = 20_000
ACF_MAX_LAG = 100


def les_startpunkt(posterior_sti: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Leser posterior mean/std og final_scale fra JSON-fil."""
    with posterior_sti.open() as fh:
        d = json.load(fh)
    summ = d["summary"]
    meta = d.get("meta", {})
    theta = np.array([summ[n]["mean"] for n in PARAM_NAMES])
    std   = np.array([max(summ[n]["std"], 1e-4) for n in PARAM_NAMES])
    scale = float(meta.get("final_scale", 0.676))
    return theta, std, scale


def beregn_acf(chain: np.ndarray, max_lag: int = ACF_MAX_LAG) -> np.ndarray:
    """Returnerer (max_lag × N_PARAMS) ACF-matrise."""
    n = len(chain)
    acf = np.zeros((max_lag, N_PARAMS))
    for j in range(N_PARAMS):
        x = chain[:, j] - chain[:, j].mean()
        var = np.var(x)
        if var < 1e-14:
            continue
        for k in range(1, max_lag + 1):
            acf[k - 1, j] = np.sum(x[k:] * x[:-k]) / (n * var)
    return acf


def beregn_iat(acf: np.ndarray) -> np.ndarray:
    """Integrert autokorrelasjonistid via summasjon frem til første negative ACF."""
    iat = np.ones(N_PARAMS)
    for j in range(N_PARAMS):
        for k in range(ACF_MAX_LAG):
            if acf[k, j] < 0.05:
                break
            iat[j] += 2 * acf[k, j]
    return iat


def korrelasjonsmatrise(chain: np.ndarray) -> np.ndarray:
    """Posteriorkorrelasjonsmatrise (N_PARAMS × N_PARAMS)."""
    return np.corrcoef(chain.T)


def lag_acf_rapport(
    chain: np.ndarray,
    acf: np.ndarray,
    iat: np.ndarray,
    korr: np.ndarray,
) -> str:
    n = len(chain)
    ess = n / np.maximum(iat, 1.0)
    psrf = compute_psrf(chain)

    linjer: list[str] = []
    linjer.append("# C8 — MCMC Mixing og ESS-diagnose\n")
    linjer.append(
        f"Kjøring: {N_BURNIN:,} burn-in + {N_PROD:,} produksjon "
        f"(Fase 0.5, Spor C8)\n"
    )

    linjer.append("## 1. ESS og IAT per parameter\n")
    linjer.append(
        "| Parameter | K&M-verdi | Posterior mean | IAT | ESS | ESS/n | PSRF | Vurdering |"
    )
    linjer.append(
        "|-----------|-----------|---------------|-----|-----|-------|------|-----------|"
    )
    for j, name in enumerate(PARAM_NAMES):
        mean_j = chain[:, j].mean()
        km_j = KM.get(name, float("nan"))
        ess_j = ess[j]
        iat_j = iat[j]
        essp = ess_j / n
        psrf_j = psrf[j] if psrf[j] == psrf[j] else 99.0
        vurd = []
        if essp < 0.02:
            vurd.append("⚠ lav ESS")
        if psrf_j > 1.10:
            vurd.append("⚠ PSRF")
        lb, ub = PARAM_PRIORS[name][-2], PARAM_PRIORS[name][-1]
        if mean_j > ub - 2 * chain[:, j].std():
            vurd.append("⚑ grense")
        flag = ", ".join(vurd) if vurd else "OK"
        linjer.append(
            f"| {name:<10} | {km_j:>9.4f} | {mean_j:>13.4f} "
            f"| {iat_j:>5.1f} | {ess_j:>5.0f} | {essp:>5.4f} "
            f"| {psrf_j:>5.3f} | {flag} |"
        )

    linjer.append("\n## 2. ACF ved lag 1, 5, 10, 20, 50\n")
    linjer.append("| Parameter | ACF(1) | ACF(5) | ACF(10) | ACF(20) | ACF(50) |")
    linjer.append("|-----------|--------|--------|---------|---------|---------|")
    lags = [0, 4, 9, 19, 49]
    for j, name in enumerate(PARAM_NAMES):
        vals = [f"{acf[l, j]:>+.3f}" for l in lags]
        linjer.append(f"| {name:<10} | {' | '.join(vals)} |")

    linjer.append("\n## 3. Sterkeste korrelasjoner (|r| > 0.3)\n")
    linjer.append("| Par | Korrelasjon |")
    linjer.append("|-----|-------------|")
    par_korr: list[tuple[float, str]] = []
    for i in range(N_PARAMS):
        for j in range(i + 1, N_PARAMS):
            r = korr[i, j]
            if abs(r) > 0.30:
                par_korr.append((abs(r), f"{PARAM_NAMES[i]}–{PARAM_NAMES[j]}"))
    par_korr.sort(reverse=True)
    for r_abs, navn in par_korr[:15]:
        # Hent faktisk fortegn
        i = PARAM_NAMES.index(navn.split("–")[0])
        j = PARAM_NAMES.index(navn.split("–")[1])
        linjer.append(f"| {navn} | {korr[i, j]:>+.3f} |")

    linjer.append("\n## 4. Diagnose og anbefalinger\n")

    tregeste = sorted(range(N_PARAMS), key=lambda j: iat[j], reverse=True)[:5]
    linjer.append("**Tregeste parametre (høyest IAT):**")
    for j in tregeste:
        linjer.append(
            f"- `{PARAM_NAMES[j]}`: IAT={iat[j]:.1f}, ESS/n={ess[j]/n:.4f}"
        )

    linjer.append("")
    linjer.append("**Priorbegrensningstreff:**")
    for j, name in enumerate(PARAM_NAMES):
        lb, ub = PARAM_PRIORS[name][-2], PARAM_PRIORS[name][-1]
        mean_j = chain[:, j].mean()
        std_j = chain[:, j].std()
        if mean_j + 2 * std_j > ub or mean_j - 2 * std_j < lb:
            pct_ub = 100 * np.mean(chain[:, j] > ub - 0.001)
            pct_lb = 100 * np.mean(chain[:, j] < lb + 0.001)
            linjer.append(
                f"- `{name}`: mean={mean_j:.4f}, std={std_j:.5f} "
                f"[{lb:.4f}, {ub:.4f}] — {pct_ub:.1f}% mot øvre, {pct_lb:.1f}% mot nedre"
            )

    # Finn sterkeste korrelasjonsblokker
    blokk_kandidater = [(abs(korr[i, j]), i, j) for i in range(N_PARAMS)
                        for j in range(i+1, N_PARAMS) if abs(korr[i, j]) > 0.5]
    blokk_kandidater.sort(reverse=True)

    linjer.append("")
    linjer.append("**Anbefalinger for Fase 2-sampler:**\n")
    linjer.append("Basert på diagnosen over:\n")

    ess_min = float(ess.min())
    ess_min_pct = ess_min / n

    if ess_min_pct < 0.01:
        linjer.append(
            "1. **Blokksampling (prioritet HØY):** Identifiserte sterke korrelasjonsblokker "
            "mellom parametre bremser komponentvis RWMH kraftig. "
            "Implementer blokkvis proposal for korrelerte parametre."
        )
    if any(iat[j] > 50 for j in range(N_PARAMS)):
        linjer.append(
            "2. **Reparametrisering:** Parametre med IAT > 50 indikerer sterk autokorrelasjon. "
            "Vurder logit-transformasjon for beta-parametre nær priorbegrensning."
        )
    if any(
        chain[:, j].mean() > PARAM_PRIORS[PARAM_NAMES[j]][-1] - 2 * chain[:, j].std()
        for j in range(N_PARAMS)
    ):
        linjer.append(
            "3. **Prierjustering (krever PE-godkjenning):** Parametre nær øvre grense "
            "antyder at prieren er for trang eller at modellen mangler en kanal. "
            "Se Spor C2 for H1–H4-analyse."
        )
    if len(blokk_kandidater) >= 3:
        linjer.append(
            "4. **HMC-vurdering (eskalering):** Mange sterke korrelasjoner + lav ESS "
            "taler for Hamiltonian Monte Carlo. Krever PE-godkjenning og JAX/NumPyro."
        )

    linjer.append("")
    linjer.append(
        "> **C8-konklusjon for C2:** ESS-estimatene fra denne korte kjøringen "
        "er indikative. Konklusjoner om h_c og psi_R ved priorbegrensning "
        "kan ikke trekkes med sikkerhet før ESS/n > 0.02 i en full kjøring. "
        "C2 H1–H4-analyse bør starte etter blokksampling er implementert (Fase 2)."
    )

    return "\n".join(linjer) + "\n"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rot = Path(__file__).resolve().parents[1]
    posterior_sti = rot / "data" / "results" / "chain_v3_v2_posterior.json"
    data_sti = rot / "data" / "processed" / "nemo_data_faktisk_v2.csv"
    ut_dir = rot / "data" / "results"

    # ── Last observasjonsdata ─────────────────────────────────────────────────
    logger.info("Laster observasjonsdata: %s", data_sti)
    obs_df = pd.read_csv(data_sti, index_col=0, parse_dates=True)
    pre  = obs_df[obs_df.index <= "2019-12-31"][OBS_NAMES].values
    post = obs_df[obs_df.index >= "2022-01-01"][OBS_NAMES].values
    logger.info("  Pre=%d kv  Post=%d kv", len(pre), len(post))

    H = build_H()
    Sv = build_Sv()

    # ── Startpunkt fra eksisterende posterior ─────────────────────────────────
    logger.info("Laster startpunkt fra: %s", posterior_sti)
    theta_start, post_std, scale_init = les_startpunkt(posterior_sti)
    logger.info("  h_c start=%.4f  psi_R start=%.4f  scale=%.4f",
                theta_start[PARAM_NAMES.index("h_c")],
                theta_start[PARAM_NAMES.index("psi_R")],
                scale_init)

    # Verifiser startpunkt
    lp0 = log_posterior(theta_start, H, Sv, pre, post)
    logger.info("  Startverdi log-posterior: %.2f", lp0)
    if not np.isfinite(lp0):
        raise RuntimeError("Startverdi gir -inf log-posterior. Sjekk data og parametre.")

    # ── Kort MCMC-kjøring ─────────────────────────────────────────────────────
    logger.info("\nStarter kort MCMC-kjøring for C8-diagnose")
    logger.info("  Burn-in: %d  Produksjon: %d", N_BURNIN, N_PROD)

    chain_sti = str(ut_dir / "C8_chain")
    chain, _, meta = adaptive_mcmc_with_monitoring(
        Y_pre=pre, Y_post=post, H=H, Sv=Sv,
        theta_init=theta_start, post_std_init=post_std,
        n_production=N_PROD,
        burnin=N_BURNIN,
        adapt_every=200,
        check_every=N_PROD,   # bare én konvergenssjekk (siste)
        max_recalib=2,
        psrf_thr=1.10,
        ess_pct_thr=0.02,
        scale_init=scale_init,
        seed=2024,
        verbose=True,
        save_prefix=chain_sti,
    )

    # ── Diagnose ──────────────────────────────────────────────────────────────
    logger.info("\nBeregner ACF, IAT, ESS, korrelasjonsmatrise ...")
    acf  = beregn_acf(chain, max_lag=ACF_MAX_LAG)
    iat  = beregn_iat(acf)
    korr = korrelasjonsmatrise(chain)
    ess  = np.array([compute_ess(chain[:, j]) for j in range(N_PARAMS)])
    psrf = compute_psrf(chain)

    # ── Skriv rapport ─────────────────────────────────────────────────────────
    rapport = lag_acf_rapport(chain, acf, iat, korr)
    md_sti  = ut_dir / "C8_acf_rapport.md"
    md_sti.write_text(rapport, encoding="utf-8")
    logger.info("Rapport lagret: %s", md_sti)

    # ── Lagre numerisk diagnose ───────────────────────────────────────────────
    resultat = {
        "n_burnin":  N_BURNIN,
        "n_prod":    N_PROD,
        "acf_max_lag": ACF_MAX_LAG,
        "params":    PARAM_NAMES,
        "ess":       ess.tolist(),
        "iat":       iat.tolist(),
        "psrf":      psrf.tolist(),
        "mean":      chain.mean(axis=0).tolist(),
        "std":       chain.std(axis=0).tolist(),
        "corr_matrix": korr.tolist(),
        "acf":       acf.tolist(),
        "meta":      meta,
    }
    json_sti = ut_dir / "C8_mcmc_diagnostics.json"
    json_sti.write_text(json.dumps(resultat, indent=2))
    logger.info("Diagnosedata lagret: %s", json_sti)

    # ── Sammendrag til konsoll ────────────────────────────────────────────────
    logger.info("\n%s", rapport)


if __name__ == "__main__":
    main()
