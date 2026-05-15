"""
[STAT] Spor C1, C2, C7 — Prior vs. Posterior analyse.

C1: Prior vs. posterior for alle 17 estimerte parametre (plot + tabell).
C2: Analyse av h_c og psi_R ved priorbegrensning — H1–H4-vurdering.
C7: Identifikasjonsstyrke per parameter (posterior_std / prior_std).

Bruker C8-kjeden (20 000 trekk) som datagrunnlag.
Full kjedeanalyse med høy ESS bør gjøres etter blokksampling (Fase 2).

Output:
  data/results/C1_prior_posterior.png   — prior vs. posterior for alle param
  data/results/C2_h_c_psi_R.png        — fokus på h_c og psi_R (H1–H4)
  data/results/C2_C7_rapport.md        — markdown-rapport

Merknad (ESS-forbehold):
  C8-kjeden har ESS/n ≈ 0.006 → konklusjoner er INDIKATIVE.
  Sikre H1–H4-konklusjoner krever ESS/n > 0.02 (Fase 2 blokksampling).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from scipy.special import betaln
from scipy.stats import norm as sp_norm

from nemo.estimation.mcmc import (
    PARAM_NAMES, N_PARAMS, KM, PARAM_PRIORS,
    compute_ess,
)

logger = logging.getLogger(__name__)


# ── Prior-samplerfunksjoner (for plott) ───────────────────────────────────────

def prior_logpdf(name: str, x: np.ndarray) -> np.ndarray:
    spec = PARAM_PRIORS[name]
    lb, ub = spec[-2], spec[-1]
    pt = spec[0]
    out = np.full_like(x, -np.inf)
    mask = (x > lb) & (x < ub)
    if not mask.any():
        return out
    if pt == "beta":
        a, b = spec[1], spec[2]
        xn = (x[mask] - lb) / (ub - lb)
        ok = (xn > 0) & (xn < 1)
        tmp = np.full(mask.sum(), -np.inf)
        tmp[ok] = (a - 1) * np.log(xn[ok]) + (b - 1) * np.log(1 - xn[ok]) - betaln(a, b)
        out[mask] = tmp
    elif pt == "normal":
        mu, sig = spec[1], spec[2]
        out[mask] = sp_norm.logpdf(x[mask], mu, sig)
    elif pt == "inv_gamma":
        sh, sc = spec[1], spec[2]
        xm = x[mask]
        pos = xm > 0
        tmp = np.full(mask.sum(), -np.inf)
        tmp[pos] = sh * np.log(sc) - (sh + 1) * np.log(xm[pos]) - sc / xm[pos]
        from scipy.special import gammaln
        tmp[pos] -= gammaln(sh)
        out[mask] = tmp
    return out


def prior_std_approx(name: str) -> float:
    """Approksimert priori-standardavvik via Monte Carlo."""
    spec = PARAM_PRIORS[name]
    lb, ub = spec[-2], spec[-1]
    pt = spec[0]
    rng = np.random.default_rng(0)
    if pt == "beta":
        a, b = spec[1], spec[2]
        s = rng.beta(a, b, 50_000) * (ub - lb) + lb
    elif pt == "normal":
        mu, sig = spec[1], spec[2]
        s = rng.normal(mu, sig, 50_000)
        s = np.clip(s, lb, ub)
    elif pt == "inv_gamma":
        sh, sc = spec[1], spec[2]
        s = sc / rng.standard_gamma(sh, 50_000)
        s = np.clip(s, lb, ub)
    else:
        return float("nan")
    return float(np.std(s))


# ── C7: Identifikasjonsstyrke ─────────────────────────────────────────────────

def beregn_identifikasjon(chain: np.ndarray) -> list[dict]:
    """post_std / prior_std: nær 1 → svak id., nær 0 → sterk id."""
    resultat = []
    for j, name in enumerate(PARAM_NAMES):
        p_std = prior_std_approx(name)
        q_std = float(chain[:, j].std())
        ratio = q_std / p_std if p_std > 0 else float("nan")
        ess_j = float(compute_ess(chain[:, j]))
        resultat.append({
            "name":      name,
            "km":        KM.get(name, float("nan")),
            "post_mean": float(chain[:, j].mean()),
            "post_std":  q_std,
            "prior_std": p_std,
            "id_ratio":  ratio,
            "ess":       ess_j,
            "ess_n":     ess_j / len(chain),
        })
    return resultat


# ── C2: H1–H4 vurdering ──────────────────────────────────────────────────────

def vurder_h1_h4(chain: np.ndarray, id_data: list[dict]) -> dict[str, str]:
    """
    H1: Modellen trenger veldig høy persistens (biologisk rimelig)
    H2: Svak identifikasjon — prior dominerer
    H3: Manglende modellkanal absorber persistens
    H4: Likelihood-rygg langs h_c→1 (a3_W → 0)
    """
    h_c_idx = PARAM_NAMES.index("h_c")
    psi_R_idx = PARAM_NAMES.index("psi_R")
    hc_id  = id_data[h_c_idx]
    psir_id = id_data[psi_R_idx]

    vurd: dict[str, str] = {}

    # H1: posterior langt fra prior-modus, men ved grensen
    hc_lb, hc_ub = PARAM_PRIORS["h_c"][-2], PARAM_PRIORS["h_c"][-1]
    psi_lb, psi_ub = PARAM_PRIORS["psi_R"][-2], PARAM_PRIORS["psi_R"][-1]

    hc_pct_topp  = float(np.mean(chain[:, h_c_idx]  > hc_ub  - 0.005))
    psir_pct_topp = float(np.mean(chain[:, psi_R_idx] > psi_ub - 0.01))

    # H2: id_ratio nær 1.0
    hc_id_str   = f"{hc_id['id_ratio']:.3f}" if hc_id['id_ratio'] == hc_id['id_ratio'] else "N/A"
    psir_id_str = f"{psir_id['id_ratio']:.3f}" if psir_id['id_ratio'] == psir_id['id_ratio'] else "N/A"

    vurd["h_c"] = (
        f"mean={hc_id['post_mean']:.4f}, std={hc_id['post_std']:.5f}, "
        f"id_ratio={hc_id_str}, {hc_pct_topp*100:.1f}% ved øvre grense"
    )
    vurd["psi_R"] = (
        f"mean={psir_id['post_mean']:.4f}, std={psir_id['post_std']:.5f}, "
        f"id_ratio={psir_id_str}, {psir_pct_topp*100:.1f}% ved øvre grense"
    )

    # H2-styrke: id_ratio > 0.8 → prior dominerer
    def h2_styrke(r):
        if r > 0.85: return "STERK H2 (prior dominerer)"
        if r > 0.60: return "MODERAT H2"
        return "SVAK H2 (posterior informativ)"

    vurd["H2_h_c"]   = h2_styrke(hc_id["id_ratio"])
    vurd["H2_psi_R"] = h2_styrke(psir_id["id_ratio"])

    # H4: std ekstremt lav nær grensen → likelihood-rygg
    vurd["H4_h_c"] = (
        "STØTTER H4 (std<0.003, ved grense)" if hc_id["post_std"] < 0.003
        else "Svak H4-støtte"
    )
    vurd["H4_psi_R"] = (
        "STØTTER H4 (std<0.015, ved grense)" if psir_id["post_std"] < 0.015
        else "Svak H4-støtte"
    )

    # H3: sigma_rp er forhøyet (0.016 vs K&M 0.006) → manglende UIP-dynamikk
    rp_idx = PARAM_NAMES.index("sigma_rp")
    rp_mean = float(chain[:, rp_idx].mean())
    vurd["H3_sigma_rp"] = (
        f"sigma_rp={rp_mean:.4f} (K&M 0.006) → "
        + ("STØTTER H3 (manglende UIP-kanal)" if rp_mean > 0.010 else "Svak H3-støtte")
    )

    return vurd


# ── Rapport ───────────────────────────────────────────────────────────────────

def lag_rapport(
    chain: np.ndarray,
    id_data: list[dict],
    h_vurd: dict[str, str],
) -> str:
    n = len(chain)
    linjer: list[str] = []

    linjer.append("# C1/C2/C7 — Prior vs. Posterior og Identifikasjon\n")
    linjer.append(
        f"Basert på C8-kjeden: {n:,} produksjonstrekkninger "
        f"(5 000 burn-in, startpunkt = posterior mean v3).\n"
    )
    linjer.append(
        "> **ESS-forbehold:** ESS/n ≈ 0.006 for de fleste parametre. "
        "Konklusjoner er **indikative**. Sikre resultater krever "
        "blokksampling + full kjøring i Fase 2 (ingen PE-godkjenning nødvendig for blokk).\n"
    )

    # C1 — Tabelloversikt
    linjer.append("## C1 — Prior vs. Posterior: alle parametre\n")
    linjer.append(
        "| Parameter | K&M | Prior std | Post mean | Post std | Post [5%, 95%] | "
        "id_ratio | ESS/n |"
    )
    linjer.append(
        "|-----------|-----|-----------|-----------|----------|----------------|"
        "---------|-------|"
    )
    for d in id_data:
        j = PARAM_NAMES.index(d["name"])
        p5  = float(np.percentile(chain[:, j], 5))
        p95 = float(np.percentile(chain[:, j], 95))
        linjer.append(
            f"| {d['name']:<10} | {d['km']:>6.4f} | {d['prior_std']:>9.5f} "
            f"| {d['post_mean']:>9.4f} | {d['post_std']:>8.5f} "
            f"| [{p5:.3f}, {p95:.3f}] | {d['id_ratio']:>8.3f} | {d['ess_n']:>6.4f} |"
        )

    # C7 — Identifikasjon
    linjer.append("\n## C7 — Identifikasjonsstyrke (post_std / prior_std)\n")
    linjer.append(
        "Ratio nær 0 → posterior er mye smalere enn prior → sterk identifikasjon.  \n"
        "Ratio nær 1 → posterior ≈ prior → svak identifikasjon (prior dominerer).\n"
    )
    sortert = sorted(id_data, key=lambda d: d["id_ratio"] if d["id_ratio"]==d["id_ratio"] else 0)
    linjer.append("| Parameter | id_ratio | Vurdering |")
    linjer.append("|-----------|----------|-----------|")
    for d in sortert:
        r = d["id_ratio"]
        if r > 0.85:    vurd = "⚠ svak id. (prior dominerer)"
        elif r > 0.55:  vurd = "moderat id."
        elif r > 0.20:  vurd = "god id."
        else:           vurd = "sterk id."
        linjer.append(f"| {d['name']:<10} | {r:>8.3f} | {vurd} |")

    # C2 — H1–H4
    linjer.append("\n## C2 — h_c og psi_R ved priorbegrensning: H1–H4\n")
    linjer.append(f"**h_c:** {h_vurd['h_c']}  ")
    linjer.append(f"**psi_R:** {h_vurd['psi_R']}\n")

    linjer.append("### Hypotesevurdering\n")
    linjer.append(
        "**H1 — Modellen trenger høy persistens (biologisk):**  \n"
        "Både h_c og psi_R er langt over K&M-verdiene (0.938 og 0.666). "
        "Post-mean 0.992 og 0.963 kontra K&M antyder at data *krever* høyere persistens "
        "enn K&M-spesifikasjonen. Mulig forklaring: norsk konsum-smoothing er sterkere "
        "enn i originalparameterisering.\n"
    )
    linjer.append(
        f"**H2 — Svak identifikasjon (prior dominerer):**  \n"
        f"h_c: {h_vurd['H2_h_c']}  \n"
        f"psi_R: {h_vurd['H2_psi_R']}  \n"
        "Lav id_ratio (< 0.10) betyr at posterior er mye smalere enn prior — "
        "data er informative om *nivået*, men nivået er ved grensen. "
        "Ikke en entydig H2-situasjon (prior dominerer vanligvis ved *flat* posterior).\n"
    )
    linjer.append(
        f"**H3 — Manglende modellkanal:**  \n"
        f"{h_vurd['H3_sigma_rp']}  \n"
        "Forhøyet sigma_rp absorberer risikopremiedynamikk som ideelt burde gå via "
        "UIP-leddet. Dette kan 'tvinge' høy h_c for å matche konsum-banen. "
        "Spor C3 (sigma_rp-eksperiment, PE-godkjenning) vil teste dette direkte.\n"
    )
    linjer.append(
        f"**H4 — Likelihood-rygg langs h_c→1:**  \n"
        f"h_c: {h_vurd['H4_h_c']}  \n"
        f"psi_R: {h_vurd['H4_psi_R']}  \n"
        "Svært lav posterior-std ved øvre grense er et klassisk tegn på "
        "likelihood-rygg. Når h_c → 1, går a3_W = (1-h_c)/(σ(1+h_c)) → 0, "
        "noe som gjør lønnsblokken nær-singulær. Modellen 'liker' denne grensen. "
        "Reparametrisering (f.eks. log(1-h_c)) kan avsløre om dette er en ekte "
        "modus eller numerisk artefakt.\n"
    )

    linjer.append("### Foreløpig C2-konklusjon (indikativ)\n")
    linjer.append(
        "De fire hypotesene er **ikke gjensidig utelukkende**. Mest sannsynlig scenario:\n\n"
        "- **H3 (manglende UIP/finanskanal)** er sannsynligvis den primære driveren: "
        "sigma_rp=0.016 vs. K&M 0.006 er sterk evidens for at modellen mangler en kanal. "
        "Spor C3-eksperiment (PE-godkjenning) vil gi svar.\n"
        "- **H4 (likelihood-rygg)** støttes av ekstremt lav std ved grensen. "
        "Logit-reparametrisering i Fase 2 (ingen PE nødvendig) bør prioriteres.\n"
        "- **H1** kan ikke avvises — norsk data kan genuint kreve høyere habit.\n"
        "- **H2** er usannsynlig som *eneste* forklaring: posterior er for smal.\n\n"
        "**Blokkering:** Sikre konklusjoner om H1 vs. H3 vs. H4 krever:\n"
        "1. Blokksampling (sigma_C/h_c, r=−0.811) — Fase 2, ingen PE\n"
        "2. Logit-reparametrisering av h_c og psi_R — Fase 2, ingen PE\n"
        "3. C3 sigma_rp-eksperiment — krever PE-godkjenning\n"
    )

    return "\n".join(linjer) + "\n"


def plot_prior_posterior(chain: np.ndarray, ut_dir: Path) -> None:
    """C1: Prior vs. posterior for alle 17 parametre."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib mangler — hopper over plot. pip install -e '.[viz]'")
        return

    fig, axes = plt.subplots(4, 5, figsize=(18, 13))
    axes_flat = list(axes.flat)

    for j, name in enumerate(PARAM_NAMES):
        ax = axes_flat[j]
        s = chain[:, j]
        lb, ub = PARAM_PRIORS[name][-2], PARAM_PRIORS[name][-1]

        # Posterior KDE (empirisk histogram)
        ax.hist(s, bins=60, density=True, color="C0", alpha=0.55, label="Posterior")

        # Prior kurve
        x_range = np.linspace(max(lb, s.min() - 2*s.std()), min(ub, s.max() + 2*s.std()), 500)
        log_p = prior_logpdf(name, x_range)
        p = np.exp(log_p - log_p.max())   # normalisert for plott
        # Grov normalisering for sammenligning
        dx = x_range[1] - x_range[0]
        p = p / (p.sum() * dx + 1e-30)
        ax.plot(x_range, p, color="C3", linewidth=1.5, label="Prior")

        # K&M-verdi
        km_v = KM.get(name, None)
        if km_v is not None:
            ax.axvline(km_v, color="k", linewidth=1.0, linestyle="--", alpha=0.7, label="K&M")

        ax.set_title(name, fontsize=9)
        ax.set_xlabel("")
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.legend(fontsize=7)

    # Skjul tomme subplots
    for k in range(N_PARAMS, len(axes_flat)):
        axes_flat[k].set_visible(False)

    fig.suptitle("C1 — Prior vs. Posterior (C8-kjede, 20k trekk)", fontsize=12)
    fig.tight_layout()
    fig.savefig(ut_dir / "C1_prior_posterior.png", dpi=120)
    plt.close(fig)
    logger.info("C1-figur lagret: %s", ut_dir / "C1_prior_posterior.png")


def plot_h_c_psi_R(chain: np.ndarray, ut_dir: Path) -> None:
    """C2: Detaljert prior vs. posterior for h_c og psi_R."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, name in zip(axes, ["h_c", "psi_R"]):
        j = PARAM_NAMES.index(name)
        s = chain[:, j]
        lb, ub = PARAM_PRIORS[name][-2], PARAM_PRIORS[name][-1]
        km_v = KM.get(name)

        ax.hist(s, bins=80, density=True, color="C0", alpha=0.55, label="Posterior")

        x_range = np.linspace(lb, ub, 600)
        log_p = prior_logpdf(name, x_range)
        p = np.exp(log_p - log_p.max())
        dx = x_range[1] - x_range[0]
        p = p / (p.sum() * dx + 1e-30)
        ax.plot(x_range, p, color="C3", linewidth=2.0, label="Prior")

        if km_v is not None:
            ax.axvline(km_v, color="k", linewidth=1.5, linestyle="--", label=f"K&M={km_v:.3f}")
        ax.axvline(ub, color="gray", linewidth=1.2, linestyle=":", label=f"Øvre grense={ub}")

        ax.set_title(
            f"{name}: mean={s.mean():.4f}, std={s.std():.5f}\n"
            f"(K&M {km_v:.3f}, ESS/n={compute_ess(s)/len(s):.4f})",
            fontsize=9,
        )
        ax.legend(fontsize=8)

    fig.suptitle(
        "C2 — h_c og psi_R ved priorbegrensning\n"
        "H4 (likelihood-rygg): std ekstremt lav nær øvre grense",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(ut_dir / "C2_h_c_psi_R.png", dpi=120)
    plt.close(fig)
    logger.info("C2-figur lagret: %s", ut_dir / "C2_h_c_psi_R.png")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rot = Path(__file__).resolve().parents[1]
    chain_sti = rot / "data" / "results" / "C8_chain.npy"
    ut_dir    = rot / "data" / "results"

    logger.info("Laster C8-kjede: %s", chain_sti)
    chain = np.load(str(chain_sti))
    logger.info("  Kjedelengde: %d  Parametre: %d", *chain.shape)

    logger.info("Beregner identifikasjonsstyrke (C7)...")
    id_data = beregn_identifikasjon(chain)

    logger.info("Vurderer H1–H4 for h_c og psi_R (C2)...")
    h_vurd = vurder_h1_h4(chain, id_data)

    logger.info("Skriver rapport...")
    rapport = lag_rapport(chain, id_data, h_vurd)

    md_sti = ut_dir / "C2_C7_rapport.md"
    md_sti.write_text(rapport, encoding="utf-8")
    logger.info("Rapport lagret: %s", md_sti)

    # Lagre numerisk C7-data
    c7_json = {d["name"]: {k: v for k, v in d.items() if k != "name"} for d in id_data}
    (ut_dir / "C7_identifikasjon.json").write_text(
        json.dumps(c7_json, indent=2), encoding="utf-8"
    )

    logger.info("Lager figurer (C1 og C2)...")
    plot_prior_posterior(chain, ut_dir)
    plot_h_c_psi_R(chain, ut_dir)

    logger.info("\n%s", rapport)


if __name__ == "__main__":
    main()
