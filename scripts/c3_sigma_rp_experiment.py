"""
[STAT] Spor C3 — sigma_rp-eksperiment (PE-godkjent).

Hypotese H3: sigma_rp er overestimert (0.016 vs. K&M 0.006) fordi modellen
mangler UIP-dynamikk. Testen: hold sigma_rp=0.006 fast og re-estimer de
16 gjenværende parametrene.

Kjøring: 5 000 burn-in + 30 000 produksjon (~10–15 min).

Spørsmål som besvares:
  - Klarer modellen å matche dataene uten sigma_rp som sikkerhetsventil?
  - Flyttes h_c og psi_R fra priorbegrensning?
  - Forbedres RER-IRF (B5-avvik ~2× for stort)?

Output:
  data/results/C3_chain.npy
  data/results/C3_chain_posterior.json
  data/results/C3_rapport.md  — sammenligning v3 vs. C3
  data/results/C3_irf_sammenligning.png (valgfri)
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
    log_prior, log_posterior, build_H, build_Sv,
    compute_ess, compute_psrf, adaptive_mcmc_with_monitoring,
    OBS_NAMES, SIGMA_A_FIXED,
)
from nemo.model.equations import (
    build_matrices_v3, E_i, I_R, PI, Y, RER, Q_H,
)
from nemo.model.parameters import Parameters
from nemo.solver.blanchard_kahn import compute_irf, solve as bk_solve

logger = logging.getLogger(__name__)

SIGMA_RP_FIXED = 0.006   # K&M-verdi, holder fast i dette eksperimentet
N_BURNIN = 5_000
N_PROD   = 30_000
T_IRF    = 20
SHOCK_SIZE = 0.0025


# ── Tilpasset log-posterior med fast sigma_rp ────────────────────────────────

def log_posterior_c3(theta: np.ndarray, H, Sv, Y_pre, Y_post) -> float:
    """Som log_posterior, men sigma_rp holdes fast til SIGMA_RP_FIXED."""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    try:
        class Pt(Parameters):
            pass
        for i, n in enumerate(PARAM_NAMES):
            setattr(Pt, n, float(theta[i]))
        setattr(Pt, "sigma_A",  SIGMA_A_FIXED)
        setattr(Pt, "sigma_rp", SIGMA_RP_FIXED)   # overstyr estimert verdi
        G0, G1, Psi, Pi = build_matrices_v3(Pt, theta_H=0.05)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            T_n, R_n, d = bk_solve(G0, G1, Psi, Pi, verbose=False)
        if not d["stable"]:
            return -np.inf
        from nemo.estimation.mcmc import kalman_hull, build_Q
        # Bygg Q med fast sigma_rp
        from nemo.model.equations import NE, E_rp
        Q = build_Q(theta)
        Q[E_rp, E_rp] = SIGMA_RP_FIXED**2   # overstyr
        ll = kalman_hull(T_n, R_n, H, Q, Sv, Y_pre, Y_post)
        return ll + lp if np.isfinite(ll) else -np.inf
    except Exception:
        return -np.inf


def lag_irf_c3(theta: np.ndarray) -> np.ndarray | None:
    """IRF for C3-posterior (sigma_rp fast)."""
    class Pt(Parameters):
        pass
    for i, n in enumerate(PARAM_NAMES):
        setattr(Pt, n, float(theta[i]))
    setattr(Pt, "sigma_A",  SIGMA_A_FIXED)
    setattr(Pt, "sigma_rp", SIGMA_RP_FIXED)
    try:
        G0, G1, Psi, Pi = build_matrices_v3(Pt, theta_H=0.05)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            T_n, R_n, d = bk_solve(G0, G1, Psi, Pi, verbose=False)
        if not d["stable"]:
            return None
        return compute_irf(T_n, R_n, E_i, SHOCK_SIZE, T_periods=T_IRF)
    except Exception:
        return None


def normaliser_til_1pp(irf: np.ndarray) -> np.ndarray:
    topp = float(np.max(irf[:, I_R]))
    return irf / topp if topp > 0 else irf


def les_startpunkt(posterior_sti: Path) -> tuple[np.ndarray, np.ndarray, float]:
    with posterior_sti.open() as fh:
        d = json.load(fh)
    summ = d["summary"]
    meta = d.get("meta", {})
    theta = np.array([summ[n]["mean"] for n in PARAM_NAMES])
    std   = np.array([max(summ[n]["std"], 1e-4) for n in PARAM_NAMES])
    scale = float(meta.get("final_scale", 0.5))
    return theta, std, scale


def lag_sammenligning_rapport(
    chain_v3: dict, chain_c3: dict, irf_v3: np.ndarray, irf_c3: np.ndarray
) -> str:
    linjer: list[str] = []
    linjer.append("# C3 — sigma_rp-eksperiment: v3 vs. C3 (fast sigma_rp=0.006)\n")
    linjer.append(
        f"C3-kjøring: {N_BURNIN:,} burn-in + {N_PROD:,} produksjon. "
        f"sigma_rp holdes fast til {SIGMA_RP_FIXED} (K&M-verdi).\n"
    )

    linjer.append("## 1. Parameterjustering: v3 vs. C3\n")
    linjer.append(
        "| Parameter | K&M | v3 mean | C3 mean | Endring | Merknad |"
    )
    linjer.append(
        "|-----------|-----|---------|---------|---------|---------|"
    )
    interessante = ["sigma_rp", "h_c", "psi_R", "rho_rp", "sigma_C", "sigma_O",
                    "psi_P1", "psi_Y", "rho_A", "rho_C"]
    for name in interessante:
        km = KM.get(name, float("nan"))
        v3m = chain_v3.get(name, {}).get("mean", float("nan"))
        c3m = chain_c3.get(name, {}).get("mean", float("nan"))
        delta = c3m - v3m
        lb, ub = PARAM_PRIORS[name][-2], PARAM_PRIORS[name][-1]
        merknad = ""
        if name == "sigma_rp":
            merknad = f"FAST = {SIGMA_RP_FIXED}"
        elif c3m > ub - 2 * chain_c3.get(name, {}).get("std", 0.01):
            merknad = "⚑ fortsatt ved grense"
        elif abs(delta) > 0.05:
            merknad = "↑ endret" if delta > 0 else "↓ endret"
        linjer.append(
            f"| {name:<10} | {km:>6.4f} | {v3m:>7.4f} | {c3m:>7.4f} "
            f"| {delta:>+7.4f} | {merknad} |"
        )

    linjer.append("\n## 2. IRF-sammenligning: pengepolitikkssjokk (+1 pp rente)\n")
    linjer.append("| Variabel | Horisont | v3 | C3 | Endring |")
    linjer.append("|----------|----------|----|----|---------|")
    var_map = {Y: "BNP-gap", PI: "KPI-infl.", I_R: "Rente", RER: "RER-gap"}
    horisonter = [(0, "q1"), (3, "q4"), (7, "q8"), (11, "q12")]
    for var_idx, navn in var_map.items():
        for h_idx, h_navn in horisonter:
            v3_val = float(irf_v3[h_idx, var_idx]) if irf_v3 is not None else float("nan")
            c3_val = float(irf_c3[h_idx, var_idx]) if irf_c3 is not None else float("nan")
            delta  = c3_val - v3_val
            linjer.append(
                f"| {navn:<9} | {h_navn} | {v3_val:>+.3f} | {c3_val:>+.3f} | {delta:>+.3f} |"
            )

    linjer.append("\n## 3. H3-vurdering\n")
    if chain_c3:
        hc_c3 = chain_c3.get("h_c", {}).get("mean", float("nan"))
        psir_c3 = chain_c3.get("psi_R", {}).get("mean", float("nan"))
        hc_ub = PARAM_PRIORS["h_c"][-1]
        psir_ub = PARAM_PRIORS["psi_R"][-1]
        hc_v3 = chain_v3.get("h_c", {}).get("mean", 0.992)
        psir_v3 = chain_v3.get("psi_R", {}).get("mean", 0.963)

        linjer.append(
            f"**h_c:** v3={hc_v3:.4f} → C3={hc_c3:.4f} "
            f"({'UNDER grense' if hc_c3 < hc_ub - 0.01 else 'FORTSATT ved grense'})"
        )
        linjer.append(
            f"**psi_R:** v3={psir_v3:.4f} → C3={psir_c3:.4f} "
            f"({'UNDER grense' if psir_c3 < psir_ub - 0.02 else 'FORTSATT ved grense'})"
        )
        linjer.append("")
        if hc_c3 < hc_ub - 0.01 or psir_c3 < psir_ub - 0.02:
            linjer.append(
                "**Konklusjon: H3 STØTTES** — Når sigma_rp ikke kan absorbere "
                "risikopremiedynamikk, flyttes h_c og/eller psi_R fra priorbegrensningen. "
                "Dette indikerer at den opprinnelige overestimeringen av sigma_rp tvang "
                "høy persistens-via-habit som kompensasjon."
            )
        else:
            linjer.append(
                "**Konklusjon: H3 SVAK** — h_c og psi_R forblir ved grensen selv med "
                "fast sigma_rp. H4 (likelihood-rygg) og/eller H1 (genuint høy persistens) "
                "er dominerende. Prior-utvidelse og logit-reparametrisering anbefales (Fase 2)."
            )

    return "\n".join(linjer) + "\n"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rot = Path(__file__).resolve().parents[1]
    posterior_sti = rot / "data" / "results" / "chain_v3_v2_posterior.json"
    data_sti      = rot / "data" / "processed" / "nemo_data_faktisk_v2.csv"
    ut_dir        = rot / "data" / "results"

    # ── Last data ─────────────────────────────────────────────────────────────
    logger.info("Laster observasjonsdata: %s", data_sti)
    obs_df = pd.read_csv(data_sti, index_col=0, parse_dates=True)
    pre  = obs_df[obs_df.index <= "2019-12-31"][OBS_NAMES].values
    post = obs_df[obs_df.index >= "2022-01-01"][OBS_NAMES].values
    logger.info("  Pre=%d kv  Post=%d kv", len(pre), len(post))

    H = build_H()
    Sv = build_Sv()

    # ── Startpunkt fra v3-posterior ───────────────────────────────────────────
    logger.info("Startpunkt: %s", posterior_sti)
    theta_start, post_std, scale_init = les_startpunkt(posterior_sti)
    # Nullstill sigma_rp til K&M-verdi (fast i dette eksperimentet)
    rp_idx = PARAM_NAMES.index("sigma_rp")
    theta_start[rp_idx] = SIGMA_RP_FIXED
    post_std[rp_idx] = 0.001   # veldig lite — sigma_rp er fast

    lp0 = log_posterior_c3(theta_start, H, Sv, pre, post)
    logger.info("  Startverdi log-posterior (C3): %.2f", lp0)
    if not np.isfinite(lp0):
        raise RuntimeError("Startverdi gir -inf — sjekk parametre.")

    # ── MCMC med fast sigma_rp ────────────────────────────────────────────────
    # Bruker adaptiv MCMC direkte med tilpasset log-posterior
    logger.info("\nKjører C3 MCMC (%d burnin + %d produksjon) ...", N_BURNIN, N_PROD)
    logger.info("sigma_rp holdes fast til %.4f", SIGMA_RP_FIXED)

    from nemo.estimation.mcmc import compute_psrf, compute_ess
    import time

    rng = np.random.default_rng(2025)
    scale = scale_init
    C_prop = np.diag((scale * 2.38 / np.sqrt(N_PARAMS) * post_std)**2 + 1e-12)
    theta  = theta_start.copy()
    lp_cur = log_posterior_c3(theta, H, Sv, pre, post)

    def _run(n_steps: int, phase: str, adapt: bool = False) -> tuple[np.ndarray, float]:
        nonlocal theta, lp_cur, scale, C_prop
        ch = np.zeros((n_steps, N_PARAMS))
        acc = 0; acc_win = 0
        t0 = time.time()
        adapt_every = 200
        for i in range(n_steps):
            tp = theta + rng.multivariate_normal(np.zeros(N_PARAMS), C_prop)
            lpp = log_posterior_c3(tp, H, Sv, pre, post)
            if np.log(rng.uniform()) < lpp - lp_cur:
                theta = tp; lp_cur = lpp; acc += 1; acc_win += 1
            ch[i] = theta
            if adapt and (i + 1) % adapt_every == 0:
                rate = acc_win / adapt_every; acc_win = 0
                if   rate < 0.10: scale *= 0.60
                elif rate < 0.20: scale *= 0.80
                elif rate > 0.40: scale *= 1.40
                elif rate > 0.28: scale *= 1.10
                scale = float(np.clip(scale, 0.005, 10.0))
                C_prop = np.diag((scale * 2.38 / np.sqrt(N_PARAMS) * post_std)**2 + 1e-12)
            if (i + 1) % 5000 == 0:
                rem = (time.time() - t0) / (i + 1) * (n_steps - i - 1)
                logger.info("  [%7d/%d] acc=%.3f  lp=%.1f  gjenstår≈%.1fmin  [%s]",
                            i + 1, n_steps, acc / (i + 1), lp_cur, rem / 60, phase)
        return ch, acc / n_steps

    ch_bi, acc_bi = _run(N_BURNIN, "BURN-IN", adapt=True)
    logger.info("  Burn-in ferdig. acc=%.3f  scale=%.4f", acc_bi, scale)

    ch_prod, acc_prod = _run(N_PROD, "PROD")
    logger.info("  Produksjon ferdig. acc=%.3f", acc_prod)

    # ── Diagnostikk ───────────────────────────────────────────────────────────
    psrf = compute_psrf(ch_prod)
    ess  = [compute_ess(ch_prod[:, j]) for j in range(N_PARAMS)]

    logger.info("\n  %-12s %8s %8s %8s %6s %5s", "Param", "K&M", "Mean", "Std", "ESS", "PSRF")
    summ_c3: dict[str, dict] = {}
    for j, name in enumerate(PARAM_NAMES):
        s = ch_prod[:, j]
        mean_j = float(s.mean())
        std_j  = float(s.std())
        ess_j  = float(ess[j])
        psrf_j = float(psrf[j])
        km_j   = KM.get(name, float("nan"))
        actual_mean = SIGMA_RP_FIXED if name == "sigma_rp" else mean_j
        logger.info("  %-12s %8.4f %8.4f %8.5f %6.0f %5.3f",
                    name, km_j, actual_mean, std_j, ess_j, psrf_j)
        summ_c3[name] = {
            "mean":  actual_mean,
            "std":   std_j,
            "p5":    float(np.percentile(s, 5)),
            "p95":   float(np.percentile(s, 95)),
            "ess":   ess_j,
            "psrf":  psrf_j,
            "fixed": (name == "sigma_rp"),
        }
    summ_c3["sigma_rp"]["mean"] = SIGMA_RP_FIXED
    summ_c3["sigma_rp"]["std"]  = 0.0

    # ── Lagre kjede og posterior ──────────────────────────────────────────────
    np.save(str(ut_dir / "C3_chain.npy"), ch_prod)
    meta_c3 = {
        "n_burnin":       N_BURNIN,
        "n_prod":         N_PROD,
        "acc_rate":       float(acc_prod),
        "sigma_rp_fixed": SIGMA_RP_FIXED,
        "sigma_A_fixed":  SIGMA_A_FIXED,
        "psrf_max":       float(psrf.max()),
        "ess_min":        float(min(ess)),
    }
    (ut_dir / "C3_chain_posterior.json").write_text(
        json.dumps({"summary": summ_c3, "meta": meta_c3}, indent=2)
    )

    # ── IRF-sammenligning ─────────────────────────────────────────────────────
    logger.info("\nBeregner IRF for C3-posterior mean ...")
    theta_c3 = np.array([summ_c3[n]["mean"] for n in PARAM_NAMES])
    irf_c3   = lag_irf_c3(theta_c3)
    irf_c3_norm = normaliser_til_1pp(irf_c3) if irf_c3 is not None else None

    # Les v3 IRF fra eksisterende benchmark
    b5_sti = ut_dir / "B5_nb_benchmark.json"
    irf_v3_norm = None
    if b5_sti.exists():
        with b5_sti.open() as fh:
            b5 = json.load(fh)
        irf_v3_norm = np.array(b5["punkt_estimat"])

    # Les v3 posterior summary for sammenligning
    with (ut_dir / "chain_v3_v2_posterior.json").open() as fh:
        v3_data = json.load(fh)
    summ_v3 = v3_data["summary"]

    # ── Rapport ───────────────────────────────────────────────────────────────
    rapport = lag_sammenligning_rapport(summ_v3, summ_c3, irf_v3_norm, irf_c3_norm)
    (ut_dir / "C3_rapport.md").write_text(rapport)
    logger.info("Rapport lagret: %s", ut_dir / "C3_rapport.md")

    # ── Valgfritt plot ────────────────────────────────────────────────────────
    if irf_c3_norm is not None and irf_v3_norm is not None:
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 2, figsize=(11, 7))
            paneler = [(Y, "BNP-gap (%)"), (PI, "KPI-inflasjon (%)"),
                       (I_R, "Styringsrente (pp)"), (RER, "RER-gap (%)")]
            kvartaler = np.arange(1, T_IRF + 1)
            for ax, (var_idx, tittel) in zip(axes.flat, paneler):
                ax.axhline(0, color="0.6", lw=0.5)
                ax.plot(kvartaler, irf_v3_norm[:, var_idx],
                        color="C3", lw=1.8, linestyle="--", label="v3 (sigma_rp=0.016)")
                ax.plot(kvartaler, irf_c3_norm[:, var_idx],
                        color="C0", lw=1.8, label=f"C3 (sigma_rp={SIGMA_RP_FIXED})")
                ax.set_title(tittel); ax.set_xlabel("Kvartal")
                ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
            fig.suptitle(
                "C3 — Pengepolitikkssjokk: v3 vs. C3 (fast sigma_rp=0.006)\n"
                "Normalisert: +1 pp toppunkt i styringsrenten",
                fontsize=10,
            )
            fig.tight_layout()
            fig.savefig(str(ut_dir / "C3_irf_sammenligning.png"), dpi=120)
            plt.close(fig)
            logger.info("IRF-figur lagret: %s", ut_dir / "C3_irf_sammenligning.png")
        except ImportError:
            logger.warning("matplotlib mangler — hopper over plot")

    logger.info("\n%s", rapport)


if __name__ == "__main__":
    main()
