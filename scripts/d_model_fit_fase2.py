"""
[NUM] Modell-fit-analyse — Kalman-filter tilpassede verdier vs. faktiske data.

Evaluerer hvor godt NEMO v3 (posterior mean) treffer de 14 observerte seriene
over estimeringsperioden (2001Q2–2019Q4 og 2022Q1–2025Q3, COVID utelatt).

Leveranser:
  data/results/D_modellfit_fase2.png     — faktisk vs. filtrert for alle 14 serier
  data/results/D_modellfit_fase2.md      — RMSE, R², korrelasjon per serie
  data/results/D_modellfit_fase2.json    — numeriske resultater
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve as sp_solve, cholesky, LinAlgError, solve_discrete_lyapunov

from nemo.estimation.mcmc import (
    PARAM_NAMES, OBS_NAMES, build_H, build_Sv, build_Q, SIGMA_A_FIXED,
)
from nemo.model.equations import build_matrices_v3, NZ
from nemo.model.parameters import Parameters
from nemo.solver.blanchard_kahn import solve as bk_solve

logger = logging.getLogger(__name__)

OBS_LABELS = {
    'dy_obs':   'BNP-vekst (Δy)',
    'dc_obs':   'Konsum-vekst (Δc)',
    'dinv_obs': 'Invest.-vekst (ΔInv)',
    'dx_obs':   'Eksport-vekst (Δx)',
    'dm_obs':   'Import-vekst (Δm)',
    'pi_obs':   'KPI-inflasjon (π)',
    'dw_obs':   'Lønnsvekst (Δw)',
    'i_R_obs':  'Styringsrente (i_R)',
    'i_3m_obs': '3m pengemarked (i_3m)',
    'ds_obs':   'Valutakurs (Δs)',
    'dpO_obs':  'Oljepris (ΔpO)',
    'dyS_obs':  'Utenl. BNP (ΔyS)',
    'dh_obs':   'Boligpris (Δh)',
    'db_obs':   'Gjeld (Δb)',
}


def kalman_filter_states(
    T: np.ndarray, R: np.ndarray, H: np.ndarray,
    Q: np.ndarray, Sv: np.ndarray,
    Y_obs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Kjører Kalman-filter og returnerer filtrerte tilstander og predikerte obs.

    Returnerer
    ----------
    z_filt : (T_obs × NZ) — filtrerte tilstander E[z_t | y_1..y_t]
    y_pred : (T_obs × N_OBS) — ett-stegs prediksjon H @ E[z_t | y_1..y_{t-1}]
    """
    T_obs = len(Y_obs)
    RQR = R @ Q @ R.T
    try:
        P = solve_discrete_lyapunov(T, RQR)
    except Exception:
        P = np.eye(NZ) * 0.01

    z = np.zeros(NZ)
    z_filt = np.zeros((T_obs, NZ))
    y_pred = np.full((T_obs, H.shape[0]), np.nan)

    for t in range(T_obs):
        zp = T @ z
        Pp = T @ P @ T.T + RQR
        yt = Y_obs[t]
        ms = np.isnan(yt)

        # Predikert observasjon (fra prior-tilstand)
        y_pred[t] = H @ zp

        if ms.all():
            z = zp; P = Pp
            z_filt[t] = z
            continue

        Ht  = H[~ms]
        yo  = yt[~ms]
        Sv_t = Sv[np.ix_(~ms, ~ms)]
        inn = yo - Ht @ zp
        S   = Ht @ Pp @ Ht.T + Sv_t
        S   = (S + S.T) / 2

        try:
            Kg = Pp @ Ht.T @ np.linalg.inv(S)
            z  = zp + Kg @ inn
            P  = (np.eye(NZ) - Kg @ Ht) @ Pp
            P  = (P + P.T) / 2
        except Exception:
            z = zp; P = Pp

        z_filt[t] = z

    y_filt = (H @ z_filt.T).T   # (T_obs × N_OBS) — filtrert (bruker nåværende obs)
    return z_filt, y_pred, y_filt


def beregn_fit_statistikk(
    actual: np.ndarray, fitted: np.ndarray
) -> dict[str, float]:
    """RMSE, MAE, R², korrelasjon — kun på ikke-NaN par."""
    mask = ~(np.isnan(actual) | np.isnan(fitted))
    a = actual[mask]
    f = fitted[mask]
    n = len(a)
    if n < 3:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "corr": np.nan, "n": n}
    res  = a - f
    rmse = float(np.sqrt(np.mean(res**2)))
    mae  = float(np.mean(np.abs(res)))
    ss_res = float(np.sum(res**2))
    ss_tot = float(np.sum((a - a.mean())**2))
    r2   = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    corr = float(np.corrcoef(a, f)[0, 1]) if np.std(f) > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2, "corr": corr, "n": n}


def lag_rapport(
    stat_pred: dict[str, dict],
    stat_filt: dict[str, dict],
    periode_info: str,
) -> str:
    linjer: list[str] = []
    linjer.append("# D — Modell-fit-analyse: NEMO v3 vs. faktiske data\n")
    linjer.append(f"**Estimeringsperiode:** {periode_info}  ")
    linjer.append("**COVID utelatt:** 2020Q1–2021Q4 (8 kvartaler)\n")
    linjer.append(
        "To mål:\n"
        "- **Filtrert** (H@z_{t|t}): modellens forklaring etter å ha sett data t.o.m. t — "
        "mål på in-sample tilpasning\n"
        "- **Predikert** (H@z_{t|t-1}): ett-stegs fremover, uten å se data i t — "
        "strengere mål på prediksjonskraft\n"
    )

    def fmt(v):
        return f"{v:.3f}" if (v == v and v is not None) else "N/A"

    def r2_flag(r2):
        if r2 != r2 or r2 is None: return ""
        if r2 > 0.7:   return " ✓"
        if r2 < 0.0:   return " ✗"
        if r2 < 0.3:   return " ⚠"
        return ""

    linjer.append("## Fit-statistikk per serie\n")
    linjer.append(
        "| Serie | Beskrivelse | Filt. R² | Filt. korr | Pred. R² | Pred. korr |"
    )
    linjer.append(
        "|-------|-------------|----------|------------|----------|------------|"
    )
    for navn in OBS_NAMES:
        sf = stat_filt.get(navn, {})
        sp = stat_pred.get(navn, {})
        label = OBS_LABELS.get(navn, navn)
        fr2 = sf.get("r2"); fc = sf.get("corr")
        pr2 = sp.get("r2"); pc = sp.get("corr")
        linjer.append(
            f"| `{navn}` | {label} "
            f"| {fmt(fr2)}{r2_flag(fr2)} | {fmt(fc)} "
            f"| {fmt(pr2)}{r2_flag(pr2)} | {fmt(pc)} |"
        )

    # Oppsummering
    fr2_vals = [v["r2"] for v in stat_filt.values() if v.get("r2") == v.get("r2")]
    pr2_vals = [v["r2"] for v in stat_pred.values() if v.get("r2") == v.get("r2")]
    fc_vals  = [v["corr"] for v in stat_filt.values()
                if v.get("corr") == v.get("corr") and v.get("corr") is not None]
    linjer.append(f"\n**Gj.snitt filtrert R²:** {np.mean(fr2_vals):.3f}  ")
    linjer.append(f"**Gj.snitt predikert R²:** {np.mean(pr2_vals):.3f}  ")
    linjer.append(f"**Gj.snitt filtrert korrelasjon:** {np.mean(fc_vals):.3f}\n")

    linjer.append("\n## Tolkning\n")
    linjer.append(
        "Filtrert R² > 0.7 (✓) = god in-sample tilpasning.  \n"
        "Filtrert R² < 0.3 (⚠) eller negativ (✗) = modellen klarer ikke å forklare "
        "variansen i serien, selv med full tilgang til data t.o.m. t.  \n"
        "Predikert R² er alltid lavere enn filtrert — det er ventet.\n"
    )
    linjer.append(
        "> **Strukturell merknad:** K&M (2019) estimerte på data t.o.m. ~2019. "
        "Vi inkluderer 15 kvartal post-COVID (2022–2025) med Norges Banks "
        "rentehevingssyklus (0→4,5 %). Svak fit på realvariabler kan reflektere "
        "reelle strukturelle brudd etter 2020, ikke kun modellfeil.\n"
    )

    return "\n".join(linjer) + "\n"


def plot_fit(
    dates_all: list,
    actual_all: np.ndarray,
    fitted_all: np.ndarray,
    ut_sti: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        logger.warning("matplotlib mangler — pip install -e '.[viz]'")
        return

    n_obs = len(OBS_NAMES)
    ncols = 3
    nrows = (n_obs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.2))
    axes_flat = list(axes.flat)

    for j, navn in enumerate(OBS_NAMES):
        ax = axes_flat[j]
        a = actual_all[:, j]
        f = fitted_all[:, j]
        mask_valid = ~(np.isnan(a) | np.isnan(f))

        ax.plot(dates_all, a, color="k", lw=1.2, label="Faktisk")
        ax.plot(dates_all, f, color="C3", lw=1.0, linestyle="--", label="Modell (KF)")
        ax.axhline(0, color="0.7", lw=0.4)

        # COVID-gap markering
        ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2022-01-01"),
                   alpha=0.12, color="gray", label="COVID (ekskl.)")

        if mask_valid.any():
            a_v = a[mask_valid]; f_v = f[mask_valid]
            res = a_v - f_v
            ss_res = np.sum(res**2); ss_tot = np.sum((a_v - a_v.mean())**2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            r2_str = f"R²={r2:.2f}" if r2 == r2 else ""
        else:
            r2_str = ""

        ax.set_title(f"{OBS_LABELS.get(navn, navn)}  {r2_str}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        if j == 0:
            ax.legend(fontsize=6, loc="best")

    for k in range(n_obs, len(axes_flat)):
        axes_flat[k].set_visible(False)

    fig.suptitle(
        "NEMO v3 — Modell-fit: Kalman-filter vs. faktiske data\n"
        "(posterior mean, COVID 2020Q1–2021Q4 utelatt)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(ut_sti, dpi=120)
    plt.close(fig)
    logger.info("Figur lagret: %s", ut_sti)


def bygg_parametere_fra_posterior(posterior_sti: Path) -> type:
    with posterior_sti.open() as fh:
        d = json.load(fh)
    summ = d["summary"]

    class Pt(Parameters):
        pass

    for name in PARAM_NAMES:
        setattr(Pt, name, float(summ[name]["mean"]))
    setattr(Pt, "sigma_A", SIGMA_A_FIXED)
    return Pt


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rot = Path(__file__).resolve().parents[1]
    posterior_sti = rot / "data" / "results" / "chain_fase2_prod_posterior.json"
    data_sti      = rot / "data" / "processed" / "nemo_data_faktisk_v2.csv"
    ut_dir        = rot / "data" / "results"

    # ── Last data ─────────────────────────────────────────────────────────────
    logger.info("Laster data: %s", data_sti)
    obs_df = pd.read_csv(data_sti, index_col=0, parse_dates=True)

    # Splitt: pre-COVID og post-COVID (COVID 2020–2021 utelatt)
    pre_mask  = obs_df.index <= "2019-12-31"
    post_mask = obs_df.index >= "2022-01-01"
    use_mask  = pre_mask | post_mask

    obs_used = obs_df[use_mask]
    dates_used = obs_used.index.tolist()
    Y_used = obs_used[OBS_NAMES].values   # (T_total × N_OBS)

    pre_idx  = obs_used.index <= "2019-12-31"
    post_idx = obs_used.index >= "2022-01-01"
    logger.info("  Pre: %d kv  Post: %d kv  Totalt (uten COVID): %d kv",
                pre_idx.sum(), post_idx.sum(), len(obs_used))

    # ── Bygg modell fra posterior mean ────────────────────────────────────────
    logger.info("Laster posterior mean: %s", posterior_sti)
    Pt = bygg_parametere_fra_posterior(posterior_sti)

    G0, G1, Psi, Pi = build_matrices_v3(Pt)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        T_mat, R_mat, diag = bk_solve(G0, G1, Psi, Pi, verbose=False)

    if not diag["stable"]:
        raise RuntimeError("BK ustabil ved posterior mean — kan ikke kjøre Kalman")
    logger.info("  BK stabil ✓  max|eig(T)|=%.4f", np.abs(np.linalg.eigvals(T_mat)).max())

    H  = build_H()
    Sv = build_Sv()
    Q  = build_Q(np.array([getattr(Pt, n) for n in PARAM_NAMES]))

    # ── Kjør Kalman-filter (pre og post, re-init mellom) ─────────────────────
    logger.info("Kalman-filter: pre-COVID (%d kv) ...", pre_idx.sum())
    Y_pre = Y_used[pre_idx]
    z_pre, y_pred_pre, y_filt_pre = kalman_filter_states(T_mat, R_mat, H, Q, Sv, Y_pre)

    logger.info("Kalman-filter: post-COVID (%d kv, re-init) ...", post_idx.sum())
    Y_post = Y_used[post_idx]
    z_post, y_pred_post, y_filt_post = kalman_filter_states(T_mat, R_mat, H, Q, Sv, Y_post)

    # ── Kombiner til full tidsserie (med NaN i COVID-gapet) ───────────────────
    full_dates  = obs_df.index.tolist()
    T_full      = len(full_dates)
    N_obs       = len(OBS_NAMES)
    actual_full = np.full((T_full, N_obs), np.nan)
    pred_full   = np.full((T_full, N_obs), np.nan)   # ett-stegs fremover
    filt_full   = np.full((T_full, N_obs), np.nan)   # filtrert

    pre_dates  = list(obs_used.index[pre_idx])
    post_dates = list(obs_used.index[post_idx])

    for i, dt in enumerate(full_dates):
        if dt in pre_dates:
            j = pre_dates.index(dt)
            actual_full[i] = Y_used[pre_idx][j]
            pred_full[i]   = y_pred_pre[j]
            filt_full[i]   = y_filt_pre[j]
        elif dt in post_dates:
            j = post_dates.index(dt)
            actual_full[i] = Y_used[post_idx][j]
            pred_full[i]   = y_pred_post[j]
            filt_full[i]   = y_filt_post[j]

    # ── Beregn fit-statistikk (begge mål) ─────────────────────────────────────
    logger.info("Beregner fit-statistikk ...")
    stat_pred: dict[str, dict] = {}
    stat_filt: dict[str, dict] = {}
    for j, navn in enumerate(OBS_NAMES):
        stat_pred[navn] = beregn_fit_statistikk(actual_full[:, j], pred_full[:, j])
        stat_filt[navn] = beregn_fit_statistikk(actual_full[:, j], filt_full[:, j])

    # ── Rapport og figur ──────────────────────────────────────────────────────
    periode_info = "2001Q2–2019Q4 + 2022Q1–2025Q3"
    rapport = lag_rapport(stat_pred, stat_filt, periode_info)

    md_sti = ut_dir / "D_modellfit_fase2.md"
    md_sti.write_text(rapport, encoding="utf-8")
    logger.info("Rapport: %s", md_sti)

    json_sti = ut_dir / "D_modellfit_fase2.json"
    json_sti.write_text(json.dumps({"predikert": stat_pred, "filtrert": stat_filt}, indent=2))
    logger.info("Data: %s", json_sti)

    plot_fit(full_dates, actual_full, filt_full, ut_dir / "D_modellfit_fase2.png")

    logger.info("\n%s", rapport)


if __name__ == "__main__":
    main()
