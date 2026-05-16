"""
[STAT] SYN-test: Valider logit-reparametrisering ved å gjenvinne K&M-parametre
fra syntetiske data (C5 §5).

Genererer T_pre=80 og T_post=40 observasjoner fra modellen ved K&M-parametre,
kjører deretter adaptive_mcmc_with_monitoring med use_reparam=True på disse
dataene og kontrollerer at:
  1. Kjøringen fullfører uten feil
  2. h_c og psi_R er innenfor (lb, ub) i alle trekk
  3. ≥14 av 20 parametre har K&M-verdi innenfor ±2·posterior_std

Kort kjøring: 2 000 burn-in + 3 000 produksjon (≈1 min på laptop).
Forutsetter at K&M-startverdiene gir et stabilt system.

Bruk:
    python scripts/syn_test_reparam.py

Bindende suksesskriterier (C5 §5) — tester *reparametrisering*, ikke full konvergens:
  - h_c og psi_R innenfor (lb, ub) i ALLE trekk  [FAIL]
  - ≥14/20 parametre: |posterior_mean − θ_true| < 2·posterior_std  [FAIL]
  - acc ∈ [0.10, 0.60]  [FAIL]

Rådgivende (advarsler, stopper ikke SYN-testen):
  - PSRF < 2.50 og ESS > 30 — forventes lavere ved 120-obs datasett;
    full produksjonskjøring med 200k trekk og ekte data vil gi bedre konvergens.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ROT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROT / "src"))

from nemo.estimation.mcmc import (
    KM, N_PARAMS, PARAM_NAMES, PARAM_PRIORS,
    build_H, build_Sv, build_Q,
    adaptive_mcmc_with_monitoring,
)
from nemo.estimation.reparam import REPARAM_PARAMS


def _build_state_space_km():
    """Bygg T og R ved K&M-parametre."""
    from nemo.model.equations import build_matrices_v3
    from nemo.model.parameters import Parameters
    from nemo.solver.blanchard_kahn import solve as bk_solve
    from nemo.estimation.mcmc import SIGMA_A_FIXED

    class KM_Params(Parameters):
        pass

    for name in PARAM_NAMES:
        if name in KM:
            setattr(KM_Params, name, KM[name])
    setattr(KM_Params, 'sigma_A', SIGMA_A_FIXED)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        G0, G1, Psi, Pi = build_matrices_v3(KM_Params, theta_H=0.05)
        T, R, d = bk_solve(G0, G1, Psi, Pi, verbose=False)

    if not d['stable']:
        raise RuntimeError("K&M-parametre gir ustabilt system — kan ikke generere syntetiske data")
    return T, R


def simulate_observations(T: np.ndarray, R: np.ndarray, H: np.ndarray,
                           Q: np.ndarray, Sv: np.ndarray,
                           n_periods: int, rng: np.random.Generator) -> np.ndarray:
    """Simuler observasjoner fra linearisert tilstandsromrepresentasjon.

    Args:
        T:        Overgangsmatrise (NZ × NZ)
        R:        Sjokk-loadingsmatrise (NZ × NE)
        H:        Observasjonsmatrise (N_OBS × NZ)
        Q:        Sjokk-kovariansmatrise (NE × NE)
        Sv:       Målefeil-kovariansmatrise (N_OBS × N_OBS)
        n_periods: Antall perioder
        rng:      NumPy-generator

    Returns:
        Y: (n_periods × N_OBS) observasjonsmatrise
    """
    NZ = T.shape[0]; NE = R.shape[1]; N_OBS = H.shape[0]
    chol_Q = np.linalg.cholesky(Q + 1e-14 * np.eye(NE))
    chol_Sv = np.linalg.cholesky(Sv + 1e-14 * np.eye(N_OBS))

    z = np.zeros(NZ)
    Y = np.empty((n_periods, N_OBS))
    for t in range(n_periods):
        eps = chol_Q @ rng.standard_normal(NE)
        z = T @ z + R @ eps
        eta = chol_Sv @ rng.standard_normal(N_OBS)
        Y[t] = H @ z + eta
    return Y


def main() -> None:
    rng = np.random.default_rng(42)
    print("=" * 65)
    print("  SYN-TEST: Logit-reparametrisering (C5 §5)")
    print("=" * 65)

    # ── Bygg modell ved K&M-parametre ────────────────────────────────────────
    print("\n[1/4] Bygg state space ved K&M-parametre …")
    T_mat, R_mat = _build_state_space_km()
    H = build_H()
    theta_true = np.array([KM.get(n, 0.5) for n in PARAM_NAMES], dtype=float)

    # Bruk K&M theta_true for Q
    Q = build_Q(theta_true)
    Sv = build_Sv()
    print(f"  NZ={T_mat.shape[0]}, NE={R_mat.shape[1]}, N_OBS={H.shape[0]}")

    # ── Generer syntetiske observasjoner ─────────────────────────────────────
    T_PRE, T_POST = 80, 40
    print(f"\n[2/4] Genererer syntetiske data: T_pre={T_PRE}, T_post={T_POST} …")
    Y_pre  = simulate_observations(T_mat, R_mat, H, Q, Sv, T_PRE,  rng)
    Y_post = simulate_observations(T_mat, R_mat, H, Q, Sv, T_POST, rng)
    print(f"  Y_pre shape:  {Y_pre.shape}")
    print(f"  Y_post shape: {Y_post.shape}")

    # ── Kjør MCMC med use_reparam=True ────────────────────────────────────────
    BURNIN, N_PROD = 4000, 6000
    print(f"\n[3/4] Kjører MCMC (burnin={BURNIN}, produksjon={N_PROD}) …")

    save_prefix = str(ROT / "data" / "results" / "syn_test_chain")

    # post_std i naturlig rom (brukes til proposal-diagonal før empirisk cov slår inn).
    # For reparametriserte parametre: vi overskriver til 2.0 (bred logit-skala) i mcmc.py,
    # men setter her et rimelig startpunkt for de andre.
    post_std_init = np.array([0.05] * N_PARAMS, dtype=float)
    # Bredere forslag for inv_gamma-parametre (sigma-er) som opererer i liten skala
    for i, name in enumerate(PARAM_NAMES):
        if PARAM_PRIORS[name][0] == 'inv_gamma':
            post_std_init[i] = KM.get(name, 0.01) * 0.30

    prod_ch, lp_prod, meta = adaptive_mcmc_with_monitoring(
        Y_pre=Y_pre,
        Y_post=Y_post,
        H=H,
        Sv=Sv,
        theta_init=theta_true.copy(),
        post_std_init=post_std_init,
        n_production=N_PROD,
        burnin=BURNIN,
        adapt_every=200,
        check_every=N_PROD,     # sjekk én gang til slutt
        max_recalib=0,
        scale_init=0.15,        # lavere startscale — unc-rom trenger smalere trinn
        seed=123,
        verbose=True,
        save_prefix=save_prefix,
        use_reparam=True,
    )

    acc = meta.get("acc_rate", float("nan"))
    print(f"\n  Aksepteringsrate: {acc:.3f}")
    print(f"  Chain shape: {prod_ch.shape}")
    print(f"  use_reparam i meta: {meta['use_reparam']}")

    # ── Valider resultater ────────────────────────────────────────────────────
    print("\n[4/4] Validerer …")

    n_ok = 0
    n_reparam_ok = 0
    header = f"  {'Parameter':12s}  {'θ_true':>8s}  {'post_mean':>9s}  {'post_std':>8s}  {'|bias|/σ':>8s}  {'OK?'}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for i, name in enumerate(PARAM_NAMES):
        true_val  = theta_true[i]
        post_mean = float(np.mean(prod_ch[:, i]))
        post_std  = float(np.std(prod_ch[:, i]))
        bias_sig  = abs(post_mean - true_val) / max(post_std, 1e-10)
        ok        = bias_sig < 2.0
        if ok:
            n_ok += 1
        if name in REPARAM_PARAMS:
            lb, ub = PARAM_PRIORS[name][-2], PARAM_PRIORS[name][-1]
            all_in_bounds = bool(np.all(prod_ch[:, i] > lb) and np.all(prod_ch[:, i] < ub))
            if all_in_bounds:
                n_reparam_ok += 1
            status = "OK✓" if ok and all_in_bounds else ("BOUNDS!" if not all_in_bounds else "BIAS!")
        else:
            status = "OK" if ok else "BIAS!"
        print(f"  {name:12s}  {true_val:8.4f}  {post_mean:9.4f}  {post_std:8.4f}  {bias_sig:8.2f}  {status}")

    # Priorgrensesjekk for reparametriserte parametre
    print(f"\n  Priorgrenser OK for {n_reparam_ok}/{len(REPARAM_PARAMS)} reparametriserte param: ", end="")
    reparam_ok = n_reparam_ok == len(REPARAM_PARAMS)
    print("PASS ✓" if reparam_ok else "FAIL ✗")

    # Gjenvinnbar andel
    frac_ok = n_ok / N_PARAMS
    threshold = 14 / 20   # C5 §5: ≥14/20
    recovery_ok = frac_ok >= threshold
    print(f"  Gjenvinning: {n_ok}/{N_PARAMS} parametre innenfor ±2σ — ", end="")
    print("PASS ✓" if recovery_ok else f"FAIL ✗ (krav: ≥{int(threshold*N_PARAMS)})")

    # PSRF-sjekk (løs for kort kjøring)
    from nemo.estimation.mcmc import compute_psrf, compute_ess
    psrf = compute_psrf(prod_ch)
    psrf_max = float(np.nanmax(psrf))
    ess_vals = [compute_ess(prod_ch[:, i]) for i in range(N_PARAMS)]
    ess_min = min(ess_vals)
    # PSRF og ESS er rådgivende — kort kjøring med 120 obs gir naturlig lav ESS
    psrf_ok = psrf_max < 2.50
    ess_ok = ess_min > 30
    print(f"  PSRF_max={psrf_max:.3f} — {'OK ✓' if psrf_ok else 'ADVARSEL ⚠ (mål: <2.50; forventet lav med 120 obs)'}")
    print(f"  ESS_min={ess_min:.0f}   — {'OK ✓' if ess_ok else 'ADVARSEL ⚠ (mål: >30; forventet lav med 120 obs)'}")

    # Aksepteringsrate er bindende
    acc_ok = 0.10 <= acc <= 0.60
    print(f"  Acc-rate={acc:.3f}    — {'PASS ✓' if acc_ok else 'FAIL ✗ (krav: 0.10–0.60)'}")

    # Oppsummering — kun bindende kriterier avgjør PASS/FAIL
    all_pass = reparam_ok and recovery_ok and acc_ok
    print("\n" + "=" * 65)
    if all_pass:
        print("  RESULTAT: ALLE BINDENDE SYN-TEST KRITERIER BESTÅTT ✓")
        if not (psrf_ok and ess_ok):
            print("  NB: PSRF/ESS advarsler er forventet med 120-obs datasett.")
            print("      Full konvergens testes i produksjonskjøringen (200k trekk).")
        print("  Logit-reparametrisering er klar for full produksjonskjøring.")
    else:
        print("  RESULTAT: BINDENDE KRITERIER IKKE BESTÅTT ✗")
        print("  Sjekk detaljer ovenfor før produksjonskjøring.")
    print("=" * 65)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
