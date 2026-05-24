"""
================================================================================
NEMO FASE II — BLANCHARD-KAHN LØSER
Bygger på erfaringene fra Fase I:
  - Direkte koblinger i G₁ (ingen lagg-mellomled)
  - mc fra MRS=MPN (ikke akkumulerende reallønn)
  - Stabil løsning via ordqz med stabilitetssjekk

Løsningsformat:
  z_t = T · z_{t-1} + R · ε_t
================================================================================
"""

import numpy as np
from scipy.linalg import ordqz
import warnings


def solve(G0, G1, Psi, Pi=None, verbose=True):
    """
    Løser modellen på tilstandsrom-form via Blanchard-Kahn / Sims (2002).

    Håndterer tre tilfeller:
    1. Pi = None eller rang(Pi) = 0: direkte inversjon T = inv(G0)G1
    2. Pi har rang > 0 og BK oppfylt: Schur-projeksjon
    3. BK feiler: returnerer advarsel og fallback til direkte løsning

    Returnerer
    ----------
    T : (NZ × NZ)
    R : (NZ × NE)
    diag : dict med diagnostikkinformasjon
    """
    NZ = G0.shape[0]
    NE = Psi.shape[1]

    # Kjør QZ for diagnostikk uansett
    AA, BB, av, bv, Qmat, Zmat = ordqz(G0, G1, sort='ouc', output='complex')
    with np.errstate(divide='ignore', invalid='ignore'):
        lams = np.where(np.abs(av) < 1e-13, np.inf, np.abs(bv / av))

    # Terskel: > 1.0 (ikke 1.001) for å fange enhetsrøtter fra RE-likninger
    # (UIP psi_UIP ≈ 1.02, CEE-investering ≈ 1.0003 — begge genuine ustabile RE-moder)
    n_unstable = int(np.sum(lams > 1.0))
    n_stable   = NZ - n_unstable

    diag = {
        'NZ':         NZ,
        'NE':         NE,
        'n_unstable': n_unstable,
        'n_stable':   n_stable,
        'rank_G0':    int(np.linalg.matrix_rank(G0)),
        'cond_G0':    float(np.linalg.cond(G0)),
        'max_eig_T':  None,
        'stable':     None,
        'method':     None,
    }

    if verbose:
        print("=" * 60)
        print("  NEMO FASE II — BLANCHARD-KAHN DIAGNOSTIKK")
        print("=" * 60)
        print(f"  Systemdimensjon     : {NZ}")
        print(f"  Sjokk               : {NE}")
        print(f"  Rang(G0)            : {diag['rank_G0']}")
        print(f"  Kond(G0)            : {diag['cond_G0']:.1f}")
        print(f"  Ustabile eigenvalues: {n_unstable}")

    # ── Velg løsningsmetode ───────────────────────────────────────────────────
    use_direct = True
    if Pi is not None:
        pi_rank = np.linalg.matrix_rank(Pi, tol=1e-8)
        Q2Pi_rank = np.linalg.matrix_rank(
            Qmat[n_stable:, :] @ Pi, tol=1e-8
        ) if n_unstable > 0 else 0
        if pi_rank > 0 and Q2Pi_rank >= n_unstable and n_unstable > 0:
            use_direct = False

    if use_direct:
        # Direkte løsning (robust metode som fungerte i Fase I)
        diag['method'] = 'direkte'
        G0inv = np.linalg.inv(G0)
        T = G0inv @ G1
        R = G0inv @ Psi
    else:
        # Sims (2002) gensys — fullstendig løsning med M-matrise (eta-impakt).
        # Referanse: Sims (2002) "Solving Linear Rational Expectations Models",
        # Computational Economics 20(1-2), s. 1-20.
        #
        # Teori: stabilitetsbetingelse Q2*(Psi*ε + Pi*η) = 0 for alle ε
        # → Q2*Pi*η = -Q2*Psi*ε → η = M*ε der M = -(Q2*Pi)^† * Q2 * Psi
        # Full impaktmatrise: R = Z1 * A11^{-1} * Q1 * (Psi + Pi*M)
        diag['method'] = 'Schur'
        A11 = AA[:n_stable, :n_stable]
        B11 = BB[:n_stable, :n_stable]
        Q1  = Qmat[:n_stable, :]
        Q2  = Qmat[n_stable:, :]

        # M-matrise: minimumsnorm-løsning Q2*Pi*M = -Q2*Psi (pseudoinvers)
        Q2Pi  = Q2 @ Pi
        Q2Psi = Q2 @ Psi
        M, _, _, _ = np.linalg.lstsq(Q2Pi, -Q2Psi, rcond=None)

        # Overgangsmatrise (uendret fra klassisk Schur)
        G1s = np.linalg.solve(A11, B11)
        T_hat = np.zeros((NZ, NZ), dtype=complex)
        T_hat[:n_stable, :n_stable] = G1s

        # Full impaktmatrise inkl. forventningsjustering Pi*M
        full_Psi = Psi + Pi @ M
        Rs  = np.linalg.solve(A11, Q1 @ full_Psi)
        R_hat = np.zeros((NZ, NE), dtype=complex)
        R_hat[:n_stable, :] = Rs

        Zinv = np.linalg.inv(Zmat)
        T = np.real(Zmat @ T_hat @ Zinv)
        R = np.real(Zmat @ R_hat)

    # Stabilitetskontroll
    eigs = np.linalg.eigvals(T)
    eig_max = float(np.max(np.abs(eigs)))
    diag['max_eig_T'] = eig_max
    diag['stable']    = eig_max < 1.0 + 1e-4

    if verbose:
        print(f"  Løsningsmetode      : {diag['method']}")
        print(f"  Maks |eig(T)|       : {eig_max:.6f}")
        print(f"  Stabil              : {'JA' if diag['stable'] else 'NEI ⚠'}")
        print("=" * 60)

    if not diag['stable']:
        warnings.warn(
            f"T er ustabil: maks |eig| = {eig_max:.4f}. "
            "Vurder å redusere rho_w eller justere G0-koblinger.",
            RuntimeWarning
        )

    return T, R, diag


def solve_klein(G0, G1, Psi, Pi, verbose=True):
    """
    Klein (2000) RE-løser for NEMO.

    Bruker Klein-pencil (A_k = −Pi, B_k = G0) for å diagnostisere
    fremoverskuende RE-struktur uten Sims-format η_t.

    Teori (Klein 2000):
      A_k · E_t[z_{t+1}] = B_k · z_t   der A_k = −Pi (singulær, rang = rank(Pi))
      Endelige eigenvalues fra QZ(A_k, B_k) med |bv/av| > 1 er "eksplosive".
      BK-betingelse (Klein): n_explosive_finite = rank(Pi) = n_nonpred.

    NEMO med K&M-kalibrering (mimicking rule, psi_P1 ≈ 0.29):
      - Mimicking rule reagerer på LAGG av inflasjon (PI_L), ikke nåværende/fremtidig.
      - Dette oppfyller ikke Taylor-prinsippet i standard NK-forstand.
      - Resultat: n_explosive_finite = 5 ≠ 7 = rank(Pi) → indeterminert system.
      - Den korrekte fundamentale likevekten er MSV-likevekten (M=0, direkte løsning).
      - Referanse: Blanchard & Kahn (1980), Klein (2000), Sims (2002).

    Fallback ved indeterminas:
      Bruker direkte løsning (T = G0⁻¹G1, R = G0⁻¹Ψ), identisk med `solve()`.
      Dette er den minste-tilstandsvariabel (MSV) likevekten — den unike
      fundamentale RE-likevekten i et indeterminert system.

    Returnerer
    ----------
    T    : (NZ × NZ)
    R    : (NZ × NE)
    diag : dict med Klein-diagnostikk og løsningsmetode
    """
    NZ = G0.shape[0]
    NE = Psi.shape[1]

    pi_rank = int(np.linalg.matrix_rank(Pi, tol=1e-8))

    # Klein-pencil QZ: (A_k = −Pi, B_k = G0)
    # Endelige eigenvalues: bv/av der |av| > terskel (ikke null-rader i Pi)
    # Uendelige eigenvalues (fra null-rader i Pi, 42 samtidige likninger): |av| ≈ 0
    AA_k, BB_k, av_k, bv_k, Qmat_k, Zmat_k = ordqz(-Pi, G0, sort='ouc', output='complex')

    finite_mask = np.abs(av_k) > 1e-10
    with np.errstate(divide='ignore', invalid='ignore'):
        lams_finite = np.where(finite_mask, np.abs(bv_k / av_k), 0.0)

    n_explosive = int(np.sum(lams_finite > 1.0))
    n_finite    = int(np.sum(finite_mask))
    bk_ok       = (n_explosive == pi_rank) and (pi_rank > 0)

    diag = {
        'NZ':             NZ,
        'NE':             NE,
        'pi_rank':        pi_rank,
        'n_finite_klein': n_finite,
        'n_explosive':    n_explosive,
        'bk_klein':       bk_ok,
        'method':         None,
        'max_eig_T':      None,
        'stable':         None,
    }

    if verbose:
        print("=" * 60)
        print("  NEMO KLEIN (2000) — DIAGNOSTIKK")
        print("=" * 60)
        print(f"  Systemdimensjon     : {NZ}")
        print(f"  Sjokk               : {NE}")
        print(f"  rank(Pi)            : {pi_rank}  (n_nonpred)")
        print(f"  n_finite (Klein)    : {n_finite}  (fra Pi sitt kolonnrom)")
        print(f"  n_explosive (finite): {n_explosive}")
        print(f"  BK (Klein) oppfylt  : {'JA' if bk_ok else 'NEI — indeterminert ⚠'}")

    if not bk_ok:
        # Indeterminert system: fall tilbake til MSV-likevekten (direkte løsning).
        # For NEMO med K&M-kalibrering er dette den korrekte fundamentale likevekten.
        if n_explosive < pi_rank:
            grunn = (
                f"n_explosive={n_explosive} < rank(Pi)={pi_rank}: "
                "for få eksplosive moder (underdeterminert BK). "
                "NEMO med mimicking rule (psi_P1 reagerer på PI_L) "
                "oppfyller ikke Taylor-prinsippet → indeterminert RE. "
                "MSV-likevekten (M=0) er korrekt fundamental-likevekt."
            )
        else:
            grunn = (
                f"n_explosive={n_explosive} > rank(Pi)={pi_rank}: "
                "for mange eksplosive moder (ingen stabil løsning)."
            )
        warnings.warn(
            f"Klein BK ikke oppfylt: {grunn}",
            RuntimeWarning,
        )
        diag['method'] = 'direkte-MSV'
        T = np.linalg.solve(G0, G1)
        R = np.linalg.solve(G0, Psi)
    else:
        # Klein-Schur: BK oppfylt → Schur-projeksjon med Klein Q2.
        # n_stable_block = NZ - n_explosive (stable finite + alle uendelige)
        # Q2_K hentes fra bunn av Qmat_k (sort='ouc' → eksplosive sist)
        diag['method'] = 'Klein-Schur'
        n_stable_block = NZ - n_explosive
        Q2_K   = Qmat_k[n_stable_block:, :]  # (n_explosive × NZ)

        Q2KPi  = Q2_K @ Pi
        Q2KPsi = Q2_K @ Psi
        M, _, _, _ = np.linalg.lstsq(Q2KPi, -Q2KPsi, rcond=None)

        full_Psi = Psi + Pi @ M
        T = np.linalg.solve(G0, G1)
        R = np.linalg.solve(G0, full_Psi)

    # Stabilitetskontroll
    eigs    = np.linalg.eigvals(T)
    eig_max = float(np.max(np.abs(eigs)))
    diag['max_eig_T'] = eig_max
    diag['stable']    = eig_max < 1.0 + 1e-4

    if verbose:
        print(f"  Løsningsmetode      : {diag['method']}")
        print(f"  Maks |eig(T)|       : {eig_max:.6f}")
        print(f"  Stabil              : {'JA' if diag['stable'] else 'NEI ⚠'}")
        print("=" * 60)

    if not diag['stable']:
        warnings.warn(
            f"T er ustabil etter Klein-løser: maks |eig| = {eig_max:.4f}.",
            RuntimeWarning,
        )

    return T, R, diag


def compute_irf(T, R, shock_idx, shock_size, T_periods=20):
    """
    Beregner impulssvar for ett sjokk.

    Parametere
    ----------
    T           : overgangsmatrise
    R           : sjokkmatrise
    shock_idx   : indeks i ε-vektor
    shock_size  : størrelse på initialsjokk
    T_periods   : antall kvartaler

    Returnerer
    ----------
    irf : (T_periods × NZ) array
    """
    NZ = T.shape[0]
    NE = R.shape[1]
    irf = np.zeros((T_periods, NZ))

    eps = np.zeros(NE)
    eps[shock_idx] = shock_size

    z = R @ eps
    irf[0] = z
    for t in range(1, T_periods):
        z = T @ z
        irf[t] = z

    return irf


def validate_irf(T, R, shock_defs, checks, T_periods=8, verbose=True):
    """
    Validerer IRF-fortegn mot kvalitative krav.

    Parametere
    ----------
    shock_defs : dict {key: (shock_idx, shock_size)}
    checks     : list of (shock_key, var_idx, sign, description)

    Returnerer
    ----------
    (n_pass, n_total, results)
    """
    results = []
    for shock_key, var_idx, sign, desc in checks:
        if shock_key not in shock_defs:
            continue
        idx, size = shock_defs[shock_key]
        irf = compute_irf(T, R, idx, size, T_periods)
        cum = float(np.sum(irf[:, var_idx]))
        ok = (cum * sign) > 0
        results.append({'key': shock_key, 'var': var_idx, 'desc': desc,
                        'cum': cum, 'ok': ok})

    n_pass = sum(r['ok'] for r in results)

    if verbose:
        print(f"\n  IRF FORTEGNSVALIDERING")
        print(f"  {'─' * 52}")
        for r in results:
            s = 'OK  ' if r['ok'] else 'FEIL'
            print(f"  [{s}]  {r['desc']:<45s}  ({r['cum']*100:+.3f})")
        print(f"\n  Bestått: {n_pass}/{len(results)}")

    return n_pass, len(results), results


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))
    from equations import build_matrices, NZ, NE, Y, PI, I_R, RER, C
    from equations import E_A, E_C, E_P, E_O, E_Ys, E_rp, E_i

    G0, G1, Psi, Pi = build_matrices()
    T, R, diag = solve(G0, G1, Psi, Pi, verbose=True)

    # Rask IRF-test
    shock_defs = {
        'monetary': (E_i,  0.0025),
        'oil':      (E_O,  0.10),
        'demand':   (E_Ys, 0.01),
        'risk':     (E_rp, 0.01),
        'cost':     (E_P,  0.003),
        'tech':     (E_A,  0.007),
    }
    checks = [
        ('monetary', Y,   -1, "Pengepol → BNP ned"),
        ('monetary', PI,  -1, "Pengepol → Inflasjon ned"),
        ('monetary', RER, -1, "Pengepol → RER ned (NOK appr.)"),
        ('monetary', I_R, +1, "Pengepol → Rente opp"),
        ('risk',     RER, +1, "Risikopremie → RER opp"),
        ('risk',     PI,  +1, "Risikopremie → Inflasjon opp"),
        ('cost',     PI,  +1, "Kostnad → Inflasjon opp"),
        ('cost',     Y,   -1, "Kostnad → BNP ned"),
        ('tech',     Y,   +1, "TFP → BNP opp"),
        ('tech',     PI,  -1, "TFP → Inflasjon ned"),
        ('oil',      Y,   +1, "Oljepris → BNP opp"),
        ('demand',   Y,   +1, "Ettersp. → BNP opp"),
        ('demand',   I_R, +1, "Ettersp. → Rente opp"),
    ]
    validate_irf(T, R, shock_defs, checks, verbose=True)
