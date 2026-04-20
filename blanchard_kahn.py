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

    n_unstable = int(np.sum(lams > 1.001))
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
        # Schur-projeksjon (stabil blokk)
        diag['method'] = 'Schur'
        A11 = AA[:n_stable, :n_stable]
        B11 = BB[:n_stable, :n_stable]
        Q1  = Qmat[:n_stable, :]
        G1s = np.linalg.solve(A11, B11)
        Rs  = np.linalg.solve(A11, Q1 @ Psi)
        T_hat = np.zeros((NZ, NZ), dtype=complex)
        T_hat[:n_stable, :n_stable] = G1s
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
