"""
[STAT] Logit-reparametrisering for psi_R (Fase 2, C5 §2).

psi_R treffer øvre priorbegrensning. Likelihood har sannsynligvis en
rygg som fortsetter inn i ikke-tillatt område. Logit-transformasjon flytter
grensene til ±∞ og avdekker om posterior er genuint konsentrert ved grensen
eller en numerisk artefakt.

h_c er fjernet fra reparametrisering — kalibreres fast til H_C_FIXED=0.938
(PE-godkjent 2026-05-18, C2 Alt A).

Transformasjon:
    Naturlig rom: x ∈ (lb, ub)
    Normalisert:  u = (x − lb) / (ub − lb) ∈ (0, 1)
    Logit:        y = log(u / (1 − u)) ∈ ℝ

    Invers: x = lb + (ub − lb) · σ(y) der σ(y) = 1 / (1 + e^(−y))

Jacobian (for endring av variabel i log-posterior):
    dx/dy = (ub − lb) · σ(y) · (1 − σ(y)) = (ub − lb) · u · (1 − u)

Når vi sampler i y-rom, må vi legge til log|dx/dy| til log-posterior:
    log_post_y(y) = log_post_x(x) + log|dx/dy|
                  = log_post_x(x) + log(ub − lb) + log(u) + log(1 − u)
"""

from __future__ import annotations

import numpy as np

from nemo.estimation.mcmc import PARAM_NAMES, PARAM_PRIORS

# Parametre som transformeres til ubegrenset rom
# h_c fjernet 2026-05-18 — fast til H_C_FIXED=0.938 (PE-godkjent, C2 Alt A)
# psi_R fjernet 2026-05-24 — fast til PSI_R_FIXED=0.667 (PE-godkjent, kj11)
REPARAM_PARAMS: tuple[str, ...] = ()


def _bounds(name: str) -> tuple[float, float]:
    spec = PARAM_PRIORS[name]
    return float(spec[-2]), float(spec[-1])


def _indices() -> list[int]:
    return [PARAM_NAMES.index(n) for n in REPARAM_PARAMS]


def to_unconstrained(theta_nat: np.ndarray) -> np.ndarray:
    """Transformer fra naturlig til ubegrenset rom for REPARAM_PARAMS.

    Verdier strengt innenfor (lb, ub) returneres som logit; verdier ved eller
    utenfor grensene klippes til (lb + ε, ub − ε) for numerisk stabilitet.
    """
    theta_unc = theta_nat.astype(float).copy()
    eps = 1e-10
    for name in REPARAM_PARAMS:
        i = PARAM_NAMES.index(name)
        lb, ub = _bounds(name)
        x = np.clip(theta_nat[i], lb + eps, ub - eps)
        u = (x - lb) / (ub - lb)
        theta_unc[i] = np.log(u / (1.0 - u))
    return theta_unc


def to_natural(theta_unc: np.ndarray) -> np.ndarray:
    """Invers: transformer fra ubegrenset til naturlig rom for REPARAM_PARAMS."""
    theta_nat = theta_unc.astype(float).copy()
    for name in REPARAM_PARAMS:
        i = PARAM_NAMES.index(name)
        lb, ub = _bounds(name)
        # Numerisk stabil sigmoid
        y = theta_unc[i]
        if y >= 0:
            u = 1.0 / (1.0 + np.exp(-y))
        else:
            ez = np.exp(y)
            u = ez / (1.0 + ez)
        theta_nat[i] = lb + (ub - lb) * u
    return theta_nat


def log_jacobian(theta_nat: np.ndarray) -> float:
    """log|dx/dy| evaluert ved theta_nat (naturlig rom).

    For hver reparametrisert parameter i:
        log|dx_i/dy_i| = log(ub_i − lb_i) + log(u_i) + log(1 − u_i)
    der u_i = (x_i − lb_i) / (ub_i − lb_i).

    Returnerer summen over alle reparametriserte parametre. Returnerer −∞
    hvis theta_nat ligger utenfor (lb, ub) — log-posterior i unc-rom blir da
    også −∞, hvilket samsvarer med at slike punkter ikke kan nås fra
    endelige y-verdier.
    """
    log_jac = 0.0
    eps = 1e-15
    for name in REPARAM_PARAMS:
        i = PARAM_NAMES.index(name)
        lb, ub = _bounds(name)
        x = theta_nat[i]
        if not (lb < x < ub):
            return -np.inf
        u = (x - lb) / (ub - lb)
        log_jac += np.log(ub - lb) + np.log(max(u, eps)) + np.log(max(1.0 - u, eps))
    return float(log_jac)


def wrap_log_posterior(log_posterior_nat):
    """Returner en versjon av log_posterior som tar theta i ubegrenset rom.

    Bruksmønster:
        from nemo.estimation.mcmc import log_posterior
        from nemo.estimation.reparam import wrap_log_posterior
        log_post_unc = wrap_log_posterior(log_posterior)
        lp = log_post_unc(theta_unc, H, Sv, Y_pre, Y_post)

    log_post_unc samsvarer med log-tetthet i y-rom — inkluderer Jacobian.
    """
    def log_post_unc(theta_unc, *args, **kwargs):
        theta_nat = to_natural(theta_unc)
        lp_nat = log_posterior_nat(theta_nat, *args, **kwargs)
        if not np.isfinite(lp_nat):
            return -np.inf
        jac = log_jacobian(theta_nat)
        if not np.isfinite(jac):
            return -np.inf
        return lp_nat + jac

    return log_post_unc
