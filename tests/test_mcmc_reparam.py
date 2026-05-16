"""Integrasjonstester for use_reparam-flagget i adaptive_mcmc_with_monitoring."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.slow


@pytest.fixture
def short_mcmc_inputs(tmp_path):
    """Last data og bygg matriser for kort MCMC-kjøring."""
    from nemo.estimation.mcmc import KM, PARAM_NAMES, build_H, build_Sv

    rot = Path(__file__).resolve().parents[1]
    data_sti = rot / "data" / "processed" / "nemo_data_faktisk_v2.csv"
    if not data_sti.exists():
        pytest.skip("Datafil mangler — skip integrasjonstest")

    df = pd.read_csv(data_sti, index_col=0)
    df.index = pd.to_datetime(df.index)
    Y_pre = df[df.index <= "2019-12-31"].values
    Y_post = df[df.index >= "2022-01-01"].values

    return {
        "Y_pre": Y_pre,
        "Y_post": Y_post,
        "H": build_H(),
        "Sv": build_Sv(),
        "theta_init": np.array([KM.get(n, 0.5) for n in PARAM_NAMES], dtype=float),
        "post_std_init": np.array([0.05] * len(PARAM_NAMES), dtype=float),
        "save_prefix": str(tmp_path / "test_chain"),
    }


def test_use_reparam_kjorer_uten_feil(short_mcmc_inputs):
    """Veldig kort kjøring (200 burnin + 200 prod) med reparam aktiv skal fullføres."""
    from nemo.estimation.mcmc import adaptive_mcmc_with_monitoring

    prod_ch, lp_prod, meta = adaptive_mcmc_with_monitoring(
        Y_pre=short_mcmc_inputs["Y_pre"],
        Y_post=short_mcmc_inputs["Y_post"],
        H=short_mcmc_inputs["H"],
        Sv=short_mcmc_inputs["Sv"],
        theta_init=short_mcmc_inputs["theta_init"],
        post_std_init=short_mcmc_inputs["post_std_init"],
        n_production=200,
        burnin=200,
        adapt_every=50,
        check_every=200,
        max_recalib=0,
        scale_init=0.3,
        seed=123,
        verbose=False,
        save_prefix=short_mcmc_inputs["save_prefix"],
        use_reparam=True,
    )

    assert prod_ch.shape == (200, len(short_mcmc_inputs["theta_init"]))
    assert lp_prod.shape == (200,)
    assert meta["use_reparam"] is True


def test_reparam_chain_innenfor_priorgrenser(short_mcmc_inputs):
    """Chain i naturlig rom skal ha h_c og psi_R innenfor (lb, ub)."""
    from nemo.estimation.mcmc import (
        PARAM_NAMES, PARAM_PRIORS, adaptive_mcmc_with_monitoring,
    )
    from nemo.estimation.reparam import REPARAM_PARAMS

    prod_ch, _, _ = adaptive_mcmc_with_monitoring(
        **{k: v for k, v in short_mcmc_inputs.items()},
        n_production=200, burnin=200, adapt_every=50, check_every=200,
        max_recalib=0, scale_init=0.3, seed=123, verbose=False,
        use_reparam=True,
    )

    for name in REPARAM_PARAMS:
        i = PARAM_NAMES.index(name)
        lb, ub = PARAM_PRIORS[name][-2], PARAM_PRIORS[name][-1]
        chain_vals = prod_ch[:, i]
        assert np.all(chain_vals > lb), f"{name}: noen verdier ≤ {lb}"
        assert np.all(chain_vals < ub), f"{name}: noen verdier ≥ {ub}"


def test_reparam_lagrer_unc_npy(short_mcmc_inputs):
    """Når use_reparam=True skal _unc.npy lagres ved siden av .npy."""
    from nemo.estimation.mcmc import adaptive_mcmc_with_monitoring

    adaptive_mcmc_with_monitoring(
        **{k: v for k, v in short_mcmc_inputs.items()},
        n_production=200, burnin=200, adapt_every=50, check_every=200,
        max_recalib=0, scale_init=0.3, seed=123, verbose=False,
        use_reparam=True,
    )

    prefix = short_mcmc_inputs["save_prefix"]
    assert Path(f"{prefix}.npy").exists(), "Naturlig-rom chain ikke lagret"
    assert Path(f"{prefix}_unc.npy").exists(), "Unc-rom chain ikke lagret"

    ch_nat = np.load(f"{prefix}.npy")
    ch_unc = np.load(f"{prefix}_unc.npy")
    assert ch_nat.shape == ch_unc.shape


def test_uten_reparam_uendret_oppførsel(short_mcmc_inputs):
    """Med use_reparam=False skal verken _unc.npy lagres eller meta['use_reparam']=True."""
    from nemo.estimation.mcmc import adaptive_mcmc_with_monitoring

    _, _, meta = adaptive_mcmc_with_monitoring(
        **{k: v for k, v in short_mcmc_inputs.items()},
        n_production=200, burnin=200, adapt_every=50, check_every=200,
        max_recalib=0, scale_init=0.3, seed=123, verbose=False,
        use_reparam=False,
    )

    assert meta["use_reparam"] is False
    prefix = short_mcmc_inputs["save_prefix"]
    assert not Path(f"{prefix}_unc.npy").exists(), "_unc.npy skal ikke lagres når reparam=False"


def test_posterior_json_inneholder_use_reparam_flag(short_mcmc_inputs):
    """meta i posterior JSON skal inneholde use_reparam."""
    from nemo.estimation.mcmc import adaptive_mcmc_with_monitoring

    adaptive_mcmc_with_monitoring(
        **{k: v for k, v in short_mcmc_inputs.items()},
        n_production=200, burnin=200, adapt_every=50, check_every=200,
        max_recalib=0, scale_init=0.3, seed=123, verbose=False,
        use_reparam=True,
    )

    with open(f"{short_mcmc_inputs['save_prefix']}_posterior.json") as f:
        post = json.load(f)
    assert post["meta"]["use_reparam"] is True
