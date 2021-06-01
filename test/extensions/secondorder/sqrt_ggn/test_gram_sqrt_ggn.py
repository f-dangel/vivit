"""Tests for ``vivit.extensions.secondorder.sqrt_ggn.gram_sqrt_ggn.py``."""

from test.implementation.autograd import AutogradExtensions
from test.implementation.backpack import BackpackExtensions
from test.problem import make_test_problems
from test.settings import SETTINGS
from test.utils import check_sizes_and_values

import pytest

from vivit.utils.eig import symeig

PROBLEMS = make_test_problems(SETTINGS)
IDS = [problem.make_id() for problem in PROBLEMS]


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_gram_sqrt_ggn_spectrum(problem):
    """Compare spectrum of full GGN with GGN Gram matrix."""
    problem.set_up()

    ggn_mat = AutogradExtensions(problem).ggn()
    gram_mat = BackpackExtensions(problem).gram_sqrt_ggn()

    rtol, atol = 1e-5, 1e-6
    ggn_evals, _ = symeig(ggn_mat, atol=atol, rtol=rtol)
    gram_evals, _ = symeig(gram_mat, atol=atol, rtol=rtol)

    rtol, atol = 1e-5, 1e-7
    check_sizes_and_values(ggn_evals, gram_evals, rtol=rtol, atol=atol)
    problem.tear_down()


MC_SAMPLES = [1, 5, 10]
MC_SAMPLES_IDS = [f"mc_samples={s}" for s in MC_SAMPLES]


@pytest.mark.parametrize("mc_samples", MC_SAMPLES, ids=MC_SAMPLES_IDS)
@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_gram_sqrt_ggn_mc_spectrum(problem, mc_samples):
    """Compare spectrum of full GGNMC with GGNMC Gram matrix."""
    problem.set_up()
    ggn_mat = BackpackExtensions(problem).ggn_mc(mc_samples)
    problem.tear_down()

    # need another set_up to reset the random seed for the MC samples
    problem.set_up()
    gram_mat = BackpackExtensions(problem).gram_sqrt_ggn_mc(mc_samples)
    problem.tear_down()

    rtol, atol = 1e-5, 1e-6
    ggn_evals, _ = symeig(ggn_mat, atol=atol, rtol=rtol)
    gram_evals, _ = symeig(gram_mat, atol=atol, rtol=rtol)

    rtol, atol = 1e-5, 1e-6
    check_sizes_and_values(ggn_evals, gram_evals, rtol=rtol, atol=atol)


FREE_SQRT_GGN = [True, False]
FREE_SQRT_GGN_IDS = [f"free_sqrt_ggn={f}" for f in FREE_SQRT_GGN]

LAYERWISE = [True, False]
LAYERWISE_IDS = [f"layerwise={f}" for f in LAYERWISE]


@pytest.mark.parametrize("free_sqrt_ggn", FREE_SQRT_GGN, ids=FREE_SQRT_GGN_IDS)
@pytest.mark.parametrize("layerwise", LAYERWISE, ids=LAYERWISE_IDS)
@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_GramSqrtGGNExact_free_sqrt_ggn(problem, layerwise, free_sqrt_ggn):
    """Check that ``sqrt_ggn_exact`` and layerwise matrices are deleted if enabled."""
    problem.set_up()

    BackpackExtensions(problem).gram_sqrt_ggn(
        layerwise=layerwise, free_sqrt_ggn=free_sqrt_ggn
    )

    for p in problem.model.parameters():
        if free_sqrt_ggn:
            assert not hasattr(p, "sqrt_ggn_exact")
        else:
            assert hasattr(p, "sqrt_ggn_exact")

        if layerwise:
            assert p.gram_sqrt_ggn_exact is not None
        else:
            assert p.gram_sqrt_ggn_exact is None

    problem.tear_down()


@pytest.mark.parametrize("free_sqrt_ggn", FREE_SQRT_GGN, ids=FREE_SQRT_GGN_IDS)
@pytest.mark.parametrize("layerwise", LAYERWISE, ids=LAYERWISE_IDS)
@pytest.mark.parametrize("mc_samples", MC_SAMPLES, ids=MC_SAMPLES_IDS)
@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_GramSqrtGGNMC_free_sqrt_ggn(problem, mc_samples, layerwise, free_sqrt_ggn):
    """Check that ``sqrt_ggn_mc`` and layerwise matrices are deleted if enabled."""
    problem.set_up()

    BackpackExtensions(problem).gram_sqrt_ggn_mc(
        mc_samples, layerwise=layerwise, free_sqrt_ggn=free_sqrt_ggn
    )

    for p in problem.model.parameters():
        if free_sqrt_ggn:
            assert not hasattr(p, "sqrt_ggn_mc")
        else:
            assert hasattr(p, "sqrt_ggn_mc")

        if layerwise:
            assert p.gram_sqrt_ggn_mc is not None
        else:
            assert p.gram_sqrt_ggn_mc is None

    problem.tear_down()
