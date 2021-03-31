"""Test ``lowrank.optim.computations``."""

from test.implementation.optim_autograd import AutogradOptimExtensions
from test.implementation.optim_backpack import BackpackOptimExtensions
from test.optim.settings import IDS_REDUCTION_MEAN, PROBLEMS_REDUCTION_MEAN
from test.utils import check_sizes_and_values

import pytest

TOP_K = [1, 5]
TOP_K_IDS = [f"top_k={k}" for k in TOP_K]


@pytest.mark.parametrize("top_k", TOP_K, ids=TOP_K_IDS)
@pytest.mark.parametrize("problem", PROBLEMS_REDUCTION_MEAN, ids=IDS_REDUCTION_MEAN)
def test_computations_gammas_ggn(problem, top_k):
    """Compare optimizer's 1st-order directional derivatives ``γ[n, d]`` along leading
    GGN eigenvectors with autograd.

    Args:
        problem (ExtensionsTestProblem): Test case.
        top_k (int): Number of leading eigenvectors used as directions. Will be clipped
            to ``[1, max]`` with ``max`` the maximum number of nontrivial eigenvalues.
    """
    problem.set_up()

    autograd_res = AutogradOptimExtensions(problem).gammas_ggn(top_k)
    backpack_res = BackpackOptimExtensions(problem).gammas_ggn(top_k)

    # the directions can sometimes point in the opposite direction, leading
    # to gammas of same magnitude but opposite sign.
    autograd_res = autograd_res.abs()
    backpack_res = backpack_res.abs()

    rtol = 5e-3
    atol = 1e-5

    check_sizes_and_values(autograd_res, backpack_res, atol=atol, rtol=rtol)
    problem.tear_down()


@pytest.mark.parametrize("top_k", TOP_K, ids=TOP_K_IDS)
@pytest.mark.parametrize("problem", PROBLEMS_REDUCTION_MEAN, ids=IDS_REDUCTION_MEAN)
def test_computations_lambdas_ggn(problem, top_k):
    """Compare optimizer's 2nd-order directional derivatives ``λ[n, d]`` along leading
    GGN eigenvectors with autograd.

    Args:
        problem (ExtensionsTestProblem): Test case.
        top_k (int): Number of leading eigenvectors used as directions. Will be clipped
            to ``[1, max]`` with ``max`` the maximum number of nontrivial eigenvalues.
    """
    problem.set_up()

    autograd_res = AutogradOptimExtensions(problem).lambdas_ggn(top_k)
    backpack_res = BackpackOptimExtensions(problem).lambdas_ggn(top_k)

    check_sizes_and_values(autograd_res, backpack_res)
    problem.tear_down()
