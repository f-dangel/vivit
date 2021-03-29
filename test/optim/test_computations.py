"""Test ``lowrank.optim.computations``."""

from test.implementation.optim_autograd import AutogradOptimExtensions
from test.implementation.optim_backpack import BackpackOptimExtensions
from test.optim.settings import IDS_REDUCTION_MEAN, PROBLEMS_REDUCTION_MEAN
from test.utils import check_sizes_and_values

import pytest


@pytest.mark.parametrize("problem", PROBLEMS_REDUCTION_MEAN, ids=IDS_REDUCTION_MEAN)
def test_optim_gammas_ggn_top_1(problem):
    """Compare optimizer's 1st-order directional derivatives ``γ[n, d]`` along leading
    eigenvector with autograd.
    """
    problem.set_up()

    k = 1
    autograd_res = AutogradOptimExtensions(problem).gammas_ggn(k)
    backpack_res = BackpackOptimExtensions(problem).gammas_ggn(k)

    # the directions can sometimes point in the opposite direction, leading
    # to gammas of same magnitude but opposite sign.
    autograd_res = autograd_res.abs()
    backpack_res = backpack_res.abs()

    rtol = 5e-3
    atol = 1e-5

    check_sizes_and_values(autograd_res, backpack_res, atol=atol, rtol=rtol)
    problem.tear_down()


@pytest.mark.parametrize("problem", PROBLEMS_REDUCTION_MEAN, ids=IDS_REDUCTION_MEAN)
def test_optim_lambdas_ggn_top_1(problem):
    """Compare optimizer's 2nd-order directional derivatives ``λ[n, d]`` along leading
    eigenvector with autograd.
    """
    problem.set_up()

    k = 1
    autograd_res = AutogradOptimExtensions(problem).lambdas_ggn(k)
    backpack_res = BackpackOptimExtensions(problem).lambdas_ggn(k)

    check_sizes_and_values(autograd_res, backpack_res)
    problem.tear_down()
