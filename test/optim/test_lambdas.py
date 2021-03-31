"""Tests computations of second-order directional derivatives ``lowrank.optim``."""

from test.implementation.autograd import AutogradExtensions
from test.implementation.backpack import BackpackExtensions
from test.optim.settings import IDS_REDUCTION_MEAN, PROBLEMS_REDUCTION_MEAN
from test.utils import check_sizes_and_values

import pytest

GGN_SUBSAMPLINGS = [None, [0], [0, 0]]
GGN_SUBSAMPLINGS_IDS = [
    f"ggn_subsampling={subsampling}" for subsampling in GGN_SUBSAMPLINGS
]

LAMBDA_SUBSAMPLINGS = [None, [0], [0, 0]]
LAMBDA_SUBSAMPLINGS_IDS = [
    f"lambda_subsampling={subsampling}" for subsampling in LAMBDA_SUBSAMPLINGS
]

TOP_SPACES = [0.0, 0.1, 0.3]
TOP_SPACES_IDS = [f"top_space={top_space}" for top_space in TOP_SPACES]


@pytest.mark.parametrize("ggn_subsampling", GGN_SUBSAMPLINGS, ids=GGN_SUBSAMPLINGS_IDS)
@pytest.mark.parametrize("top_space", TOP_SPACES, ids=TOP_SPACES_IDS)
@pytest.mark.parametrize(
    "lambda_subsampling", LAMBDA_SUBSAMPLINGS, ids=LAMBDA_SUBSAMPLINGS_IDS
)
@pytest.mark.parametrize("problem", PROBLEMS_REDUCTION_MEAN, ids=IDS_REDUCTION_MEAN)
def test_lambdas_ggn(
    problem,
    top_space,
    ggn_subsampling,
    lambda_subsampling,
):
    """Compare second-order directional derivatives ``λ[n, d]`` with autograd.

    Args:
        top_space (float): Ratio (between 0 and 1, relative to the nontrivial
            eigenspace) of leading eigenvectors that will be used as directions.
        ggn_subsampling ([int], optional): Sample indices used for the GGN.
        lambda_subsampling ([int], optional): Sample indices used for individual
            gradients.
    """
    problem.set_up()

    autolambda_res = AutogradExtensions(problem).lambdas_ggn(
        top_space,
        ggn_subsampling=ggn_subsampling,
        lambda_subsampling=lambda_subsampling,
    )
    backpack_res = BackpackExtensions(problem).lambdas_ggn(
        top_space,
        ggn_subsampling=ggn_subsampling,
        lambda_subsampling=lambda_subsampling,
    )

    check_sizes_and_values(autolambda_res, backpack_res)
    problem.tear_down()
