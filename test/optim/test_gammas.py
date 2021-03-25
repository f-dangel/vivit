"""Tests computations of first-order directional derivatives ``lowrank.optim``."""

from test.implementation.autograd import AutogradExtensions
from test.implementation.backpack import BackpackExtensions
from test.problem import make_test_problems
from test.settings import SETTINGS
from test.utils import check_sizes_and_values

import pytest

PROBLEMS = make_test_problems(SETTINGS)
IDS = [problem.make_id() for problem in PROBLEMS]

PROBLEMS_REDUCTION_MEAN = []
IDS_REDUCTION_MEAN = []

for problem, id_str in zip(PROBLEMS, IDS):
    if problem.reduction_string() == "mean":
        PROBLEMS_REDUCTION_MEAN.append(problem)
        IDS_REDUCTION_MEAN.append(id_str)

GGN_SUBSAMPLINGS = [
    None,
    [0],
    [0, 0],
]
GGN_SUBSAMPLINGS_IDS = [
    f"ggn_subsampling={subsampling}" for subsampling in GGN_SUBSAMPLINGS
]
GRAD_SUBSAMPLINGS = [
    None,
    [0],
    [0, 0],
]
GRAD_SUBSAMPLINGS_IDS = [
    f"grad_subsampling={subsampling}" for subsampling in GRAD_SUBSAMPLINGS
]

TOP_SPACES = [
    0.0,
    0.1,
    0.3,
]
TOP_SPACES_IDS = [f"top_space={top_space}" for top_space in TOP_SPACES]


@pytest.mark.parametrize("top_space", TOP_SPACES, ids=TOP_SPACES_IDS)
@pytest.mark.parametrize(
    "grad_subsampling", GRAD_SUBSAMPLINGS, ids=GRAD_SUBSAMPLINGS_IDS
)
@pytest.mark.parametrize("ggn_subsampling", GGN_SUBSAMPLINGS, ids=GGN_SUBSAMPLINGS_IDS)
@pytest.mark.parametrize("problem", PROBLEMS_REDUCTION_MEAN, ids=IDS_REDUCTION_MEAN)
def test_gamma_ggn(problem, top_space, ggn_subsampling, grad_subsampling):
    """Compare first-order directional derivatives ``γ[n, d]`` with autograd.

    Args:
        top_space (float): Ratio (between 0 and 1, relative to the nontrivial
            eigenspace) of leading eigenvectors that will be used as directions.
        ggn_subsampling ([int], optional): Sample indices used for the GGN.
        grad_subsampling ([int], optional): Sample indices used for individual
            gradients.
    """
    problem.set_up()

    autograd_res = AutogradExtensions(problem).gammas_ggn(
        top_space, ggn_subsampling=ggn_subsampling, grad_subsampling=grad_subsampling
    )
    backpack_res = BackpackExtensions(problem).gammas_ggn(
        top_space, ggn_subsampling=ggn_subsampling, grad_subsampling=grad_subsampling
    )

    # the directions can sometimes point in the opposite direction, leading
    # to gammas of same magnitude but opposite sign.
    autograd_res = autograd_res.abs()
    backpack_res = backpack_res.abs()

    rtol = 5e-3
    atol = 1e-5

    check_sizes_and_values(autograd_res, backpack_res, atol=atol, rtol=rtol)
    problem.tear_down()
