"""Tests for ``vivit.extensions.secondorder.sqrt_ggn.__init__.py``."""

from test.implementation.autograd import AutogradExtensions
from test.implementation.backpack import BackpackExtensions
from test.problem import make_test_problems
from test.settings import SETTINGS
from test.utils import check_sizes_and_values

import pytest
import torch

PROBLEMS = make_test_problems(SETTINGS)
IDS = [problem.make_id() for problem in PROBLEMS]

SUBSAMPLINGS = [None, [0], [0, 0]]
SUBSAMPLINGS_IDS = [f"subsampling={subsampling}" for subsampling in SUBSAMPLINGS]


@pytest.mark.parametrize("subsampling", SUBSAMPLINGS, ids=SUBSAMPLINGS_IDS)
@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_ggn(problem, subsampling):
    """Compare full GGN computed from sqrt decomposition and via autograd.

    Args:
        subsampling ([int]): Indices of samples in the mini-batch for which
            the GGN/Fisher should be computed and summed. ``None`` uses the
            entire mini-batch.
    """
    problem.set_up()

    autograd_res = AutogradExtensions(problem).ggn(subsampling=subsampling)
    backpack_res = BackpackExtensions(problem).ggn(subsampling=subsampling)

    check_sizes_and_values(autograd_res, backpack_res)
    problem.tear_down()


@pytest.mark.parametrize("subsampling", SUBSAMPLINGS, ids=SUBSAMPLINGS_IDS)
@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_ggn_mat_prod(problem, subsampling, V=3):
    """Compare multiplication with the GGN with multiplication by its factors.

    Args:
        subsampling ([int]): Indices of samples in the mini-batch for which
            the GGN/Fisher should be multiplied with. ``None`` uses the
            entire mini-batch.
        V (int, optional): Number of vectors to multiply with in parallel.
    """
    problem.set_up()

    mat_list = rand_mat_list_like_parameters(problem, V)

    autograd_res = AutogradExtensions(problem).ggn_mat_prod(
        mat_list, subsampling=subsampling
    )
    backpack_res = BackpackExtensions(problem).ggn_mat_prod(
        mat_list, subsampling=subsampling
    )

    atol = 1e-6
    rtol = 1e-5

    check_sizes_and_values(autograd_res, backpack_res, atol=atol, rtol=rtol)
    problem.tear_down()


MC_ATOL = 2e-3
MC_RTOL = 1e-2
MC_SAMPLES = 500000
MC_CHUNKS = 20


@pytest.mark.expensive
@pytest.mark.parametrize("subsampling", SUBSAMPLINGS, ids=SUBSAMPLINGS_IDS)
@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_ggn_mc(problem, subsampling):
    """Test MC approximation of the GGN using a large number of samples.

    Args:
        subsampling ([int]): Indices of samples in the mini-batch for which
            the MC-GGN/Fisher should be computed and summed. ``None`` uses the
            entire mini-batch.
    """
    problem.set_up()

    autograd_res = AutogradExtensions(problem).ggn(subsampling=subsampling)
    backpack_res = BackpackExtensions(problem).ggn_mc_chunk(
        MC_SAMPLES, chunks=MC_CHUNKS, subsampling=subsampling
    )

    check_sizes_and_values(autograd_res, backpack_res, atol=MC_ATOL, rtol=MC_RTOL)
    problem.tear_down()


@pytest.mark.expensive
@pytest.mark.parametrize("subsampling", SUBSAMPLINGS, ids=SUBSAMPLINGS_IDS)
@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_ggn_mc_mat_prod(problem, subsampling, V=3):
    """Compare GGN multiplication with multiplication by its MC-approximated factors.

    Args:
        subsampling ([int]): Indices of samples in the mini-batch for which
            the GGN/Fisher should be multiplied with. ``None`` uses the
            entire mini-batch.
        V (int, optional): Number of vectors to multiply with in parallel.
    """
    problem.set_up()

    mat_list = rand_mat_list_like_parameters(problem, V)

    autograd_res = AutogradExtensions(problem).ggn_mat_prod(
        mat_list, subsampling=subsampling
    )
    backpack_res = BackpackExtensions(problem).ggn_mc_mat_prod_chunk(
        mat_list, mc_samples=MC_SAMPLES, subsampling=subsampling, chunks=MC_CHUNKS
    )

    check_sizes_and_values(autograd_res, backpack_res, atol=MC_ATOL, rtol=MC_RTOL)
    problem.tear_down()


def rand_mat_list_like_parameters(problem, V):
    """Create list of random matrix with same trailing dimensions as parameters."""
    return [
        torch.rand(V, *p.shape, device=problem.device)
        for p in problem.model.parameters()
    ]
