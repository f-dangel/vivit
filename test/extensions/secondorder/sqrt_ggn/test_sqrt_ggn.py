"""Tests for ``lowrank.extensions.secondorder.sqrt_ggn.__init__.py``."""

from test.implementation.autograd import AutogradExtensions
from test.implementation.backpack import BackpackExtensions
from test.problem import make_test_problems
from test.settings import SETTINGS
from test.utils import check_sizes_and_values

import pytest

PROBLEMS = make_test_problems(SETTINGS)
IDS = [problem.make_id() for problem in PROBLEMS]

SUBSAMPLINGS = [None, [0]]
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


MC_ATOL = 2e-3
MC_RTOL = 1e-2


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
    mc_samples = 100000
    # NOTE May crash for large networks because of large number of samples.
    # If necessary, resolve by chunking samples into smaller batches + averaging
    backpack_res = BackpackExtensions(problem).ggn_mc(
        mc_samples, subsampling=subsampling
    )

    check_sizes_and_values(autograd_res, backpack_res, atol=MC_ATOL, rtol=MC_RTOL)
    problem.tear_down()
