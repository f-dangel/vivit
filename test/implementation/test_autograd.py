"""Double-check autograd implementation of some quantities."""

from test.implementation.autograd import AutogradExtensions
from test.implementation.backpack import BackpackExtensions
from test.problem import make_test_problems
from test.settings import SETTINGS
from test.utils import check_sizes_and_values

import pytest

PROBLEMS = make_test_problems(SETTINGS)
IDS = [problem.make_id() for problem in PROBLEMS]


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_ggn_implementation(problem):
    """Compare diagonal of full GGN with diagonal of block GGN."""
    problem.set_up()

    diag_ggn_from_full = AutogradExtensions(problem).diag_ggn_via_ggn()
    diag_ggn_from_block = AutogradExtensions(problem).diag_ggn()

    check_sizes_and_values(diag_ggn_from_full, diag_ggn_from_block)
    problem.tear_down()


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_cov_batch_grad_implementation(problem):
    """Compare gradient covariance matrices between BackPACK and autograd."""
    problem.set_up()

    autograd_res = AutogradExtensions(problem).cov_batch_grad()
    backpack_res = BackpackExtensions(problem).cov_batch_grad()

    check_sizes_and_values(autograd_res, backpack_res)
    problem.tear_down()
