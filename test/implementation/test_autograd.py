"""Double-check autograd implementation of some quantities."""

from test.implementation.autograd import AutogradExtensions
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
