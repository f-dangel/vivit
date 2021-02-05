"""Tests for ``lowrank.extensions.secondorder.sqrt_ggn.__init__.py``."""

from test.implementation.autograd import AutogradExtensions
from test.implementation.backpack import BackpackExtensions
from test.problem import make_test_problems
from test.settings import SETTINGS
from test.utils import check_sizes_and_values

import pytest

PROBLEMS = make_test_problems(SETTINGS)
IDS = [problem.make_id() for problem in PROBLEMS]


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_ggn(problem):
    """Compare full GGN computed from sqrt decomposition and via autograd."""
    problem.set_up()

    autograd_res = AutogradExtensions(problem).ggn()
    backpack_res = BackpackExtensions(problem).ggn()

    check_sizes_and_values(autograd_res, backpack_res)
    problem.tear_down()
