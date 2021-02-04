"""Tests for lowrank/gram_grad.py."""

from test.implementation.autograd import AutogradExtensions
from test.implementation.backpack import BackpackExtensions
from test.problem import make_test_problems
from test.settings import SETTINGS
from test.utils import check_sizes_and_values

import pytest

PROBLEMS = make_test_problems(SETTINGS)
IDS = [problem.make_id() for problem in PROBLEMS]


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_GramBatchGradHook_get_result(problem):
    """Compare gradient Gram matrix computed with BackPACK and autodiff."""
    problem.set_up()

    backpack_res = BackpackExtensions(problem).gram_batch_grad()
    autograd_res = AutogradExtensions(problem).gram_batch_grad()

    print(backpack_res)
    print(autograd_res)

    check_sizes_and_values(backpack_res, autograd_res)
    problem.tear_down()
