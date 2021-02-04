"""Tests for lowrank/gram_grad.py."""

from test.implementation.autograd import AutogradExtensions
from test.implementation.backpack import BackpackExtensions
from test.problem import make_test_problems
from test.settings import SETTINGS
from test.utils import check_sizes_and_values, remove_zeros

import pytest

PROBLEMS = make_test_problems(SETTINGS)
IDS = [problem.make_id() for problem in PROBLEMS]


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_GramBatchGradHook_get_result(problem):
    """Compare gradient Gram matrix computed with BackPACK and autodiff."""
    problem.set_up()

    backpack_res = BackpackExtensions(problem).gram_batch_grad()
    autograd_res = AutogradExtensions(problem).gram_batch_grad()

    check_sizes_and_values(backpack_res, autograd_res)
    problem.tear_down()


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_GramBatchGradHook_spectrum(problem):
    """Compare spectra of gradient Gram and gradient covariance matrix."""
    problem.set_up()

    cov_mat = AutogradExtensions(problem).cov_batch_grad()
    gram_mat = BackpackExtensions(problem).gram_batch_grad()

    cov_evals, _ = cov_mat.symeig()
    gram_evals, _ = gram_mat.symeig()

    rtol, atol = 1e-5, 1e-6
    filtered_cov_evals = remove_zeros(cov_evals, rtol=rtol, atol=atol)
    filtered_gram_evals = remove_zeros(gram_evals, rtol=rtol, atol=atol)

    check_sizes_and_values(filtered_cov_evals, filtered_gram_evals)
    problem.tear_down()
