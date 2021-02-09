"""Tests for lowrank/gram_grad.py."""

from test.implementation.autograd import AutogradExtensions
from test.implementation.backpack import BackpackExtensions
from test.problem import make_test_problems
from test.settings import SETTINGS
from test.utils import check_sizes_and_values, remove_zeros

import pytest

from lowrank.extensions.firstorder.batch_grad.gram_batch_grad import GramBatchGrad

PROBLEMS = make_test_problems(SETTINGS)
IDS = [problem.make_id() for problem in PROBLEMS]


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_GramBatchGrad_get_result(problem):
    """Compare gradient Gram matrix computed with BackPACK and autodiff."""
    problem.set_up()

    backpack_res = BackpackExtensions(problem).gram_batch_grad()
    autograd_res = AutogradExtensions(problem).gram_batch_grad()

    check_sizes_and_values(backpack_res, autograd_res)
    problem.tear_down()


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_GramBatchGrad_spectrum(problem):
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


FREE_GRAD_BATCH = [True, False]
FREE_GRAD_BATCH_IDS = [f"free_grad_batch={f}" for f in FREE_GRAD_BATCH]


@pytest.mark.parametrize("free_grad_batch", FREE_GRAD_BATCH, ids=FREE_GRAD_BATCH_IDS)
@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_GramBatchGrad_free_grad_batch(problem, free_grad_batch):
    """Check that ``grad_batch`` is deleted if enabled."""
    problem.set_up()

    BackpackExtensions(problem).gram_batch_grad(free_grad_batch=free_grad_batch)

    for p in problem.model.parameters():
        if free_grad_batch:
            assert not hasattr(p, GramBatchGrad._SAVEFIELD_GRAD_BATCH)
        else:
            assert hasattr(p, GramBatchGrad._SAVEFIELD_GRAD_BATCH)

    problem.tear_down()
