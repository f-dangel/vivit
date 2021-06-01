"""Tests for vivit/gram_grad.py."""

from test.implementation.autograd import AutogradExtensions
from test.implementation.backpack import BackpackExtensions
from test.problem import make_test_problems
from test.settings import SETTINGS
from test.utils import check_sizes_and_values

import pytest

from vivit.extensions.firstorder.batch_grad.gram_batch_grad import (
    CenteredGramBatchGrad,
    GramBatchGrad,
)
from vivit.utils.eig import symeig

PROBLEMS = make_test_problems(SETTINGS)
IDS = [problem.make_id() for problem in PROBLEMS]


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_GramBatchGrad_get_result(problem):
    """Compare uncentered gradient Gram matrix computed with BackPACK and autodiff."""
    problem.set_up()

    backpack_res = BackpackExtensions(problem).gram_batch_grad()
    autograd_res = AutogradExtensions(problem).gram_batch_grad()

    check_sizes_and_values(backpack_res, autograd_res)
    problem.tear_down()


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_GramBatchGrad_spectrum(problem):
    """Compare spectra of uncentered gradient Gram and gradient covariance matrix."""
    problem.set_up()

    cov_mat = AutogradExtensions(problem).cov_batch_grad()
    gram_mat = BackpackExtensions(problem).gram_batch_grad()

    rtol, atol = 1e-5, 1e-6
    cov_evals, _ = symeig(cov_mat, atol=atol, rtol=rtol)
    gram_evals, _ = symeig(gram_mat, atol=atol, rtol=rtol)

    rtol, atol = 1e-5, 1e-7
    check_sizes_and_values(cov_evals, gram_evals, rtol=rtol, atol=atol)
    problem.tear_down()


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_CenteredGramBatchGrad_get_result(problem):
    """Compare centered gradient Gram matrix computed with BackPACK and autodiff."""
    problem.set_up()

    backpack_res = BackpackExtensions(problem).centered_gram_batch_grad()
    autograd_res = AutogradExtensions(problem).centered_gram_batch_grad()

    check_sizes_and_values(backpack_res, autograd_res)
    problem.tear_down()


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_CenteredGramBatchGrad_spectrum(problem):
    """Compare spectra of centered gradient Gram and gradient covariance matrix."""
    problem.set_up()

    cov_mat = AutogradExtensions(problem).centered_cov_batch_grad()
    gram_mat = BackpackExtensions(problem).centered_gram_batch_grad()

    rtol, atol = 1e-5, 1e-6
    cov_evals, _ = symeig(cov_mat, atol=atol, rtol=rtol)
    gram_evals, _ = symeig(gram_mat, atol=atol, rtol=rtol)

    check_sizes_and_values(cov_evals, gram_evals)
    problem.tear_down()


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_CenteredBatchGrad(problem):
    """Compare centered individual gradients computed by BackPACK with autograd."""
    problem.set_up()

    autograd_res = AutogradExtensions(problem).centered_batch_grad()
    backpack_res = BackpackExtensions(problem).centered_batch_grad()

    rtol, atol = 1e-5, 1e-7
    check_sizes_and_values(autograd_res, backpack_res, rtol=rtol, atol=atol)
    problem.tear_down()


###############################################################################
#                            Keyword argument tests                           #
###############################################################################

FREE_GRAD_BATCH = [True, False]
FREE_GRAD_BATCH_IDS = [f"free_grad_batch={f}" for f in FREE_GRAD_BATCH]

LAYERWISE = [True, False]
LAYERWISE_IDS = [f"layerwise={f}" for f in LAYERWISE]


@pytest.mark.parametrize("free_grad_batch", FREE_GRAD_BATCH, ids=FREE_GRAD_BATCH_IDS)
@pytest.mark.parametrize("layerwise", LAYERWISE, ids=LAYERWISE_IDS)
@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_GramBatchGrad_free_grad_batch_and_layerwise(
    problem, layerwise, free_grad_batch
):
    """Check that ``grad_batch`` and layerwise matrices are deleted if enabled."""
    problem.set_up()

    BackpackExtensions(problem).gram_batch_grad(
        layerwise=layerwise, free_grad_batch=free_grad_batch
    )

    _check_grad_batch_and_layerwise_freed(
        problem,
        free_grad_batch,
        layerwise,
        GramBatchGrad._SAVEFIELD_GRAD_BATCH,
        "gram_grad_batch",
    )

    problem.tear_down()


@pytest.mark.parametrize("free_grad_batch", FREE_GRAD_BATCH, ids=FREE_GRAD_BATCH_IDS)
@pytest.mark.parametrize("layerwise", LAYERWISE, ids=LAYERWISE_IDS)
@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_CenteredGramBatchGrad_free_grad_batch_and_layerwise(
    problem, layerwise, free_grad_batch
):
    """Check that ``grad_batch`` and layerwise matrices are deleted if enabled."""
    problem.set_up()

    BackpackExtensions(problem).centered_gram_batch_grad(
        layerwise=layerwise, free_grad_batch=free_grad_batch
    )
    _check_grad_batch_and_layerwise_freed(
        problem,
        free_grad_batch,
        layerwise,
        CenteredGramBatchGrad._SAVEFIELD_GRAD_BATCH,
        "centered_gram_grad_batch",
    )

    problem.tear_down()


def _check_grad_batch_and_layerwise_freed(
    problem, free_grad_batch, layerwise, savefield_batch_grad, savefield_result
):
    """Verify layerwise results and ``grad_batch`` buffers were deleted if desired."""
    # buffers freed
    for p in problem.model.parameters():
        if free_grad_batch:
            assert not hasattr(p, savefield_batch_grad)
        else:
            assert hasattr(p, savefield_batch_grad)

    # layerwise results freed
    for p in problem.model.parameters():
        if layerwise:
            assert getattr(p, savefield_result) is not None
        else:
            assert getattr(p, savefield_result) is None
