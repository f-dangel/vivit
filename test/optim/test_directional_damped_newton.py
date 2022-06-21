"""Test ``vivit.optim.directional_damped_newton``."""

from test.implementation.optim_autograd import AutogradOptimExtensions
from test.implementation.optim_backpack import BackpackOptimExtensions
from test.optim.settings import (
    CRITERIA,
    CRITERIA_IDS,
    DAMPING_IDS,
    DAMPINGS,
    IDS_REDUCTION_MEAN,
    PARAM_BLOCKS_FN,
    PARAM_BLOCKS_FN_IDS,
    PROBLEMS_REDUCTION_MEAN,
    SUBSAMPLINGS_GGN,
    SUBSAMPLINGS_GGN_IDS,
    SUBSAMPLINGS_GRAD,
    SUBSAMPLINGS_GRAD_IDS,
)
from test.problem import ExtensionsTestProblem
from test.utils import check_sizes_and_values
from typing import Callable, List, Union

from pytest import mark
from torch import Tensor


@mark.parametrize("param_groups_fn", PARAM_BLOCKS_FN, ids=PARAM_BLOCKS_FN_IDS)
@mark.parametrize("subsampling_ggn", SUBSAMPLINGS_GGN, ids=SUBSAMPLINGS_GGN_IDS)
@mark.parametrize("subsampling_grad", SUBSAMPLINGS_GRAD, ids=SUBSAMPLINGS_GRAD_IDS)
@mark.parametrize("criterion", CRITERIA, ids=CRITERIA_IDS)
@mark.parametrize("damping", DAMPINGS, ids=DAMPING_IDS)
@mark.parametrize("problem", PROBLEMS_REDUCTION_MEAN, ids=IDS_REDUCTION_MEAN)
def test_directional_derivatives(
    problem: ExtensionsTestProblem,
    criterion: Callable[[Tensor], List[int]],
    subsampling_grad: Union[List[int], None],
    subsampling_ggn: Union[List[int], None],
    param_groups_fn: Callable,
    damping: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
):
    """Compare damped Newton steps.

    Args:
        problem: Test case.
        criterion: Filter function to select directions from eigenvalues.
        subsampling_grad: Indices of samples used for gradient sub-sampling.
            ``None`` (equivalent to ``list(range(batch_size))``) uses all mini-batch
            samples to compute directional gradients . Defaults to ``None`` (no
            gradient sub-sampling).
        subsampling_ggn: Indices of samples used for GGN curvature sub-sampling.
            ``None`` (equivalent to ``list(range(batch_size))``) uses all mini-batch
            samples to compute directions and directional curvatures. Defaults to
            ``None`` (no curvature sub-sampling).
        param_groups_fn: Function that creates the `param_groups` from the model's
            named parameters and ``criterion``.
        damping: Function that generates the directional dampings.
    """
    problem.set_up()

    param_groups = param_groups_fn(problem.model.named_parameters(), criterion)
    for group in param_groups:
        group["damping"] = damping

    ag_newton = AutogradOptimExtensions(problem).directional_damped_newton(
        param_groups, subsampling_grad=subsampling_grad, subsampling_ggn=subsampling_ggn
    )

    bp_newton = BackpackOptimExtensions(problem).directional_damped_newton(
        param_groups, subsampling_grad=subsampling_grad, subsampling_ggn=subsampling_ggn
    )

    check_sizes_and_values(ag_newton, bp_newton, rtol=1e-5, atol=1e-5)

    problem.tear_down()
