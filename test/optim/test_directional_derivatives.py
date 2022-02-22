"""Test ``vivit.optim.directional_derivatives``."""

from test.implementation.optim_autograd import AutogradOptimExtensions
from test.implementation.optim_backpack import BackpackOptimExtensions
from test.optim.settings import (
    CRITERIA,
    CRITERIA_IDS,
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

from pytest import mark, raises
from torch import Tensor

from vivit import DirectionalDerivativesComputation


@mark.parametrize("param_groups_fn", PARAM_BLOCKS_FN, ids=PARAM_BLOCKS_FN_IDS)
@mark.parametrize("subsampling_ggn", SUBSAMPLINGS_GGN, ids=SUBSAMPLINGS_GGN_IDS)
@mark.parametrize("subsampling_grad", SUBSAMPLINGS_GRAD, ids=SUBSAMPLINGS_GRAD_IDS)
@mark.parametrize("criterion", CRITERIA, ids=CRITERIA_IDS)
@mark.parametrize("problem", PROBLEMS_REDUCTION_MEAN, ids=IDS_REDUCTION_MEAN)
def test_directional_derivatives(
    problem: ExtensionsTestProblem,
    criterion: Callable[[Tensor], List[int]],
    subsampling_grad: Union[List[int], None],
    subsampling_ggn: Union[List[int], None],
    param_groups_fn: Callable,
):
    """Compare 1ˢᵗ- and 2ⁿᵈ-order directional derivatives along GGN eigenvectors.

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
    """
    problem.set_up()

    param_groups = param_groups_fn(problem.model.named_parameters(), criterion)

    ag_gammas, ag_lambdas = AutogradOptimExtensions(problem).directional_derivatives(
        param_groups, subsampling_grad=subsampling_grad, subsampling_ggn=subsampling_ggn
    )
    bp_gammas, bp_lambdas = BackpackOptimExtensions(problem).directional_derivatives(
        param_groups, subsampling_grad=subsampling_grad, subsampling_ggn=subsampling_ggn
    )

    # directions can vary in sign, leading to same magnitude but opposite sign
    ag_abs_gammas = [g.abs() for g in ag_gammas]
    bp_abs_gammas = [g.abs() for g in bp_gammas]
    check_sizes_and_values(ag_abs_gammas, bp_abs_gammas, rtol=1e-5, atol=1e-4)
    check_sizes_and_values(ag_lambdas, bp_lambdas, rtol=1e-5, atol=1e-5)

    problem.tear_down()


def test_get_result():
    """Test retrieving results for an unknown group fails."""
    group = {"params": []}

    with raises(KeyError):
        DirectionalDerivativesComputation().get_result(group)
