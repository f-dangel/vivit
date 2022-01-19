"""Test ``vivit.linalg.eigvalsh``."""

from test.implementation.linalg_autograd import AutogradLinalgExtensions
from test.implementation.linalg_backpack import BackpackLinalgExtensions
from test.linalg.settings import (
    IDS,
    PARAM_GROUPS_FN,
    PARAM_GROUPS_FN_IDS,
    PROBLEMS,
    SUBSAMPLINGS,
    SUBSAMPLINGS_IDS,
    keep_all,
)
from test.problem import ExtensionsTestProblem
from test.utils import check_sizes_and_values
from typing import Any, Callable, Dict, Iterator, List, Union

from pytest import mark
from torch import Tensor


@mark.parametrize("param_groups_fn", PARAM_GROUPS_FN, ids=PARAM_GROUPS_FN_IDS)
@mark.parametrize("subsampling", SUBSAMPLINGS, ids=SUBSAMPLINGS_IDS)
@mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_ggn_eigvalsh(
    problem: ExtensionsTestProblem,
    subsampling: Union[List[int], None],
    param_groups_fn: Callable[[Iterator[Tensor]], List[Dict[str, Any]]],
):
    """Compare ``V Vᵀ`` (BackPACK) eigenvalues with ``G`` (autograd).

    Args:
        problem: Test case.
        subsampling: Indices of samples used for the computation. ``None`` uses the
            entire mini-batch.
        param_groups_fn: Function that creates parameter groups.
    """
    problem.set_up()

    param_groups = param_groups_fn(problem.model.named_parameters())
    for group in param_groups:
        group["criterion"] = keep_all

    backpack_result = BackpackLinalgExtensions(problem).eigvalsh_ggn(
        param_groups, subsampling
    )
    autograd_result = AutogradLinalgExtensions(problem).eigvalsh_ggn(
        param_groups, subsampling
    )

    for group_id in backpack_result.keys():
        backpack_evals = backpack_result[group_id]
        autograd_evals = autograd_result[group_id]

        num_evals = min(autograd_evals.numel(), backpack_evals.numel())
        autograd_evals = autograd_evals[-num_evals:]
        backpack_evals = backpack_evals[-num_evals:]

        rtol, atol = 1e-4, 5e-6
        check_sizes_and_values(backpack_evals, autograd_evals, rtol=rtol, atol=atol)

    problem.tear_down()
