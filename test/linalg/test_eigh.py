"""Test ``vivit.linalg.eigh``."""

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
    keep_nonzero,
)
from test.problem import ExtensionsTestProblem
from test.utils import check_sizes_and_values
from typing import Any, Callable, Dict, Iterator, List, Union

from pytest import mark
from torch import Tensor, einsum, eye


@mark.parametrize("param_groups_fn", PARAM_GROUPS_FN, ids=PARAM_GROUPS_FN_IDS)
@mark.parametrize("subsampling", SUBSAMPLINGS, ids=SUBSAMPLINGS_IDS)
@mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_ggn_eigh_eigenvalues(
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

    backpack_eigh = BackpackLinalgExtensions(problem).eigh_ggn(
        param_groups, subsampling
    )
    autograd_eigh = AutogradLinalgExtensions(problem).eigh_ggn(
        param_groups, subsampling
    )

    for group_id in backpack_eigh[0].keys():
        backpack_evals = backpack_eigh[0][group_id]
        autograd_evals = autograd_eigh[0][group_id]

        num_evals = min(autograd_evals.numel(), backpack_evals.numel())
        autograd_evals = autograd_evals[-num_evals:]
        backpack_evals = backpack_evals[-num_evals:]

        rtol, atol = 1e-4, 5e-6
        check_sizes_and_values(backpack_evals, autograd_evals, rtol=rtol, atol=atol)

    problem.tear_down()


@mark.parametrize("param_groups_fn", PARAM_GROUPS_FN, ids=PARAM_GROUPS_FN_IDS)
@mark.parametrize("subsampling", SUBSAMPLINGS, ids=SUBSAMPLINGS_IDS)
@mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_ggn_eigh_eigenvectors(
    problem: ExtensionsTestProblem,
    subsampling: Union[List[int], None],
    param_groups_fn: Callable[[Iterator[Tensor]], List[Dict[str, Any]]],
):
    """Compare ``V Vᵀ`` (BackPACK) eigenvectors with ``G`` (autograd).

    Checks the following properties:

    - Eigenvector is scaled by eigenvalue during multiplication by the GGN.
    - Eigenvectors are orthonormal.
    - Eigenvectors of ``EighComputation`` are multiples of ``autograd``
      eigenvectors, up to a different sign.

    Note:
        For eigenvalues close to zero, the transformation from Gram space into
        parameter space through ``V`` will not succeed and lead to a zero vector (or a
        vector of small magnitude). The succeeding normalization in ``EighComputation``
        will thus be numerically unstable and eventually break the orthonormality of
        eigenvectors. That's why these eigenvalues are filtered with ``keep_nonzero``.

    Args:
        problem: Test case.
        subsampling: Indices of samples used for the computation. ``None`` uses the
            entire mini-batch.
        param_groups_fn: Function that creates parameter groups.
    """
    problem.set_up()

    param_groups = param_groups_fn(problem.model.named_parameters())
    for group in param_groups:
        group["criterion"] = keep_nonzero

    backpack_eigh = BackpackLinalgExtensions(problem).eigh_ggn(
        param_groups, subsampling
    )
    autograd_eigh = AutogradLinalgExtensions(problem).eigh_ggn(
        param_groups, subsampling
    )

    for group_id in backpack_eigh[0].keys():
        (group,) = [g for g in param_groups if id(g) == group_id]

        # Pre-processing
        backpack_evals = backpack_eigh[0][group_id]
        backpack_evecs = backpack_eigh[1][group_id]
        autograd_evals = autograd_eigh[0][group_id]
        autograd_evecs = autograd_eigh[1][group_id]

        num_evals = min(autograd_evals.numel(), backpack_evals.numel())
        backpack_evecs = [v[-num_evals:] for v in backpack_evecs]
        backpack_evals = backpack_evals[-num_evals:]
        autograd_evecs = [v[-num_evals:] for v in autograd_evecs]
        autograd_evals = autograd_evals[-num_evals:]

        # Check scaling property
        ggn_backpack_evecs = AutogradLinalgExtensions(
            problem
        ).ggn_mat_prod_from_param_list(
            backpack_evecs, group["params"], subsampling=subsampling
        )
        scaled_backpack_evecs = [
            einsum("i,i...->i...", backpack_evals, evec) for evec in backpack_evecs
        ]
        rtol, atol = 5e-4, 1e-5
        check_sizes_and_values(
            ggn_backpack_evecs, scaled_backpack_evecs, rtol=rtol, atol=atol
        )

        # Orthogonality check
        identity = eye(num_evals).to(problem.device)

        overlap_backpack = pairwise_scalar_products(backpack_evecs, backpack_evecs)
        overlap_autograd = pairwise_scalar_products(autograd_evecs, autograd_evecs)

        rtol, atol = 1e-3, 2e-4
        check_sizes_and_values(identity, overlap_autograd, rtol=rtol, atol=atol)
        check_sizes_and_values(identity, overlap_backpack, rtol=rtol, atol=atol)

        # Compare eigenvectors
        autograd_evecs_abs = [v.abs() for v in autograd_evecs]
        backpack_evecs_abs = [v.abs() for v in backpack_evecs]

        rtol, atol = 2e-2, 2e-3
        check_sizes_and_values(
            autograd_evecs_abs, backpack_evecs_abs, rtol=rtol, atol=atol
        )

    problem.tear_down()


def pairwise_scalar_products(tensors: List[Tensor], others: List[Tensor]) -> Tensor:
    """Compute pairwise scalar products of two vector sets in parameter list format.

    Args:
        tensors: First set of vectors in parameter list format, stacked along leading
            dimension.
        others: Second set of vectors in parameter list format, stacked along leading
            dimension.

    Returns:
        Rectangular matrix containing the pairwise dot products.
    """
    assert len(tensors) == len(others)
    for tensor, other in zip(tensors, others):
        assert tensor.shape[1:] == other.shape[1:]

    return sum(
        einsum("i...,j...->ij", tensor, other) for tensor, other in zip(tensors, others)
    )
