"""Tests for ``vivit.extensions.secondorder.sqrt_ggn.gram_sqrt_ggn.py``."""

from test.implementation.autograd import AutogradExtensions
from test.implementation.backpack import BackpackExtensions
from test.problem import ExtensionsTestProblem, make_test_problems
from test.settings import SETTINGS
from test.utils import check_sizes_and_values
from typing import List, Union

from pytest import mark
from torch import einsum, rand_like, stack

PROBLEMS = make_test_problems(SETTINGS)
IDS = [problem.make_id() for problem in PROBLEMS]

SUBSAMPLINGS = [None, [0, 0, 1, 0, 1]]
SUBSAMPLINGS_IDS = [f"subsampling={sub}" for sub in SUBSAMPLINGS]


@mark.parametrize("subsampling", SUBSAMPLINGS, ids=SUBSAMPLINGS_IDS)
@mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_vivit_ggn_exact_mat_prod(
    problem: ExtensionsTestProblem,
    subsampling: Union[List[int], None],
    num_vecs: int = 3,
):
    """Compare ``V Vᵀ v`` (BackPACK) with ``G v`` (autograd).

    Tests multiplication by ``V`` and ``Vᵀ``.

    Args:
        problem: Test case.
        subsampling: Indices of samples to use for the computation.
        num_vecs: Number of GGN-vector products. Default: ``3``.
    """
    problem.set_up()

    mat = [
        stack([rand_like(p) for _ in range(num_vecs)])
        for p in problem.model.parameters()
    ]

    backpack_res = BackpackExtensions(problem).vivit_ggn_mat_prod(
        mat, subsampling=subsampling
    )
    autograd_res = AutogradExtensions(problem).ggn_mat_prod(
        mat, subsampling=subsampling
    )

    rtol, atol = 1e-5, 5e-7
    check_sizes_and_values(backpack_res, autograd_res, rtol=rtol, atol=atol)
    problem.tear_down()


@mark.parametrize("subsampling", SUBSAMPLINGS, ids=SUBSAMPLINGS_IDS)
@mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_vivit_ggn_exact_eigh(
    problem: ExtensionsTestProblem, subsampling: Union[List[int], None]
):
    """Verify that eigenvectors computed via ``Vᵀ V`` are eigenvectors of ``G``.

    Tests multiplication by ``V`` and Gram matrix evaluation.

    Args:
        problem: Test case.
        subsampling: Indices of samples to use for the computation.
    """
    problem.set_up()

    evals, evecs = BackpackExtensions(problem).vivit_ggn_eigh(subsampling=subsampling)
    G_evecs = AutogradExtensions(problem).ggn_mat_prod(evecs, subsampling=subsampling)
    evals_evecs = [einsum("i,i...->i...", evals, vecs) for vecs in evecs]

    rtol, atol = 1e-5, 5e-6
    check_sizes_and_values(evals_evecs, G_evecs, rtol=rtol, atol=atol)
    problem.tear_down()


@mark.parametrize("subsampling", SUBSAMPLINGS, ids=SUBSAMPLINGS_IDS)
@mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_vivit_ggn_mc_mat_prod(
    problem: ExtensionsTestProblem,
    subsampling: Union[List[int], None],
    num_vecs: int = 3,
):
    """Compare MC-sampled ``V Vᵀ v`` (BackPACK) with exact ``G v`` (autograd).

    Tests multiplication by ``V`` and ``Vᵀ`` for ``ViViTGGNMC``.

    Args:
        problem: Test case.
        subsampling: Indices of samples to use for the computation.
        num_vecs: Number of GGN-vector products. Default: ``3``.
    """
    problem.set_up()

    mat = [
        stack([rand_like(p) for _ in range(num_vecs)])
        for p in problem.model.parameters()
    ]

    mc_samples = 50000
    chunks = 50
    backpack_res = BackpackExtensions(problem).vivit_ggn_mc_mat_prod_chunk(
        mat, mc_samples, chunks=chunks, subsampling=subsampling
    )
    autograd_res = AutogradExtensions(problem).ggn_mat_prod(
        mat, subsampling=subsampling
    )

    rtol, atol = 1e-1, 1e-3
    check_sizes_and_values(backpack_res, autograd_res, rtol=rtol, atol=atol)
    problem.tear_down()
