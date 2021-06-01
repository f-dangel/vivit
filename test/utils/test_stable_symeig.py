"""Test of stable symeig implementation"""

import os

import pytest
import torch

from vivit.utils.eig import shift_diag, stable_symeig, symeig_psd

T_1 = torch.diag(torch.Tensor([1.1, 2.2, 9.9]))
T_2 = torch.Tensor([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 2.2]])

HERE = os.path.abspath(__file__)
HEREDIR = os.path.dirname(HERE)
file_name = "tensor_causes_symeig_error.pt"
T_3 = torch.load(os.path.join(HEREDIR, file_name))

TENSOR_LIST = [T_1, T_2, T_3]
TENSOR_LIST_IDS = ["diagonal_tensor", "dense_tensor", "degenerated_tensor"]


@pytest.mark.parametrize("test_tensor", TENSOR_LIST, ids=TENSOR_LIST_IDS)
def test_stable_symeig(test_tensor):
    # TODO Deprecate ``stable_symeig`` in favor of ``symeig_psd``
    error_happened = False
    test_tensor_backup = test_tensor.clone()

    # Try to run symeig
    print("Running symeig")
    try:
        symeig_eigs = test_tensor.symeig(eigenvectors=True)
    except RuntimeError:
        error_happened = True
        print("===== ERROR: symeig did not converge")

    # Run stable_symeig. This must not crash!
    print("Running stable symeig")
    stable_eigs = stable_symeig(test_tensor, eigenvectors=True)

    # Make shure the input is still the same
    assert torch.all(
        test_tensor_backup == test_tensor
    ), "Input tensor changed by stable_symeig"

    # If symeigs crashed, don't compare results
    if not error_happened:
        eigvals_close = torch.allclose(stable_eigs[0], symeig_eigs[0])
        eigvecs_close = torch.allclose(stable_eigs[1], symeig_eigs[1])
        assert eigvals_close, "eigvals not close"
        assert eigvecs_close, "eigvecs not close"
        print("eigvals close?", eigvals_close)
        print("eigvecs close?", eigvecs_close)


TENSORS_SYMEIG_UNSTABLE = [T_3]
TENSOR_SYMEIG_UNSTABLE_IDS = ["degenerated_tensor"]


@pytest.mark.parametrize(
    "tensor", TENSORS_SYMEIG_UNSTABLE, ids=TENSOR_SYMEIG_UNSTABLE_IDS
)
def test_symeig_psd_stability(tensor):
    """Check improved stability of shifted eigenvalue decomposition for PSD matrix.

    ``torch.symeig`` does not converge for ill-conditioned PSD matrices, while
    ``symeig_psd`` with non-zero shift does.

    Args:
        tensor (torch.Tensor): 2d positive semi-definite tensor.
    """
    with pytest.raises(RuntimeError):
        _ = tensor.symeig()

    # no shift is identical to ``torch.symeig``
    with pytest.raises(RuntimeError):
        _ = symeig_psd(tensor)

    # converges with shift
    _ = symeig_psd(tensor, shift=1.0)


@pytest.mark.parametrize("tensor", TENSOR_LIST, ids=TENSOR_LIST_IDS)
def test_symeig_psd_inplace(tensor):
    """Verify input remains *exactly* the same with ``inplace=False``.

    Args:
        tensor (torch.Tensor): 2d positive semi-definite tensor.
    """
    backup = tensor.clone()

    # inplace shift does not modify the original
    shift = 1.0
    _ = symeig_psd(tensor, shift=shift, shift_inplace=False)
    assert tensor.eq(backup).all()

    # inplace shift modifies input up to floating point precision
    atol = 1e-7  # depends on ``shift``
    _ = symeig_psd(tensor, shift=shift, shift_inplace=True)
    assert tensor.eq(backup).all() or torch.allclose(tensor, backup, atol=atol)


TENSORS_SYMEIG_STABLE = [T_1, T_2]
TENSOR_SYMEIG_STABLE_IDS = ["diagonal_tensor", "dense_tensor"]

SHIFTS = [0.0, 1e-1, 1.0, 10.0]
SHIFTS_IDS = [f"shift={shift}" for shift in SHIFTS]

INPLACE = [True, False]
INPLACE_IDS = [f"shift_inplace={inplace}" for inplace in INPLACE]


@pytest.mark.parametrize("shift_inplace", INPLACE, ids=INPLACE_IDS)
@pytest.mark.parametrize("shift", SHIFTS, ids=SHIFTS_IDS)
@pytest.mark.parametrize("tensor", TENSORS_SYMEIG_STABLE, ids=TENSOR_SYMEIG_STABLE_IDS)
def test_compare_symeig_psd_symeig(tensor, shift, shift_inplace):
    """Compare spectra of ``torch.symeig`` with ``symeig_psd``.

    Args:
        tensor (torch.Tensor): 2d positive semi-definite tensor.
        shift (float): The shift applied to the diagonal of ``input``
        shift_inplace (bool): Shift the tensor inplace.
    """
    t = tensor.clone()
    evals, evecs = t.symeig(eigenvectors=True)

    t = tensor.clone()
    psd_evals, psd_evecs = symeig_psd(
        t, eigenvectors=True, shift=shift, shift_inplace=shift_inplace
    )

    rtol, atol = 1e-5, 1e-7
    assert torch.allclose(evals, psd_evals, rtol=rtol, atol=atol)
    assert torch.allclose(evecs, psd_evecs, rtol=rtol, atol=atol)


def test_shift_diag_non_square():
    """Test diagonal shift for rectangular matrices."""
    input = torch.tensor(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 4.0],
        ]
    )
    shift = 0.1

    result = torch.tensor(
        [
            [1.1, 1.0],
            [2.0, 2.1],
            [3.0, 4.0],
        ]
    )

    assert torch.allclose(shift_diag(input, shift), result)
