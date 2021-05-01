"""Test of stable symeig implementation"""

import os

import pytest
import torch

from lowrank.utils.eig import stable_symeig

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
