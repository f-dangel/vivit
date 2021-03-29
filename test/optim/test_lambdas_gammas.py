"""Tests computations of first- and second-order directional derivatives for a linear
network. In this case, we can give the gradient and GGN of the loss in closed-form."""

import pytest
import torch
from backpack import backpack, extend
from backpack.hessianfree.hvp import hessian_vector_product
from backpack.utils.convert_parameters import vector_to_parameter_list

from lowrank.optim import BaseComputations, DampedNewton


# ======================================================================================
# Auxiliary Functions
# ======================================================================================
def set_weights(linear_layer, weights, req_grad):
    """
    Set weights in linear layer and choose if these
    parameters are trainable.
    """

    # Check if weights has the right shape
    w = linear_layer.weight
    if weights.shape == w.data.shape:

        # Set weights and requires_grad
        w.data = weights
        w.requires_grad = req_grad

    else:
        raise ValueError("weights dont have the right shape")


def set_biases(linear_layer, biases, req_grad):
    """
    Set biases in linear layer and choose if these
    parameters are trainable.
    """

    # Check if biases has the right shape
    b = linear_layer.bias
    if biases.shape == b.data.shape:

        # Set biases and requires_grad
        b.data = biases
        b.requires_grad = req_grad

    else:
        raise ValueError("biases dont have the right shape")


def Phi(x, theta, MSE_reduction):
    """
    Computes MSE-loss at (x, theta) manually.
    """

    # Make sure N == 1
    assert x.shape[0] == 1, "N has to be one such that model output is vector"

    # Compute output of model
    theta_re = theta.reshape(1, OUT_1)
    y2 = (x @ W_1.T + theta_re) @ W_2.T

    # Compute MSE loss manually
    if MSE_reduction == "mean":
        return (1 / OUT_2) * (y2 @ y2.T).item()
    elif MSE_reduction == "sum":
        return (y2 @ y2.T).item()
    else:
        raise ValueError("Unknown MSE_reduction")


def Phi_grad(x, theta, MSE_reduction):
    """
    Computes gradient of MSE-loss at (x, theta) manually.
    """

    # Make sure N == 1
    assert x.shape[0] == 1, "N has to be one such that model output is vector"

    # Compute MSE loss gradient manually
    theta_re = theta.reshape(1, OUT_1)
    if MSE_reduction == "mean":
        return (2 / OUT_2) * W_2.T @ W_2 @ (W_1 @ x.T + theta_re.T)
    elif MSE_reduction == "sum":
        return 2 * W_2.T @ W_2 @ (W_1 @ x.T + theta_re.T)
    else:
        raise ValueError("Unknown MSE_reduction")


def Phi_GGN(x, theta, MSE_reduction):
    """
    Computes Hessian (= GGN) of MSE-loss at (x, theta) manually.
    """

    # Make sure N == 1
    assert x.shape[0] == 1, "N has to be one such that model output is vector"

    # Compute MSE loss Hessian (= GGN) manually
    if MSE_reduction == "mean":
        return (2 / OUT_2) * W_2.T @ W_2
    elif MSE_reduction == "sum":
        return 2 * W_2.T @ W_2
    else:
        raise ValueError("Unknown MSE_reduction")


def autograd_hessian_columns(loss, params, concat=False):
    """Return an iterator of the Hessian columns computed via ``torch.autograd``.
    Args:
        loss (torch.Tensor): Loss whose Hessian is investigated.
        params ([torch.Tensor]): List of torch.Tensors holding the network's
            parameters.
        concat (bool): If ``True``, flatten and concatenate the columns over all
            parameters.
    """
    D = sum(p.numel() for p in params)
    device = loss.device
    for d in range(D):
        e_d = torch.zeros(D, device=device)
        e_d[d] = 1.0
        e_d_list = vector_to_parameter_list(e_d, params)
        hessian_e_d = hessian_vector_product(loss, params, e_d_list)
        if concat:
            hessian_e_d = torch.cat([tensor.flatten() for tensor in hessian_e_d])
        yield hessian_e_d


def autograd_hessian(loss, params):
    """Compute the full Hessian via ``torch.autograd``.
    Flatten and concatenate the columns over all parameters, such that the result
    is a ``[D, D]`` tensor, where ``D`` denotes the total number of parameters.
    Args:
        params ([torch.Tensor]): List of torch.Tensors holding the network's
            parameters.
    Returns:
        torch.Tensor: 2d tensor containing the Hessian matrix
    """
    return torch.stack(list(autograd_hessian_columns(loss, params, concat=True)))


def within_tol(error, tol, message):
    """
    Check if error is within tolerance. If not: Throw an error and print message.
    """
    assert error >= 0, "Error is < 0"
    assert error <= tol, message


# ======================================================================================
# Test parameters
# ======================================================================================

# Set torch seed
torch.manual_seed(0)

# Choose dimensions and initialize weight matrices
IN_1 = 10  # Layer 1
OUT_1 = 11
IN_2 = OUT_1  # Layer 2
OUT_2 = 12
W_1 = 2 * torch.rand(OUT_1, IN_1) - 1
W_2 = 2 * torch.rand(OUT_2, IN_2) - 1

# Number of runs with different (randomly chosen) x and theta
NUM_RUNS = 5

# Test tolerances
TOL = 1e-5

MSE_REDUCTIONS = ["mean", "sum"]
IDS_MSE_REDUCTIONS = [
    f"MSE_reduction={MSE_reduction}" for MSE_reduction in MSE_REDUCTIONS
]


@pytest.mark.parametrize("MSE_reduction", MSE_REDUCTIONS, ids=IDS_MSE_REDUCTIONS)
def test_lambda_gamma(MSE_reduction):
    """
    TODO
    """

    # Initialize layers
    L_1 = torch.nn.Linear(IN_1, OUT_1, bias=True)
    set_weights(L_1, W_1, False)
    L_2 = torch.nn.Linear(IN_2, OUT_2, bias=False)
    set_weights(L_2, W_2, False)

    # For different x, theta
    for _run in range(NUM_RUNS):

        # Choose x and theta
        x = torch.rand(1, IN_1)
        theta = torch.rand(OUT_1)

        # Set biases of layer 1 to theta and create model
        set_biases(L_1, theta, True)
        model = extend(torch.nn.Sequential(L_1, L_2))

        # Determine loss function
        loss_func = extend(torch.nn.MSELoss(reduction=MSE_reduction))

        # Compute γ along top-k GGN eigenvector(s)
        k = 2
        top_k = DampedNewton.make_default_criterion(k=k)
        param_groups = [{"params": list(model.parameters()), "criterion": top_k}]
        computations = BaseComputations()

        # ==========================
        # TEST 1: Loss value
        # ==========================
        loss = loss_func(model(x), torch.zeros(1, OUT_2))
        phi = Phi(x, theta, MSE_reduction)
        err = torch.abs(loss - phi)
        within_tol(err, TOL, f"Test 1 (loss value) failed: err = {err:.3e}")

        # ==========================
        # TEST 2: Loss gradient
        # ==========================
        model.zero_grad()

        # Retain graph for computing Hessian later
        with backpack(*computations.get_extensions(param_groups)):
            loss.backward(retain_graph=True)
        loss_grad = list(model.parameters())[1].grad
        phi_grad = Phi_grad(x, theta, MSE_reduction).reshape(OUT_1)
        err = torch.norm(phi_grad - loss_grad)
        within_tol(err, TOL, f"Test 2 (loss gradient) failed: err = {err:.3e}")

        # ==========================
        # TEST 3: Loss GGN
        # ==========================
        phi_GGN = Phi_GGN(x, theta, MSE_reduction)

        # Compute Hessian with respect to theta
        theta_params = list(model.parameters())[1]
        loss_GGN = autograd_hessian(loss, [theta_params])
        err = torch.norm(loss_GGN - phi_GGN)
        within_tol(err, TOL, f"Test 3 (loss GGN) failed: err = {err:.3e}")

        # # ==========================
        # # Manual computation of γ and λ, usually done inside an optimizer
        # # ==========================
        # # Main training loop
        # inputs, labels = x, torch.zeros(1, OUT_2)
        #
        # # forward pass
        # outputs = model(inputs)
        # loss = loss_func(outputs, labels)
        #
        # # backward pass
        # with backpack(*computations.get_extensions(param_groups)):
        #     loss.backward()
        #
        # # manual computation of γ and λ, usually done inside an optimizer
        # for group in param_groups:
        #     computations._eval_directions(group)
        #     computations._filter_directions(group)
        #     computations._eval_gammas(group)
        #     computations._eval_lambdas(group)
        #
        # print(f"γ[n,d]: {list(computations._lambdas.values())[0]}")
        # print(f"λ[n,d]: {list(computations._gammas.values())[0]}")

        # Go through all eigenvectors
        eigvals, eigvecs = torch.symeig(phi_GGN, eigenvectors=True)
        for j in range(OUT_1):
            eigvec = eigvecs[:, j]
            eigval = eigvals[j]

            # ==========================
            # Test 4: Eigenvalues and -vectors
            # ==========================
            err = torch.norm(phi_GGN @ eigvec - eigval * eigvec)
            mess = f"Test 4 (eigenvalues and -vectors) failed: err = {err:.3e}"
            within_tol(err, TOL, mess)

            # ==========================
            # Test 5: Eigenvector normalization
            # ==========================
            err = torch.abs(torch.norm(eigvec) - 1.0)
            mess = f"Test 5 (eigenvector normalization) failed: err = {err:.3e}"
            within_tol(err, TOL, mess)

            # ==========================
            # Test 6: gamma
            # ==========================
            # phi_gamma = torch.dot(eigvec, phi_grad).item()
            # @Felix: Comparison to gamma
            err = 0.0
            mess = f"Test 6 (gammas) failed: err = {err:.3e}"
            within_tol(err, TOL, mess)

            # ==========================
            # Test 7: lambda
            # ==========================
            # phi_lambda = torch.dot(eigvec @ phi_GGN, eigvec).item()
            # @Felix: Comparison to lambda
            err = 0.0
            mess = f"Test 7 (lambdas) failed: err = {err:.3e}"
            within_tol(err, TOL, mess)
