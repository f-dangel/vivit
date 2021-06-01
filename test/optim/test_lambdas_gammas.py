"""In the existing test for gamma and lamdba (test_gammas.py and test_lambdas.py), the
vivit-computations are compared to autograd. There might be the chance that there is
the "same" mistake in both versions. So, here is another apporach to test the gammas and
lambdas: We use a very simple linear network. In this case, we can give the loss, its
gradient and GGN in closed-form. We use these closed-form expressions to compute
reference lambdas and gammas, that we can compare the vivit-computations with.

The following tests are performed:
- TEST 1 (Loss value): We compare the loss evaluated on the actual model with the loss
         that we derived theoretically
- TEST 2 (Loss gradient): We compare the loss gradient computed by pytorch with the loss
         gradient that we derived theoretically
- TEST 3 (Loss GGN): We compare the loss GGN computed by autograd (see section
         "Auxiliary Functions (3)") with the loss GGN that we derived theoretically
- TEST 4, 5 (gammas and lambdas): We compute the lambdas and gammas with the vivit-
         utilities. As a comparison, we also compute the theoretically derived GGN, its
         eigenvectors and compute the lambdas and gammas "manually".
- TEST 6 (Newton step): Finally, we compare the Newton step computed by vivit with a
         "manual" computation.
"""

from test.optim.settings import make_criterion
from test.utils import check_sizes_and_values

import pytest
import torch
from backpack import backpack, extend
from backpack.hessianfree.hvp import hessian_vector_product
from backpack.utils.convert_parameters import vector_to_parameter_list

from vivit.optim.computations import BaseComputations
from vivit.optim.damping import ConstantDamping

# ======================================================================================
# Auxiliary Functions (1)
# Set weights and biases for linear layer and choose if these parameters are trainable
# ======================================================================================


def set_weights(linear_layer, weights, req_grad):
    """
    Set weights in linear layer and choose if these parameters are trainable.
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
    Set biases in linear layer and choose if these parameters are trainable.
    """

    # Check if biases has the right shape
    b = linear_layer.bias
    if biases.shape == b.data.shape:

        # Set biases and requires_grad
        b.data = biases
        b.requires_grad = req_grad

    else:
        raise ValueError("biases dont have the right shape")


# ======================================================================================
# Auxiliary Functions (2)
# The MSE-loss corresponds to Phi. Here, we define functions for evaluating Phi, its
# sample gadients and GGNs.
# ======================================================================================


def Phi(x, theta, MSE_reduction, W_1, W_2):
    """
    Computes MSE-loss at (x, theta) manually.
    """

    # Make sure N == 1
    assert x.shape[0] == 1, "N has to be one such that model output is a vector"

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


def Phi_batch(X, theta, MSE_reduction, W_1, W_2):
    """
    Computes MSE-loss for batch X containing N samples (rows) by averaging or summing
    the individual sample losses.
    """

    N = X.shape[0]

    # Accumulate loss over all batches
    loss_batch = 0.0
    for n in range(N):
        x = X[n, :].reshape(1, IN_1)
        loss_batch += Phi(x, theta, MSE_reduction, W_1, W_2)

    # Return accumulated loss or return average
    if MSE_reduction == "mean":
        return (1 / N) * loss_batch
    elif MSE_reduction == "sum":
        return loss_batch
    else:
        raise ValueError("Unknown MSE_reduction")


def Phi_grad(x, theta, MSE_reduction, W_1, W_2):
    """
    Computes gradient of MSE-loss at (x, theta) manually.
    """

    # Make sure N == 1
    assert x.shape[0] == 1, "N has to be one such that model output is a vector"

    # Compute MSE loss gradient manually
    theta_re = theta.reshape(1, OUT_1)
    grad = 2 * (W_2.T @ W_2 @ (W_1 @ x.T + theta_re.T)).reshape(OUT_1)
    if MSE_reduction == "mean":
        return grad / OUT_2
    elif MSE_reduction == "sum":
        return grad
    else:
        raise ValueError("Unknown MSE_reduction")


def Phi_grads_list(X, theta, MSE_reduction, W_1, W_2):
    """
    Computes MSE-loss gradients for batch X containing N samples (rows) and retuns them
    as a list
    """

    N = X.shape[0]

    grads_list = []
    for n in range(N):
        x = X[n, :].reshape(1, IN_1)
        grads_list.append(Phi_grad(x, theta, MSE_reduction, W_1, W_2))

    return grads_list


def Phi_GGN(x, theta, MSE_reduction, W_1, W_2):
    """
    Computes Hessian (= GGN) of MSE-loss at (x, theta) manually.
    """

    # Make sure N == 1
    assert x.shape[0] == 1, "N has to be one such that model output is a vector"

    # Compute MSE loss Hessian (= GGN) manually
    GGN = 2 * W_2.T @ W_2
    if MSE_reduction == "mean":
        return GGN / OUT_2
    elif MSE_reduction == "sum":
        return GGN
    else:
        raise ValueError("Unknown MSE_reduction")


def Phi_GGNs_list(X, theta, MSE_reduction, W_1, W_2):
    """
    Computes MSE-loss GGNs for batch X containing N samples (rows) and retuns them
    as a list
    """

    N = X.shape[0]

    GGNs_list = []
    for n in range(N):
        x = X[n, :].reshape(1, IN_1)
        GGNs_list.append(Phi_GGN(x, theta, MSE_reduction, W_1, W_2))

    return GGNs_list


def reduce_list(the_list, reduction):
    """
    Auxiliary function that computes the sum or mean over all list entries. The list
    entries are assumed to be torch.Tensors.
    """

    # Check that list entries are torch.Tensors
    if not torch.is_tensor(the_list[0]):
        raise ValueError("List entries have to be torch.Tensors")

    # Sum over list entries
    sum_over_list_entries = torch.zeros_like(the_list[0])
    for i in range(len(the_list)):
        sum_over_list_entries += the_list[i]

    if reduction == "mean":
        return sum_over_list_entries / len(the_list)
    elif reduction == "sum":
        return sum_over_list_entries
    else:
        raise ValueError("Unknown reduction")


# ======================================================================================
# Auxiliary Functions (3)
# Utilities for computing the Hessian for a given model. We will use this as a
# comparison to Phi_GGN
# ======================================================================================


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


# # ====================================================================================
# # Auxiliary Functions (4)
# # Check if results are within given tolerances. This is basically a copy from
# # test.utils
# # ====================================================================================
#
# atol = 1e-8
# rtol = 1e-5
#
#
# def report_nonclose_values(x, y, atol=atol, rtol=rtol):
#     x_numpy = x.data.cpu().numpy().flatten()
#     y_numpy = y.data.cpu().numpy().flatten()
#
#     close = np.isclose(x_numpy, y_numpy, atol=atol, rtol=rtol)
#     where_not_close = np.argwhere(np.logical_not(close))
#     for idx in where_not_close:
#         x, y = x_numpy[idx], y_numpy[idx]
#         print("{} versus {}. Ratio of {}".format(x, y, y / x))
#
#
# def check_sizes_and_values(*plists, atol=atol, rtol=rtol):
#     check_sizes(*plists)
#     list1, list2 = plists
#     check_values(list1, list2, atol=atol, rtol=rtol)
#
#
# def check_sizes(*plists):
#     for i in range(len(plists) - 1):
#         assert len(plists[i]) == len(plists[i + 1])
#
#     for params in zip(*plists):
#         for i in range(len(params) - 1):
#             assert params[i].size() == params[i + 1].size()
#
#
# def check_values(list1, list2, atol=atol, rtol=rtol):
#     for i, (g1, g2) in enumerate(zip(list1, list2)):
#         report_nonclose_values(g1, g2, atol=atol, rtol=rtol)
#         assert torch.allclose(g1, g2, atol=atol, rtol=rtol)


# ======================================================================================
# Define Test Parameters
# ======================================================================================

# Test tolerances
ATOL = 1e-5
RTOL = 1e-4

# Choose dimensions and
N = 8
IN_1 = 10  # Layer 1
OUT_1 = 11
IN_2 = OUT_1  # Layer 2
OUT_2 = IN_2
if OUT_2 < IN_2:
    print("Warning: The GGN won't have full rank")


# ======================================================================================
# Run Tests
# ======================================================================================

# MSE-reductions
MSE_REDUCTIONS = ["mean"]
IDS_MSE_REDUCTIONS = [
    f"MSE_reduction={MSE_reduction}" for MSE_reduction in MSE_REDUCTIONS
]

# Dampings
DAMPINGS = [1.0, 2.5]
IDS_DAMPINGS = [f"Damping={delta}" for delta in DAMPINGS]

# Seed values
SEED_VALS = [0, 1, 42]
IDS_SEED_VALS = [f"SeedVal={seed_val}" for seed_val in SEED_VALS]


@pytest.mark.parametrize("MSE_reduction", MSE_REDUCTIONS, ids=IDS_MSE_REDUCTIONS)
@pytest.mark.parametrize("delta", DAMPINGS, ids=IDS_DAMPINGS)
@pytest.mark.parametrize("seed_val", SEED_VALS, ids=IDS_SEED_VALS)
def test_lambda_gamma(MSE_reduction, delta, seed_val):

    # Set torch seed
    torch.manual_seed(seed_val)

    # Initialize weight matrices, theta and X
    W_1 = 2 * torch.rand(OUT_1, IN_1) - 1
    W_2 = 2 * torch.rand(OUT_2, IN_2) - 1
    theta = torch.rand(OUT_1)
    X = torch.rand(N, IN_1)

    # Initialize layers, create model and loss function
    L_1 = torch.nn.Linear(IN_1, OUT_1, bias=True)
    L_2 = torch.nn.Linear(IN_2, OUT_2, bias=False)
    set_weights(L_1, W_1, False)
    set_biases(L_1, theta, True)
    set_weights(L_2, W_2, False)
    model = extend(torch.nn.Sequential(L_1, L_2))
    loss_func = extend(torch.nn.MSELoss(reduction=MSE_reduction))

    # ==========================
    # TEST 1: Loss value
    # ==========================
    phi = torch.Tensor([Phi_batch(X, theta, MSE_reduction, W_1, W_2)]).reshape(1, 1)
    loss = loss_func(model(X), torch.zeros(N, OUT_2)).reshape(1, 1)
    check_sizes_and_values(loss, phi, atol=ATOL, rtol=RTOL)

    # ==========================
    # TEST 2: Loss gradient
    # ==========================
    phi_grads_list = Phi_grads_list(X, theta, MSE_reduction, W_1, W_2)
    phi_batch_grad = reduce_list(phi_grads_list, MSE_reduction)
    model.zero_grad()
    loss.backward(retain_graph=True)  # Retain graph for computing Hessian later
    loss_grad = list(model.parameters())[1].grad
    check_sizes_and_values(loss_grad, phi_batch_grad, atol=ATOL, rtol=RTOL)

    # ==========================
    # TEST 3: Loss GGN
    # ==========================
    phi_GGNs_list = Phi_GGNs_list(X, theta, MSE_reduction, W_1, W_2)
    phi_batch_GGN = reduce_list(phi_GGNs_list, MSE_reduction)
    theta_params = list(model.parameters())[1]
    loss_GGN = autograd_hessian(loss, [theta_params])
    check_sizes_and_values(loss_GGN, phi_batch_GGN, atol=ATOL, rtol=RTOL)

    # Go through all eigenvectors and compute lambdas and gammas
    eigvals, eigvecs = torch.symeig(phi_batch_GGN, eigenvectors=True)
    phi_lambdas = torch.zeros(N, OUT_1)
    phi_gammas = torch.zeros(N, OUT_1)
    for i in range(N):
        phi_grad = phi_grads_list[i]
        phi_GGN = phi_GGNs_list[i]
        for j in range(OUT_1):
            eigvec = eigvecs[:, j]

            # Compute gammas and lambdas
            phi_gammas[i, j] = torch.dot(eigvec, phi_grad).item()
            phi_lambdas[i, j] = torch.dot(eigvec @ phi_GGN, eigvec).item()

    # Now, compute lambdas and gammas with vivit-utilities
    top_k = make_criterion(k=OUT_1)
    param_groups = [
        {
            "params": [p for p in model.parameters() if p.requires_grad],
            "criterion": top_k,
        }
    ]
    computations = BaseComputations()
    savefield = "test_newton_step"
    const_damping = ConstantDamping(delta)

    # Forward and backward pass
    loss = loss_func(model(X), torch.zeros(N, OUT_2))
    with backpack(
        *computations.get_extensions(param_groups),
        extension_hook=computations.get_extension_hook(
            param_groups, const_damping, savefield
        ),
    ):
        loss.backward()

    # ==========================
    # Test 4: gammas
    # ==========================
    gammas_abs = torch.abs(list(computations._gram_computation._gammas.values())[0])
    check_sizes_and_values(gammas_abs, torch.abs(phi_gammas), atol=ATOL, rtol=RTOL)

    # ==========================
    # Test 5: lambdas
    # ==========================
    lambdas = list(computations._gram_computation._lambdas.values())[0]
    check_sizes_and_values(lambdas, phi_lambdas, atol=ATOL, rtol=RTOL)

    # ==========================
    # Test 6: Newton step
    # ==========================
    newton_step = [
        [getattr(param, savefield) for param in group["params"]]
        for group in param_groups
    ][0][0]
    damped_GGN = phi_batch_GGN + delta * torch.eye(OUT_1)
    phi_newton_step = torch.solve(-phi_batch_grad.reshape(OUT_1, 1), damped_GGN)
    phi_newton_step = phi_newton_step.solution.reshape(-1)
    check_sizes_and_values(newton_step, phi_newton_step, atol=ATOL, rtol=RTOL)
