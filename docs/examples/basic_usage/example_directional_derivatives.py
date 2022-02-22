"""
Computing directional derivatives along GGN eigenvectors
========================================================

In this example we demonstrate how to use ViViT's
:py:class:`DirectionalDerivativesComputation <vivit.DirectionalDerivativesComputation>`
to obtain the 1ˢᵗ- and 2ⁿᵈ-order directional derivatives along the leading GGN
eigenvectors. We verify the result with :py:mod:`torch.autograd`.

First, the imports.
"""

from typing import List

from backpack import backpack, extend
from backpack.utils.examples import _autograd_ggn_exact_columns
from torch import Tensor, cuda, device, einsum, isclose, manual_seed, rand, stack, zeros
from torch.autograd import grad
from torch.nn import Linear, MSELoss, ReLU, Sequential
from torch.nn.utils.convert_parameters import parameters_to_vector

from vivit.optim.directional_derivatives import DirectionalDerivativesComputation

# make deterministic
manual_seed(0)

# %%
# Data, model & loss
# ^^^^^^^^^^^^^^^^^^
# For this demo, we use toy data and a small MLP with sufficiently few
# parameters such that we can store the GGN matrix to verify our results
# (yes, one could use matrix-free GGN-vector products instead).
# We use mean squared error as loss function.

N = 4
D_in = 7
D_hidden = 5
D_out = 3

DEVICE = device("cuda" if cuda.is_available() else "cpu")

X = rand(N, D_in).to(DEVICE)
y = rand(N, D_out).to(DEVICE)

model = Sequential(
    Linear(D_in, D_hidden),
    ReLU(),
    Linear(D_hidden, D_hidden),
    ReLU(),
    Linear(D_hidden, D_out),
).to(DEVICE)

loss_function = MSELoss(reduction="mean").to(DEVICE)

# %%
# Integrate BackPACK
# ^^^^^^^^^^^^^^^^^
# Next, :py:func:`extend <backpack.extend>` the model and loss function to be able
# to use BackPACK. Then, we perform a forward pass to compute the loss.

model = extend(model)
loss_function = extend(loss_function)

loss = loss_function(model(X), y)

# %%
# Specify GGN approximation and directions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# By default, :py:class:`vivit.DirectionalDerivativesComputation` uses the exact GGN.
# Furthermore, we need to specify the GGN's parameters via a ``param_groups`` argument
# that might be familiar to you from :py:mod:`torch.optim`. It also contains a filter
# function that selects the eigenvalues whose eigenvectors will be used as directions
# to evaluate directional derivatives.

computation = DirectionalDerivativesComputation()


def select_top_k(evals: Tensor, k=4) -> List[int]:
    """Select the top-k eigenvalues as directions to evaluate derivatives.

    Args:
        evals: Eigenvalues, sorted in ascending order.
        k: Number of leading eigenvalues. Defaults to ``4``.

    Returns:
        Indices of top-k eigenvalues.
    """
    return [evals.numel() - k + idx for idx in range(k)]


group = {
    "params": [p for p in model.parameters() if p.requires_grad],
    "criterion": select_top_k,
}
param_groups = [group]

# %%
# Backward pass with BackPACK
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We can now build the BackPACK extensions and extension hook that will compute
# directional derivatives, pass them to a :py:class:`with backpack <backpack.backpack>`,
# and perform the backward pass.

extensions = computation.get_extensions()
extension_hook = computation.get_extension_hook(param_groups)

with backpack(*extensions, extension_hook=extension_hook):
    loss.backward()

# %%
# This will compute the directional derivatives for each
# parameter group and store them internally in the
# :py:class:`DirectionalDerivativesComputation<vivit.DirectionalDerivativesComputation>`
# instance. We can use the parameter group to request them.

gammas_vivit, lambdas_vivit = computation.get_result(group)

# %%
# Verify results
# ^^^^^^^^^^^^^^
# To verify the above, let's first compute the per-sample gradients and GGNs using
# :py:mod:`torch.autograd`.
batch_grad = []
batch_ggn = []

for n in range(N):
    x_n, y_n = X[[n]], y[[n]]

    grad_n = grad(
        loss_function(model(x_n), y_n),
        [p for p in model.parameters() if p.requires_grad],
    )
    batch_grad.append(parameters_to_vector(grad_n))

    ggn_n = stack(
        [col for _, col in _autograd_ggn_exact_columns(x_n, y_n, model, loss_function)]
    )
    batch_ggn.append(ggn_n)

# %%
# We also need the GGN eigenvectors as directions
ggn = stack([col for _, col in _autograd_ggn_exact_columns(X, y, model, loss_function)])
evals, evecs = ggn.symeig(eigenvectors=True)
keep = select_top_k(evals)
evals, evecs = evals[keep], evecs[:, keep]

# %%
# We are now ready to compute and compare the target quantities.
#
# First, compute and compare the first-order directional derivatives. Note that since
# the GGN eigenvectors used as directions are not unique but can point in the opposite
# direction. The directional gradient can thus be of different sign and we only compare
# the absolute value.
K = evals.numel()
gammas_torch = zeros(N, K, device=evals.device, dtype=evals.dtype)

for n in range(N):
    grad_n = batch_grad[n]
    for k in range(K):
        e_k = evecs[:, k]

        gammas_torch[n, k] = einsum("i,i", grad_n, e_k)

for gamma_vivit, gamma_torch in zip(gammas_vivit.flatten(), gammas_torch.flatten()):
    close = isclose(abs(gamma_vivit), abs(gamma_torch), rtol=1e-4, atol=1e-7)
    print(f"{gamma_vivit:.5e} vs. {gamma_torch:.5e}, close: {close}")
    if not close:
        raise ValueError("1ˢᵗ-order directional derivatives don't match!")

print("1ˢᵗ-order directional derivatives match!")

# %%
# Next, compute and compare the second-order directional derivatives.
lambdas_torch = zeros(N, K, device=evals.device, dtype=evals.dtype)

for n in range(N):
    ggn_n = batch_ggn[n]
    for k in range(K):
        e_k = evecs[:, k]

        lambdas_torch[n, k] = einsum("i,ij,j", e_k, ggn_n, e_k)

for lambda_vivit, lambda_torch in zip(lambdas_vivit.flatten(), lambdas_torch.flatten()):
    close = isclose(lambda_vivit, lambda_torch, rtol=1e-4, atol=1e-7)
    print(f"{lambda_vivit:.5e} vs. {lambda_torch:.5e}, close: {close}")
    if not close:
        raise ValueError("2ⁿᵈ-order directional derivatives don't match!")

print("2ⁿᵈ-order directional derivatives match!")

# %%
# Last, we check that the sample means of second-order derivatives coincide with
# the eigenvalues.

for eval_vivit, eval_torch in zip(lambdas_vivit.mean(0), evals):
    close = isclose(eval_vivit, eval_torch, rtol=1e-4, atol=1e-7)
    print(f"{eval_vivit:.5e} vs. {eval_torch:.5e}, close: {close}")
    if not close:
        print("Averaged 2ⁿᵈ-order directional derivatives don't match eigenvalues!")

print("Averaged 2ⁿᵈ-order directional derivatives match eigenvalues!")
