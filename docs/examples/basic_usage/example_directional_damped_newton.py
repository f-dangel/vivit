"""
Computing directionally damped Newton steps
===========================================

TODO

First, the imports.
"""
from typing import List

from backpack import backpack, extend
from backpack.utils.examples import _autograd_ggn_exact_columns
from torch import (
    Tensor,
    allclose,
    cat,
    cuda,
    device,
    einsum,
    manual_seed,
    ones_like,
    rand,
    stack,
    zeros_like,
)
from torch.autograd import grad
from torch.nn import Linear, MSELoss, ReLU, Sequential
from torch.nn.utils.convert_parameters import parameters_to_vector

from vivit.optim.directional_damped_newton import DirectionalDampedNewtonComputation

# make deterministic
manual_seed(0)

# %%
# Data, model & loss
# ^^^^^^^^^^^^^^^^^^
#
# For this demo, we use toy data and a small MLP with sufficiently few
# parameters such that we can store the GGN matrix to verify our results. We
# use mean squared error as loss function.

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
#
# Next, :py:func:`extend <backpack.extend>` the model and loss function to be
# able to use BackPACK. Then, we perform a forward pass to compute the loss.

model = extend(model)
loss_function = extend(loss_function)

loss = loss_function(model(X), y)

# %%
# Specify GGN approximation and directions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# By default, :py:class:`vivit.DirectionalDampedNewtoComputation` uses the exact GGN.
# Furthermore, we need to specify the GGN's parameters via a ``param_groups`` argument
# that might be familiar to you from :py:mod:`torch.optim`. It also contains a filter
# function that selects the eigenvalues whose eigenvectors will be used as directions
# for the Newton step.

computation = DirectionalDampedNewtonComputation()


def select_top_k(evals: Tensor, k=4) -> List[int]:
    """Select the top-k eigenvalues as directions to evaluate derivatives.

    Args:
        evals: Eigenvalues, sorted in ascending order.
        k: Number of leading eigenvalues. Defaults to ``4``.

    Returns:
        Indices of top-k eigenvalues.
    """
    return [evals.numel() - k + idx for idx in range(k)]


# %%
#
# Also need a damping function


def constant_damping(
    evals: Tensor, evecs: Tensor, gammas: Tensor, lambdas: Tensor
) -> Tensor:
    return ones_like(evals)


group = {
    "params": [p for p in model.parameters() if p.requires_grad],
    "criterion": select_top_k,
    "damping": constant_damping,
}
param_groups = [group]


# %%
# Backward pass with BackPACK
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We can now build the BackPACK extensions and extension hook that will compute
# directional derivatives, pass them to a :py:class:`with backpack
# <backpack.backpack>`, and perform the backward pass.

extensions = computation.get_extensions()
extension_hook = computation.get_extension_hook(param_groups)

with backpack(*extensions, extension_hook=extension_hook):
    loss.backward()

# %%
#
# This will compute the damped Newton step for each parameter group and store
# it internally in the :py:class:`vivit.DirectionalDampedNewtonComputation`
# instance. We can use the parameter group to request them.

newton_step = computation.get_result(group)

# %%
# Verify results
# ^^^^^^^^^^^^^^
#
# To verify the above, let's first compute the gradient and the GGN using
# :py:mod:`torch.autograd`.
gradient = grad(
    loss_function(model(X), y), [p for p in model.parameters() if p.requires_grad]
)
# flatten it into a vector
gradient = cat([g.flatten() for g in gradient])
ggn = stack([col for _, col in _autograd_ggn_exact_columns(X, y, model, loss_function)])


# %%
#
# Next, we need the filtered GGN eigenvectors:

evals, evecs = ggn.symeig(eigenvectors=True)
keep = select_top_k(evals)
evals, evecs = evals[keep], evecs[:, keep]

# %%
#
# We can now form the Newton step:

newton_step_torch = zeros_like(ggn[0])
print(newton_step_torch.shape)

K = evals.numel()

for k in range(K):
    evec = evecs[:, k]
    gamm = einsum("i,i", gradient, evec)
    lamb = evals[k]
    delta = 1.0

    ns = (-gamm / (lamb + delta)) * evec
    newton_step_torch = newton_step_torch + ns

# %%
#
# Convert into flat format:

newton_step = parameters_to_vector(newton_step)


# %%
# Verify results
# ^^^^^^^^^^^^^^
#
print(allclose(newton_step, newton_step_torch, rtol=1e-5, atol=1e-7))
