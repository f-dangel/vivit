"""
Computing directionally damped Newton steps
===========================================

In this example we demonstrate how to use ViViT's
:py:class:`DirectionalDampedNewtonComputation
<vivit.DirectionalDampedNewtonComputation>` to compute directionally damped
Newton steps with the GGN. We verify the result with :py:mod:`torch.autograd`.

First, the imports.
"""
from typing import List

from backpack import backpack, extend
from backpack.utils.examples import _autograd_ggn_exact_columns
from torch import (
    Tensor,
    allclose,
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
# By default, :py:class:`vivit.DirectionalDampedNewtonComputation` uses the
# exact GGN. Furthermore, we need to specify the GGN's parameters via a
# ``param_groups`` argument that might be familiar to you from
# :py:mod:`torch.optim`. It also contains a filter function that selects the
# eigenvalues whose eigenvectors will be used as directions for the Newton
# step.

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
# Specify directional damping
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We also need a damping function that provides the damping value for each
# direction. This function receives the GGNs eigenvalues, Gram matrix
# eigenvectors, as well as first- and second-order directional derivatives. It
# returns a one-dimensional tensor that contains the damping values for all
# directions.
#
# This seems overly complicated. But this approach allows for incorporating
# information about gradient and curvature noise into the damping value.
#
# For simplicity, we will use a constant damping of 1 for all
# directions.

DAMPING = 1.0


def constant_damping(
    evals: Tensor, evecs: Tensor, gammas: Tensor, lambdas: Tensor
) -> Tensor:
    """Constant damping along all directions.

    Args:
        evals: GGN eigenvalues. Shape ``[K]``.
        evecs: GGN Gram matrix eigenvectors. Shape ``[NC, K]``.
        gammas: Directional gradients. Shape ``[N, K]``.
        lambdas: Directional curvatures. Shape ``[N, K]``.

    Returns:
        Directional dampings. Shape ``[K]``.
    """
    return DAMPING * ones_like(evals)


# %%
#
# Let's put everything together and set up the parameter groups.

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
# the damped Newton step, pass them to a :py:class:`with backpack
# <backpack.backpack>`, and perform the backward pass.

extensions = computation.get_extensions()
extension_hook = computation.get_extension_hook(param_groups)

with backpack(*extensions, extension_hook=extension_hook):
    loss.backward()

# %%
#
# This will compute the damped Newton step for each parameter group and store
# it internally in the :py:class:`vivit.DirectionalDampedNewtonComputation`
# instance. We can use the parameter group to request it.

newton_step = computation.get_result(group)

# %%
#
# It has the same format as the ``group['params']`` entry:

for param, newton in zip(group["params"], newton_step):
    print(f"Parameter shape:   {param.shape}\nNewton step shape: {newton.shape}\n")

# %%
#
# We will flatten and concatenate the Newton step over parameters to simplify
# the comparison with :py:mod:`torch.autograd`.

newton_step_flat = parameters_to_vector(newton_step)
print(newton_step_flat.shape)

# %%
# Verify results
# ^^^^^^^^^^^^^^
#
# Let's compute the damped Newton step with :py:mod:`torch.autograd` and verify
# it leads to the same result.
#
# We need the gradient and the GGN.
gradient = grad(
    loss_function(model(X), y), [p for p in model.parameters() if p.requires_grad]
)
gradient = parameters_to_vector(gradient)

ggn = stack([col for _, col in _autograd_ggn_exact_columns(X, y, model, loss_function)])

print(gradient.shape, ggn.shape)

# %%
#
# Next, eigen-decompose the GGN and filter the relevant eigenpairs:

evals, evecs = ggn.symeig(eigenvectors=True)
keep = select_top_k(evals)
evals, evecs = evals[keep], evecs[:, keep]

# %%
#
# This is sufficient to form the damped Newton step
#
# .. math::
#    s = \sum_{k=1}^K \frac{-\gamma_k}{\lambda_k + \delta} e_k
#
# with constant damping :math:`\delta = 1`.

newton_step_torch = zeros_like(gradient)

K = evals.numel()

for k in range(K):
    evec = evecs[:, k]
    gamm = einsum("i,i", gradient, evec)
    lamb = evals[k]

    newton = (-gamm / (lamb + DAMPING)) * evec
    newton_step_torch += newton

print(newton_step_torch.shape)

# %%
#
# Both damped Newton steps should be identical.

close = allclose(newton_step_flat, newton_step_torch, rtol=1e-5, atol=1e-7)
if not close:
    raise ValueError("Directionally damped Newton steps don't match!")

print("Directionally damped Newton steps match!")
