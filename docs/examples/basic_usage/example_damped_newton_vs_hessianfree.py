r"""
Computing directionally damped Newton steps
===========================================

In this example we demonstrate how to use ViViT's
:py:class:`DirectionalDampedNewtonComputation
<vivit.DirectionalDampedNewtonComputation>` to compute directionally damped
Newton steps with the GGN. We verify the result with :py:mod:`torch.autograd`.

The math is as follows: Let's consider the damped Newton step

.. math::
    - (G + \delta I)^{-1} g
    =
    \sum_{k=1}^K \frac{-\gamma_k}{\lambda_k + \delta} e_k
    +
    \sum_{k=K+1}^D \frac{-\gamma_k}{\delta} e_k

We can rewrite this into

.. math::
    =
    \sum_{k=1}^K
    (
    \frac{-\gamma_k}{\lambda_k + \delta} + \frac{\gamma_k}{\delta}
    ) e_k
    +
    \sum_{k=1}^D \frac{-\gamma_k}{\delta} e_k
    =
    \sum_{k=1}^K
    (
    \frac{-\gamma_k}{\lambda_k + \delta} + \frac{\gamma_k}{\delta}
    ) e_k
    - \frac{-g}{\delta}

The coefficients of the first term can be rearranged into the damping parameterization

.. math::
    =
    \sum_{k=1}^K
    (
    \frac{-\gamma_k}{\lambda_k - (\frac{(\lambda_k + \delta) \delta}{\lambda_k}}
    ) e_k
    - \frac{-g}{\delta}
    :=
    \sum_{k=1}^K
    (
    \frac{-\gamma_k}{\lambda_k + d}
    ) e_k
    - \frac{-g}{\delta}

with the damping :math:`d`.

First, the imports.
"""
from time import time
from typing import List, Optional

from backpack import backpack, extend
from backpack.hessianfree.ggnvp import ggn_vector_product_from_plist

# make deterministic
from torch import Tensor, allclose, cuda, device, manual_seed, zeros_like
from torch.autograd import grad
from torch.nn import (
    Conv2d,
    CrossEntropyLoss,
    Flatten,
    Linear,
    MaxPool2d,
    ReLU,
    Sequential,
)

from vivit.optim.directional_damped_newton import DirectionalDampedNewtonComputation

manual_seed(0)
# set_seeds(0)

from backpack.utils.examples import load_one_batch_mnist

# %%
# Data, model & loss
# ^^^^^^^^^^^^^^^^^^
#
# For this demo, we use toy data and a small MLP with sufficiently few
# parameters such that we can store the GGN matrix to verify our results. We
# use mean squared error as loss function.

N = 64

DEVICE = device("cuda" if cuda.is_available() else "cpu")

X, y = load_one_batch_mnist(N)
X = X.to(DEVICE)
y = y.to(DEVICE)

model = Sequential(
    Conv2d(1, 20, 5, 1),
    ReLU(),
    MaxPool2d(2, 2),
    Conv2d(20, 50, 5, 1),
    ReLU(),
    MaxPool2d(2, 2),
    Flatten(),
    Linear(4 * 4 * 50, 500),
    ReLU(),
    Linear(500, 10),
).to(DEVICE)
loss_function = CrossEntropyLoss(reduction="mean").to(DEVICE)


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


def select_larger_than(evals: Tensor, eps=1e-4) -> List[int]:
    """Select eigenvalues that are larger than eps.

    Args:
        evals: Eigenvalues, sorted in ascending order.
        eps: Minimum value of an eigenvalue to be selected. Defaults to ``4``.

    Returns:
        Indices of eigenvalues larger than eps.
    """
    return [idx for idx, lamb in enumerate(evals) if lamb > eps]


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

DAMPING = 0.1


def damping_function(
    evals: Tensor, evecs: Tensor, gammas: Tensor, lambdas: Tensor
) -> Tensor:
    # """Constant damping along all directions.

    # Args:
    #     evals: GGN eigenvalues. Shape ``[K]``.
    #     evecs: GGN Gram matrix eigenvectors. Shape ``[NC, K]``.
    #     gammas: Directional gradients. Shape ``[N, K]``.
    #     lambdas: Directional curvatures. Shape ``[N, K]``.

    # Returns:
    #     Directional dampings. Shape ``[K]``.
    # """
    # return DAMPING * ones_like(evals)
    return -(evals + ((evals + DAMPING) * DAMPING) / evals)


# %%
#
# Let's put everything together and set up the parameter groups.

group = {
    "params": [p for p in model.parameters() if p.requires_grad],
    "criterion": select_larger_than,
    "damping": damping_function,
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

start = time()

with backpack(*extensions, extension_hook=extension_hook):
    loss.backward()

# %%
#
# This will compute the damped Newton step for each parameter group and store
# it internally in the :py:class:`vivit.DirectionalDampedNewtonComputation`
# instance. We can use the parameter group to request it.

newton_step = list(computation.get_result(group))

for idx, param in enumerate(group["params"]):
    newton_step[idx] -= param.grad / DAMPING

print(f"Time BackPACK: {time() - start}")

# %%
#
# It has the same format as the ``group['params']`` entry:

# for param, newton in zip(group["params"], newton_step):
#     print(f"Parameter shape:   {param.shape}\nNewton step shape: {newton.shape}\n")

# %%
#
# We will flatten and concatenate the Newton step over parameters to simplify
# the comparison with :py:mod:`torch.autograd`.

# %%
# Verify results
# ^^^^^^^^^^^^^^
#
# Let's compute the damped Newton step with :py:mod:`torch.autograd` and verify
# it leads to the same result.
#
# We need the gradient and the GGN.
output = model(X)
loss = loss_function(output, y)
neg_gradient = [
    -g
    for g in grad(
        loss, [p for p in model.parameters() if p.requires_grad], retain_graph=True
    )
]


def ggn_matvec(v_list):
    p_list = group["params"]
    ggn_v_list = ggn_vector_product_from_plist(loss, output, p_list, v_list)
    return [ggn_v + DAMPING * v for ggn_v, v in zip(ggn_v_list, v_list)]


def cg(matvec, b: List[Tensor], x0: Optional[List[Tensor]] = None, maxiter: int = None):
    if maxiter is None:
        maxiter = sum(b_el.numel() for b_el in b)

    x = x0 if x0 is not None else [zeros_like(b_el) for b_el in b]

    # initialize parameters
    r = [b_el - Ab_el for b_el, Ab_el in zip(b, matvec(x))]
    p = [r_el.clone() for r_el in r]
    rs_old = sum((r_el**2).sum() for r_el in r)

    # iterate
    iterations = 0
    while iterations < maxiter:
        Ap = matvec(p)
        alpha = rs_old / sum((p_el * Ap_el).sum() for p_el, Ap_el in zip(p, Ap))

        for x_el, p_el in zip(x, p):
            x_el.add_(alpha * p_el)
        for r_el, Ap_el in zip(r, Ap):
            r_el.sub_(alpha * Ap_el)
        rs_new = sum((r_el**2).sum() for r_el in r)

        iterations += 1

        for p_el, r_el in zip(p, r):
            p_el.mul_(rs_new / rs_old)
            p_el.add_(r_el)

        rs_old = rs_new

    return x


MAXITER = 50

start = time()
newton_step_torch = cg(ggn_matvec, neg_gradient, maxiter=MAXITER)

print(f"Time HF: {time() - start}")


# %%
#
# Both damped Newton steps should be identical.
assert len(newton_step) == len(newton_step_torch)

for ns, ns_torch in zip(newton_step, newton_step_torch):
    assert ns.shape == ns_torch.shape
    close = allclose(ns, ns_torch, rtol=5e-4, atol=1e-5)
    if not close:
        for a, b in zip(ns.flatten(), ns_torch.flatten()):
            print(a, b, a / b)
        raise ValueError("Directionally damped Newton steps don't match!")
    else:
        print("Directionally damped Newton steps match!")
