r"""
Computing empirical NTKs
========================

In this example we will use ``vivit`` to compute empirical NTK matrices.

The ``functorch`` package allows to do this efficiently. `One of its tutorials
<https://pytorch.org/functorch/stable/notebooks/neural_tangent_kernels.html>`_
states that doing this in stock PyTorch is hard ... well, challenge accepted!
Let's see how ``vivit`` and ``functorch`` compare.

Given two data sets :math:`\mathbf{X}_1`, :math:`\mathbf{X}_2`, and a model
:math:`f_\theta`, the empirical NTK is :math:`\mathbf{J}_\theta
f_\theta(\mathbf{X}_1) [\mathbf{J}_\theta f_\theta(\mathbf{X}_2)]^{\top}`.

``vivit`` can compute the GGN Gram matrix :math:`[\mathbf{J}_\theta
f_\theta(\mathbf{X}) \sqrt{\mathbf{H}}] [\mathbf{J}_\theta f_\theta(\mathbf{X})
\sqrt{\mathbf{H}}]^\top` on a data set :math:`\mathbf{X}` where
:math:`\sqrt{\mathbf{H}}` is the matrix square root of the loss Hessian w.r.t.
the model's prediction.

For ``MSELoss`` we have :math:`\sqrt{\mathbf{H}} = 2 \mathbf{I}` and therefore
we can compute :math:`[\mathbf{J}_\theta f_\theta (\mathbf{X}) \sqrt{2}
\mathbf{I}] [\mathbf{J}_\theta f_\theta (\mathbf{X}) \sqrt{2} \mathbf{I}]^\top
= \mathbf{J}_\theta f_\theta(\mathbf{X}) [\mathbf{J}_\theta
f_\theta(\mathbf{X})]^{\top}`. If we stack :math:`\mathbf{X}_1` and
:math:`\mathbf{X}_2` into a data set :math:`\mathbf{X}`, a submatrix of the Gram matrix
is proportional to the empirical NTK!

Let's get the imports out of our way.
"""

import time

from backpack import backpack, extend
from functorch import jacrev, jvp, make_functional, vjp, vmap
from torch import allclose, cat, einsum, eye, manual_seed, randn, stack, zeros_like
from torch.nn import Conv2d, Flatten, Linear, MSELoss, ReLU, Sequential

from vivit.extensions.secondorder.vivit import ViViTGGNExact

device = "cpu"
manual_seed(0)

# %%
# Setup
# -----
#
# We will use the same CNN as the ``functorch`` tutorial and create the data
# sets :math:`\mathbf{X}_1`, :math:`\mathbf{X}_2`.


def CNN():
    """Same as in the functorch tutorial. Sequential for compatibility with BackPACK."""
    return Sequential(
        Conv2d(3, 32, (3, 3)),
        ReLU(),
        Conv2d(32, 32, (3, 3)),
        ReLU(),
        Conv2d(32, 32, (3, 3)),
        Flatten(),
        Linear(21632, 10),
    )


x_train = randn(20, 3, 32, 32, device=device)
x_test = randn(5, 3, 32, 32, device=device)

net = CNN().to(device)

# %%
# NTK with functorch
# ------------------
#
# The functorch tutorial provides two different methods to compute the
# empirical NTK. We just copy them over here.

fnet, params = make_functional(net)


def fnet_single(params, x):
    """From the functorch tutorial."""
    return fnet(params, x.unsqueeze(0)).squeeze(0)


def empirical_ntk_functorch(fnet_single, params, x1, x2):
    """From the functorch tutorial."""
    # Compute J(x1)
    jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1)
    jac1 = [j.flatten(2) for j in jac1]

    # Compute J(x2)
    jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2)
    jac2 = [j.flatten(2) for j in jac2]

    # Compute J(x1) @ J(x2).T
    result = stack([einsum("Naf,Mbf->NMab", j1, j2) for j1, j2 in zip(jac1, jac2)])
    result = result.sum(0)
    return result


def empirical_ntk_implicit_functorch(func, params, x1, x2):
    """From the functorch tutorial."""

    def get_ntk(x1, x2):
        def func_x1(params):
            return func(params, x1)

        def func_x2(params):
            return func(params, x2)

        output, vjp_fn = vjp(func_x1, params)

        def get_ntk_slice(vec):
            # This computes vec @ J(x2).T
            # `vec` is some unit vector (a single slice of the Identity matrix)
            vjps = vjp_fn(vec)
            # This computes J(X1) @ vjps
            _, jvps = jvp(func_x2, (params,), vjps)
            return jvps

        # Here's our identity matrix
        basis = eye(output.numel(), dtype=output.dtype, device=output.device).view(
            output.numel(), -1
        )
        return vmap(get_ntk_slice)(basis)

    # get_ntk(x1, x2) computes the NTK for a single data point x1, x2
    # Since the x1, x2 inputs to empirical_ntk_implicit are batched,
    # we actually wish to compute the NTK between every pair of data points
    # between {x1} and {x2}. That's what the vmaps here do.
    return vmap(vmap(get_ntk, (None, 0)), (0, None))(x1, x2)


# %%
# Let's compute an NTK matrix:

ntk_functorch = empirical_ntk_functorch(fnet_single, params, x_train, x_train)

# %%
# NTK with ViViT
# --------------
#
# As outlined above, to compute the NTK with ``vivit``, we need to stack the
# two data sets, feed them through the network and an ``MSELoss`` function, then
# compute the GGN Gram matrix during backpropagation. The latter is done by
# ``vivit``'s ``ViViTGGNExact`` extension, which gives access to the per-layer
# Gram matrix. We have to accumulate the Gram matrices over layers. To do that,
# we use the following hook:


class AccumulateGramHook:
    """Accumulate the Gram matrix during backpropagation with BackPACK."""

    def __init__(self, delete_buffers):
        self.gram = None
        self.delete_buffers = delete_buffers

    def __call__(self, module):
        for p in module.parameters():
            gram_p = p.vivit_ggn_exact["gram_mat"]()
            self.gram = gram_p if self.gram is None else self.gram + gram_p

            if self.delete_buffers:
                del p.vivit_ggn_exact


# %%
# The above steps are then implemented by the following function:


def empirical_ntk_vivit(net, x1, x2, delete_buffers=True):
    """Compute the empirical NTK matrix with ViViT."""
    N1 = x1.shape[0]
    X = cat([x1, x2])

    # make BackPACK-ready
    net = extend(net)
    loss_func = extend(MSELoss(reduction="sum"))
    hook = AccumulateGramHook(delete_buffers)

    with backpack(ViViTGGNExact(), extension_hook=hook):
        output = net(X)
        y = zeros_like(output)  # anything, won't affect NTK
        loss = loss_func(output, y)
        loss.backward()

    gram_reordered = einsum("cndm->nmcd", hook.gram)

    # slice out relevant blocks & fix scaling of MSELoss
    return 0.5 * gram_reordered[:N1, N1:]


# %%
# Check
# -----
#
# Let's check that the ``vivit`` and ``functorch`` implementations produce the
# same NTK matrix:

ntk_functorch = empirical_ntk_functorch(fnet_single, params, x_train, x_test)
ntk_vivit = empirical_ntk_vivit(net, x_train, x_test)

close = allclose(ntk_functorch, ntk_vivit, atol=1e-6)
if close:
    print("NTK from functorch and vivit match!")
else:
    raise ValueError("NTK from functorch and vivit don't match!")


# %%
# Runtime
# -------
#
# Last but not least, let's compare the three methods in terms of runtime:

t_functorch = time.time()
empirical_ntk_functorch(fnet_single, params, x_train, x_test)
t_functorch = time.time() - t_functorch

t_functorch_implicit = time.time()
empirical_ntk_implicit_functorch(fnet_single, params, x_train, x_test)
t_functorch_implicit = time.time() - t_functorch_implicit

t_vivit = time.time()
empirical_ntk_vivit(net, x_train, x_test)
t_vivit = time.time() - t_vivit

t_min = min(t_functorch, t_functorch_implicit, t_vivit)

print(f"Time [s] functorch:          {t_functorch:.4f} (x{t_functorch / t_min:.2f})")
print(f"Time [s] vivit:              {t_vivit:.4f} (x{t_vivit/ t_min:.2f})")
print(
    f"Time [s] functorch implicit: {t_functorch_implicit:.4f}"
    + f" (x{t_functorch_implicit / t_min:.2f})"
)

# %%
# We can see that ``vivit`` is competitive with ``functorch``.
