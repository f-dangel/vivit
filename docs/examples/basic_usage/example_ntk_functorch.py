"""Relation between the GGN Gram matrix in ViViT and the NTK in functorch.


The functorch code was taken from the tutorial at
https://pytorch.org/functorch/stable/notebooks/neural_tangent_kernels.html.

It challenges the reader: 'good luck trying to write an efficient version of
the above using stock PyTorch.' Challenge accepted

——–

Given two data sets x1, x2, and a model f, the NTK is Jf(x1) [Jf(x2)]ᵀ.

In ViViT we compute the GGN Gram matrix on a data set x via  [Jf(x) √H] [Jf(x) √H]ᵀ
where √H is the matrix square root of the loss Hessian w.r.t. the model.

The connection between these two is that for MSELoss we have √H = √2 I and
therefore we can compute [Jf(x) √2 I] [Jf(x) √2 I]ᵀ = 2 Jf(x) [Jf(x)]ᵀ, which
is proportional to the NTK!
"""

import time

import torch
import torch.nn as nn
from backpack import backpack, extend
from functorch import jacrev, jvp, make_functional, vjp, vmap

from vivit.extensions.secondorder.vivit import ViViTGGNExact

device = "cpu"


def CNN():
    """Same as in the functorch tutorial. Sequential for compatibility with BackPACK."""
    return nn.Sequential(
        nn.Conv2d(3, 32, (3, 3)),
        nn.ReLU(),
        nn.Conv2d(32, 32, (3, 3)),
        nn.ReLU(),
        nn.Conv2d(32, 32, (3, 3)),
        nn.Flatten(),
        nn.Linear(21632, 10),
    )


# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 =
#         self.conv2 =
#         self.conv3 =
#         self.fc =

#     def forward(self, x):
#         x = self.conv1(x)
#         x = x.relu()
#         x = self.conv2(x)
#         x = x.relu()
#         x = self.conv3(x)
#         x = x.flatten(1)
#         x = self.fc(x)
#         return x


x_train = torch.randn(20, 3, 32, 32, device=device)
x_test = torch.randn(5, 3, 32, 32, device=device)

net = CNN().to(device)
fnet, params = make_functional(net)


def fnet_single(params, x):
    return fnet(params, x.unsqueeze(0)).squeeze(0)


def empirical_ntk(fnet_single, params, x1, x2):
    # Compute J(x1)
    jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1)
    jac1 = [j.flatten(2) for j in jac1]

    # Compute J(x2)
    jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2)
    jac2 = [j.flatten(2) for j in jac2]

    # Compute J(x1) @ J(x2).T
    result = torch.stack(
        [torch.einsum("Naf,Mbf->NMab", j1, j2) for j1, j2 in zip(jac1, jac2)]
    )
    result = result.sum(0)
    return result


def empirical_ntk_implicit(func, params, x1, x2):
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
        basis = torch.eye(
            output.numel(), dtype=output.dtype, device=output.device
        ).view(output.numel(), -1)
        return vmap(get_ntk_slice)(basis)

    # get_ntk(x1, x2) computes the NTK for a single data point x1, x2
    # Since the x1, x2 inputs to empirical_ntk_implicit are batched,
    # we actually wish to compute the NTK between every pair of data points
    # between {x1} and {x2}. That's what the vmaps here do.
    return vmap(vmap(get_ntk, (None, 0)), (0, None))(x1, x2)


result = empirical_ntk(fnet_single, params, x_train, x_train)
print(result.shape)

# BackPACK & ViViT

net = extend(net)
loss_func = extend(nn.MSELoss(reduction="sum"))

with backpack(ViViTGGNExact()):
    output = net(x_train)
    y = torch.zeros_like(output)  # anything, won't affect NTK
    loss = loss_func(output, y)
    loss.backward()

# for p in net.parameters():
#     print(p.vivit_ggn_exact)

# The Hessian of MSE is 2I, hence we need to divide by 2
gram = 0.5 * sum(p.vivit_ggn_exact["gram_mat"]() for p in net.parameters())
print(gram.shape)

gram_reordered = torch.einsum("cndm->nmcd", gram)
print(gram_reordered.shape)

print(result[0, 0, 0, 0])
print(gram_reordered[0, 0, 0, 0])

print(torch.allclose(result, gram_reordered, atol=1e-6))


class Hook:
    def __init__(self, delete_buffers):
        self.gram = None
        self.delete_buffers = delete_buffers

    def __call__(self, module):
        for p in module.parameters():
            gram_p = p.vivit_ggn_exact["gram_mat"]()
            self.gram = gram_p if self.gram is None else self.gram + gram_p

            if self.delete_buffers:
                del p.vivit_ggn_exact


def empirical_ntk_backpack(net, x1, x2, delete_buffers=True):
    N1 = x1.shape[0]
    x = torch.cat([x1, x2])

    net = extend(net)
    loss_func = extend(nn.MSELoss(reduction="sum"))

    hook = Hook(delete_buffers)

    with backpack(ViViTGGNExact(), extension_hook=hook):
        output = net(x)
        y = torch.zeros_like(output)  # anything, won't affect NTK
        loss = loss_func(output, y)
        loss.backward()

    # The Hessian of MSE is 2I, hence we need to divide by 2
    gram = hook.gram
    gram = 0.5 * gram
    gram_reordered = torch.einsum("cndm->nmcd", gram)

    # slice out relevant blocks
    return gram_reordered[:N1, N1:]


start_functorch = time.time()
ntk_functorch = empirical_ntk(fnet_single, params, x_train, x_test)
end_functorch = time.time()

start_functorch_implicit = time.time()
ntk_functorch = empirical_ntk_implicit(fnet_single, params, x_train, x_test)
end_functorch_implicit = time.time()

start_backpack = time.time()
ntk_backpack = empirical_ntk_backpack(net, x_train, x_test)
end_backpack = time.time()
print(ntk_functorch.shape)
print(ntk_backpack.shape)

print(torch.allclose(ntk_functorch, ntk_backpack, atol=1e-6))
print(f"Time functorch: {end_functorch - start_functorch}")
print(f"Time backpack: {end_backpack - start_backpack}")
print(f"Time functorch implicit: {end_functorch_implicit - start_functorch_implicit}")
