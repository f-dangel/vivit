"""
Computing GGN eigenpairs
========================

In this example we demonstrate how to use ViViT's
:py:class:`EighComputation <vivit.EighComputation>` to obtain the leading GGN
eigenpairs (eigenvalues and associated eigenvectors). We verify the result with
:py:mod:`torch.autograd`.

First, the imports.
"""

from typing import List

from backpack import backpack, extend
from backpack.utils.examples import _autograd_ggn_exact_columns
from torch import Tensor, allclose, cat, cuda, device, manual_seed, rand, stack
from torch.nn import Linear, MSELoss, ReLU, Sequential

from vivit.linalg.eigh import EighComputation

# make deterministic
manual_seed(0)

# %%
# Data, model & loss
# ^^^^^^^^^^^^^^^^^^
# For this demo, we use toy data and a small MLP with sufficiently few
# parameters such that we can store the GGN matrix to verify the eigen-properties
# of our results (yes, one could use matrix-free GGN-vector products instead, but by
# expanding the GGN matrix we will familiarize ourselves more with the format of
# the results). We use mean squared error as loss function.

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
# Specify GGN approximation and eigenpair filter
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# By default, :py:class:`vivit.EighComputation` uses the exact GGN. Furthermore, we need
# to specify the GGN's parameters via a ``param_groups`` argument that might be familiar
# to you from :py:mod:`torch.optim`. It also contains a filter function that selects the
# eigenvalues whose eigenvectors should be computed (computing all eigenvectors is
# infeasible for big architectures).

computation = EighComputation()


def select_top_k(evals: Tensor, k=4) -> List[int]:
    """Select the top-k eigenvalues for the eigenvector computation.

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
# We can now build the BackPACK extension and extension hook that will compute
# eigenpairs, pass them to a :py:class:`with backpack <backpack.backpack>`, and
# perform the backward pass.

extension = computation.get_extension()
extension_hook = computation.get_extension_hook(param_groups)

with backpack(extension, extension_hook=extension_hook):
    loss.backward()

# %%
# This will compute the GGN eigenpairs for each parameter group and store them
# internally in the :py:class:`EighComputation <vivit.EighComputation>` instance.
# We can use the parameter group to request the eigenpairs.

evals, evecs = computation.get_result(group)

# %%
# The eigenvectors have a similar format than the parameters. The leading axis
# allows to access eigenvectors for an eigenvalue.
print("Parameter shape    |  Eigenvector shape")
for p, v in zip(group["params"], evecs):
    print(f"{str(p.shape):<19}|  {v.shape}")

# %%
# In the following, we will flatten and concatenate them among parameters, such that
# ``evecs_flat[k,:]`` is the GGN eigenvector with eigenvalue ``evals[k]``:
evecs_flat = cat([e.flatten(start_dim=1) for e in evecs], dim=1)

# %%
# Verify results
# ^^^^^^^^^^^^^^
# To verify the above, let's compute the GGN matrix, column by column, using GGN-vector
# products that only rely on :py:mod:`torch.autograd`.
ggn = stack([col for _, col in _autograd_ggn_exact_columns(X, y, model, loss_function)])

# %%
# We can then check that application of the GGN to an eigenvector rescales the latter by
# its eigenvalue.
for e, v in zip(evals, evecs_flat):
    ggn_v = ggn @ v
    close = allclose(e * v, ggn_v, rtol=1e-4, atol=1e-7)

    print(f"Eigenvalue {e:.5e}, Eigenvector properties: {close}")
    if not close:
        raise ValueError("Eigenvector properties failed!")

print("Eigenvector properties confirmed!")
