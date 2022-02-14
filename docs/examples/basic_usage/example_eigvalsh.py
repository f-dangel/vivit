"""
Computing GGN eigenvalues
=========================

In this example we demonstrate how to use ViViT's
:py:class:`EigvalshComputation <vivit.EigvalshComputation>` to obtain the GGN's
eigenvalues and verify the result with :py:mod:`torch.autograd`.

First, the imports.
"""

from backpack import backpack, extend
from backpack.utils.examples import _autograd_ggn_exact_columns
from torch import cuda, device, isclose, manual_seed, rand, stack
from torch.linalg import eigvalsh
from torch.nn import Linear, MSELoss, ReLU, Sequential

from vivit.linalg.eigvalsh import EigvalshComputation

# make deterministic
manual_seed(0)

# %%
# Data, model & loss
# ^^^^^^^^^^^^^^^^^^
# For this demo, we use toy data and a small MLP with sufficiently few
# parameters such that we can store and eigen-decompose the GGN matrix to
# verify correctness. We use mean squared error as loss function.

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
# Specify GGN approximation
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# By default, :py:class:`vivit.EigvalshComputation` uses the exact GGN. We only need to
# specify the GGN's parameters via a ``param_groups`` argument that might be familiar
# to you from :py:mod:`torch.optim`.

computation = EigvalshComputation()

group = {"params": [p for p in model.parameters() if p.requires_grad]}
param_groups = [group]

# %%
# Backward pass with BackPACK
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We can now build the BackPACK extension and extension hook that will compute GGN
# eigenvalues, pass them to a :py:class:`with backpack <backpack.backpack>`, and
# perform the backward pass.

extension = computation.get_extension()
extension_hook = computation.get_extension_hook(param_groups)

with backpack(extension, extension_hook=extension_hook):
    loss.backward()

# %%
# This will compute the GGN eigenvalues for each parameter group and store them
# internally in the :py:class:`EigvalshComputation <vivit.EigvalshComputation>`
# instance. We can use the parameter group to request the eigenvalues.

evals = computation.get_result(group)

# %%
# Verify results
# ^^^^^^^^^^^^^^
# Let's compute the GGN matrix, column by column, using GGN-vector products that
# only rely on :py:mod:`torch.autograd`. We can then compute its eigenvalues and compare them.
ggn = stack([col for _, col in _autograd_ggn_exact_columns(X, y, model, loss_function)])
ggn_evals = eigvalsh(ggn)

# %%
# ViViT eigen-decomposes the GGN Gram matrix, which is smaller than the GGN.
# Hence, we compare against the leading eigenvalues from the GGN eigen-decomposition:
gram_dim = evals.numel()
ggn_evals = ggn_evals[-gram_dim:]

# %%
# Let's see if the eigenvalues match.

for eval_vivit, eval_torch in zip(evals, ggn_evals):
    close = isclose(eval_vivit, eval_torch, rtol=1e-4, atol=1e-7)
    print(f"{eval_vivit:.5e} vs. {eval_torch:.5e}, close: {close}")
    if not close:
        raise ValueError("Eigenvalues don't match!")

print("Eigenvalues match!")
