"""Tests for ``vivit.hessianfree.__init__``."""

from test.utils import classification_targets, get_available_devices

from backpack.hessianfree.ggnvp import ggn_vector_product_from_plist
from backpack.hessianfree.hvp import hessian_vector_product
from backpack.utils.convert_parameters import vector_to_parameter_list
from numpy import allclose
from pytest import mark
from torch import cat, device, from_numpy, manual_seed, rand
from torch.nn import CrossEntropyLoss, Linear, ReLU, Sequential
from torch.nn.utils.convert_parameters import parameters_to_vector

from vivit.hessianfree import GGNLinearOperator, HessianLinearOperator

DEVICES = get_available_devices()
DEVICES_IDS = [f"dev={d}" for d in DEVICES]

REDUCTIONS = ["mean", "sum"]
REDUCTIONS_IDS = [f"reduction={r}" for r in REDUCTIONS]


@mark.parametrize("reduction", REDUCTIONS, ids=REDUCTIONS_IDS)
@mark.parametrize("dev", DEVICES, ids=DEVICES_IDS)
def test_HessianLinearOperator_and_GGNLinearOperator(reduction: str, dev: device):
    """Test correctness of HVP and GGNVP in LinearOperator interface.

    Args:
        dev: Device where computations are carried out.
        reduction: Which reduction method to use in the loss.
    """
    manual_seed(0)

    N, D_in, H, C = 3, 10, 5, 2
    num_batches = 4

    model = Sequential(Linear(D_in, H), ReLU(), Linear(H, C)).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in params)
    loss_func = CrossEntropyLoss(reduction=reduction).to(dev)

    # data set with some mini-batches
    data = []
    for _ in range(num_batches):
        X, y = rand(N, D_in, device=dev), classification_targets((N,), C).to(dev)
        data.append((X, y))

    X_merged, y_merged = cat([X for (X, _) in data]), cat([y for (_, y) in data])
    data_merged = [(X_merged, y_merged)]

    v = rand(num_params).numpy()
    v_torch = [vec.to(dev) for vec in vector_to_parameter_list(from_numpy(v), params)]

    # Hessian
    # check correct normalization when looping over data
    H1v = HessianLinearOperator(model, loss_func, data, dev) @ v
    H2v = HessianLinearOperator(model, loss_func, data_merged, dev) @ v

    assert allclose(H1v, H2v)

    # compare with PyTorch
    loss = loss_func(model(X_merged), y_merged)
    H3v = (
        parameters_to_vector(hessian_vector_product(loss, params, v_torch))
        .cpu()
        .numpy()
    )

    assert allclose(H1v, H3v)

    # GGN
    # check correct normalization when looping over data
    G1v = GGNLinearOperator(model, loss_func, data, dev) @ v
    G2v = GGNLinearOperator(model, loss_func, data_merged, dev) @ v

    assert allclose(G1v, G2v)

    # compare with PyTorch
    output = model(X_merged)
    loss = loss_func(output, y_merged)
    G3v = (
        parameters_to_vector(
            ggn_vector_product_from_plist(loss, output, params, v_torch)
        )
        .cpu()
        .numpy()
    )

    assert allclose(G1v, G3v)
