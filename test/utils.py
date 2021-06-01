"""Utility functions to test `vivit`."""

import numpy as np
import torch


def get_available_devices():
    """Return CPU and, if present, GPU device.

    Returns:
        [torch.device]: Available devices for `torch`.
    """
    devices = [torch.device("cpu")]

    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    return devices


def classification_targets(size, num_classes):
    """Create random targets for classes 0, ..., `num_classes - 1`."""
    return torch.randint(size=size, low=0, high=num_classes)


def regression_targets(size):
    """Create random targets for regression."""
    return torch.rand(size=size)


# copied from
#    https://github.com/f-dangel/backpack/blob/development/test/automated_test.py#L60-L91  # noqa: B950

atol = 1e-8
rtol = 1e-5


def report_nonclose_values(x, y, atol=atol, rtol=rtol):
    x_numpy = x.data.cpu().numpy().flatten()
    y_numpy = y.data.cpu().numpy().flatten()

    close = np.isclose(x_numpy, y_numpy, atol=atol, rtol=rtol)
    where_not_close = np.argwhere(np.logical_not(close))
    for idx in where_not_close:
        x, y = x_numpy[idx], y_numpy[idx]
        print("{} versus {}. Ratio of {}".format(x, y, y / x))


def check_sizes_and_values(*plists, atol=atol, rtol=rtol):
    check_sizes(*plists)
    list1, list2 = plists
    check_values(list1, list2, atol=atol, rtol=rtol)


def check_sizes(*plists):
    for i in range(len(plists) - 1):
        assert len(plists[i]) == len(plists[i + 1])

    for params in zip(*plists):
        for i in range(len(params) - 1):
            assert params[i].size() == params[i + 1].size()


def check_values(list1, list2, atol=atol, rtol=rtol):
    for i, (g1, g2) in enumerate(zip(list1, list2)):
        print(i)
        print(g1.size())
        report_nonclose_values(g1, g2, atol=atol, rtol=rtol)
        assert torch.allclose(g1, g2, atol=atol, rtol=rtol)
