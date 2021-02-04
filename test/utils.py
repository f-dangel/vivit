"""Utility functions to test `lowrank`."""

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
