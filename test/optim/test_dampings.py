"""Integration tests for damping policies."""

from test.utils import get_available_devices

import pytest
import torch

from vivit.optim.damping import BootstrapDamping

DAMPING_GRIDS = [torch.logspace(-3, 2, 150)]
DAMPING_GRIDS_IDS = ["damping_grid=torch.logspace(-3, 2, 150)"]

PERCENTILES = [95]
PERCENTILES_IDS = [f"percentile={percentile}" for percentile in PERCENTILES]

NUM_RESAMPLES = [100]
NUM_RESAMPLES_IDS = [f"num_resample={num_resample}" for num_resample in NUM_RESAMPLES]

DEVICES = get_available_devices()
DEVICES_IDS = [f"device={device}" for device in DEVICES]

SEED_VALS = [0, 1, 42]
SEED_VALS_IDS = [f"seed_val={seed_val}" for seed_val in SEED_VALS]


@pytest.mark.parametrize("damping_grid", DAMPING_GRIDS, ids=DAMPING_GRIDS_IDS)
@pytest.mark.parametrize("percentile", PERCENTILES, ids=PERCENTILES_IDS)
@pytest.mark.parametrize("num_resamples", NUM_RESAMPLES, ids=NUM_RESAMPLES_IDS)
@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_IDS)
@pytest.mark.parametrize("seed_val", SEED_VALS, ids=SEED_VALS_IDS)
def test_bootstrap_damping(damping_grid, percentile, num_resamples, device, seed_val):

    # Make deterministic
    torch.manual_seed(seed_val)

    # Define Setting
    N_1 = 5  # Number of 1st derivative samples for for each direction
    N_2 = 6  # Number of 2nd derivative samples for for each direction
    D = 3  # Number of directions

    # Sample 1st and 2nd order derivatives for each direction
    first_lower = -0.5
    first_upper = 2.0
    first_derivs = (first_upper - first_lower) * torch.rand(N_1, D) + first_lower
    first_derivs = first_derivs.to(device)

    second_lower = 1.0
    second_upper = 1.5
    second_derivs = (second_upper - second_lower) * torch.rand(N_2, D) + second_lower
    second_derivs = second_derivs.to(device)

    # Compute dampings
    damping = BootstrapDamping(damping_grid, num_resamples, percentile)
    _ = damping(first_derivs, second_derivs)
