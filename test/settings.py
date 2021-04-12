from test.utils import classification_targets, regression_targets

import torch

SETTINGS = [
    # classification
    {
        "input_fn": lambda: torch.rand(3, 10),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.Linear(10, 7), torch.nn.Linear(7, 5)
        ),
        "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(reduction="mean"),
        "target_fn": lambda: classification_targets((3,), 5),
    },
    {
        "input_fn": lambda: torch.rand(3, 10),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.Linear(10, 7), torch.nn.ReLU(), torch.nn.Linear(7, 5)
        ),
        "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(reduction="sum"),
        "target_fn": lambda: classification_targets((3,), 5),
    },
    # Regression
    {
        "input_fn": lambda: torch.rand(3, 10),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.Linear(10, 7), torch.nn.Sigmoid(), torch.nn.Linear(7, 5)
        ),
        "loss_function_fn": lambda: torch.nn.MSELoss(reduction="mean"),
        "target_fn": lambda: regression_targets((3, 5)),
    },
]

# additional dimensions
SETTINGS += [
    # nn.Linear with one additional dimension
    {
        "input_fn": lambda: torch.rand(3, 4, 5),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.Linear(5, 3),
            torch.nn.Sigmoid(),
            torch.nn.Linear(3, 2),
            torch.nn.Flatten(),  # flatten input to MSELoss
        ),
        "loss_function_fn": lambda: torch.nn.MSELoss(reduction="mean"),
        "target_fn": lambda: regression_targets((3, 4 * 2)),
        # MSELoss currently only supports 2d inputs
        "id_prefix": "one-additional",
    },
    # nn.Linear with two additional dimensions
    {
        "input_fn": lambda: torch.rand(3, 4, 2, 5),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.Linear(5, 3),
            torch.nn.Tanh(),
            torch.nn.Linear(3, 2),
            torch.nn.Flatten(),  # flatten input to MSELoss
        ),
        "loss_function_fn": lambda: torch.nn.MSELoss(reduction="mean"),
        "target_fn": lambda: regression_targets((3, 4 * 2 * 2)),
        # MSELoss currently only supports 2d inputs
        "id_prefix": "two-additional",
    },
    # nn.Linear with three additional dimensions
    {
        "input_fn": lambda: torch.rand(3, 4, 2, 3, 5),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.Linear(5, 3),
            torch.nn.ReLU(),
            torch.nn.Linear(3, 2),
            torch.nn.Flatten(),  # flatten input to MSELoss
        ),
        "loss_function_fn": lambda: torch.nn.MSELoss(reduction="mean"),
        # MSELoss currently only supports 2d inputs
        "target_fn": lambda: regression_targets((3, 4 * 2 * 3 * 2)),
        "id_prefix": "three-additional",
    },
]
