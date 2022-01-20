from test.utils import (
    classification_targets,
    initialize_training_false_recursive,
    regression_targets,
)

from backpack.custom_module.branching import Parallel
from backpack.custom_module.pad import Pad
from backpack.custom_module.slicing import Slicing
from torch import rand
from torch.nn import (
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
    Conv2d,
    CrossEntropyLoss,
    Flatten,
    Identity,
    Linear,
    MaxPool2d,
    MSELoss,
    ReLU,
    Sequential,
    Sigmoid,
    Tanh,
)

SETTINGS = [
    # classification
    {
        "input_fn": lambda: rand(3, 7),
        "module_fn": lambda: Sequential(Linear(7, 6), Linear(6, 5), ReLU()),
        "loss_function_fn": lambda: CrossEntropyLoss(reduction="mean"),
        "target_fn": lambda: classification_targets((3,), 5),
    },
    {
        "input_fn": lambda: rand(3, 7),
        "module_fn": lambda: Sequential(Linear(7, 6), ReLU(), Linear(6, 5)),
        "loss_function_fn": lambda: CrossEntropyLoss(reduction="sum"),
        "target_fn": lambda: classification_targets((3,), 5),
    },
    {
        "input_fn": lambda: rand(4, 3, 6, 6),
        "module_fn": lambda: Sequential(
            Conv2d(3, 2, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2),
            Flatten(),
            Linear(8, 5),
            Sigmoid(),
        ),
        "loss_function_fn": lambda: CrossEntropyLoss(reduction="mean"),
        "target_fn": lambda: classification_targets((4,), 5),
    },
    # Regression
    {
        "input_fn": lambda: rand(3, 7),
        "module_fn": lambda: Sequential(
            Linear(7, 6), Sigmoid(), Linear(6, 5), Sigmoid()
        ),
        "loss_function_fn": lambda: MSELoss(reduction="mean"),
        "target_fn": lambda: regression_targets((3, 5)),
    },
]

# additional dimensions
SETTINGS += [
    # nn.Linear with one additional dimension
    {
        "input_fn": lambda: rand(3, 4, 5),
        "module_fn": lambda: Sequential(
            Linear(5, 3),
            Sigmoid(),
            Linear(3, 2),
            Sigmoid(),
            Flatten(),  # flatten input to MSELoss
        ),
        "loss_function_fn": lambda: MSELoss(reduction="mean"),
        "target_fn": lambda: regression_targets((3, 4 * 2)),
        # MSELoss currently only supports 2d inputs
        "id_prefix": "one-additional",
    },
    # nn.Linear with two additional dimensions
    {
        "input_fn": lambda: rand(3, 4, 2, 5),
        "module_fn": lambda: Sequential(
            Linear(5, 3),
            Tanh(),
            Linear(3, 2),
            Tanh(),
            Flatten(),  # flatten input to MSELoss
        ),
        "loss_function_fn": lambda: MSELoss(reduction="mean"),
        "target_fn": lambda: regression_targets((3, 4 * 2 * 2)),
        # MSELoss currently only supports 2d inputs
        "id_prefix": "two-additional",
    },
    # nn.Linear with three additional dimensions
    {
        "input_fn": lambda: rand(3, 4, 2, 3, 5),
        "module_fn": lambda: Sequential(
            Linear(5, 3),
            ReLU(),
            Linear(3, 2),
            Sigmoid(),
            Flatten(),  # flatten input to MSELoss
        ),
        "loss_function_fn": lambda: MSELoss(reduction="mean"),
        # MSELoss currently only supports 2d inputs
        "target_fn": lambda: regression_targets((3, 4 * 2 * 3 * 2)),
        "id_prefix": "three-additional",
    },
]

##################################################################
#                      BatchNorm settings                        #
##################################################################
SETTINGS += [
    {
        "input_fn": lambda: rand(2, 3, 4),
        "module_fn": lambda: initialize_training_false_recursive(
            Sequential(BatchNorm1d(num_features=3), Flatten(), Linear(12, 3), Sigmoid())
        ),
        "loss_function_fn": lambda: MSELoss(),
        "target_fn": lambda: regression_targets((2, 3)),
    },
    {
        "input_fn": lambda: rand(3, 2, 4, 3),
        "module_fn": lambda: initialize_training_false_recursive(
            Sequential(BatchNorm2d(num_features=2), Flatten(), Linear(24, 3))
        ),
        "loss_function_fn": lambda: CrossEntropyLoss(),
        "target_fn": lambda: classification_targets((3,), 3),
    },
    {
        "input_fn": lambda: rand(3, 3, 4, 1, 2),
        "module_fn": lambda: initialize_training_false_recursive(
            Sequential(BatchNorm3d(num_features=3), Flatten(), Linear(24, 3))
        ),
        "loss_function_fn": lambda: CrossEntropyLoss(),
        "target_fn": lambda: classification_targets((3,), 3),
    },
    {
        "input_fn": lambda: rand(3, 3, 4, 1, 2),
        "module_fn": lambda: initialize_training_false_recursive(
            Sequential(
                Linear(2, 3),
                BatchNorm3d(num_features=3),
                Sigmoid(),
                Flatten(),
            )
        ),
        "loss_function_fn": lambda: MSELoss(),
        "target_fn": lambda: regression_targets((3, 4 * 1 * 3 * 3)),
    },
]

###############################################################################
#                               Branched models                               #
###############################################################################
SETTINGS += [
    {
        "input_fn": lambda: rand(3, 7),
        "module_fn": lambda: Sequential(
            Linear(7, 4),
            ReLU(),
            Pad((1, 1), mode="constant", value=0.5),
            # skip connection
            Parallel(
                Identity(),
                Sequential(Linear(6, 8), Slicing((slice(None), slice(0, 6)))),
            ),
            # end of skip connection
            Sigmoid(),
            Linear(6, 4),
        ),
        "loss_function_fn": lambda: CrossEntropyLoss(),
        "target_fn": lambda: classification_targets((3,), 4),
        "id_prefix": "branching-linear-slicing-pad",
    },
]
