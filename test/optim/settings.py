"""Directional derivatives assume ``reduction='mean'``."""

from test.problem import make_test_problems
from test.settings import SETTINGS

PROBLEMS = make_test_problems(SETTINGS)
IDS = [problem.make_id() for problem in PROBLEMS]

PROBLEMS_REDUCTION_MEAN = []
IDS_REDUCTION_MEAN = []

for problem, id_str in zip(PROBLEMS, IDS):
    if problem.reduction_string() == "mean":
        PROBLEMS_REDUCTION_MEAN.append(problem)
        IDS_REDUCTION_MEAN.append(id_str)


def make_criterion(k, must_exceed=1e-5):
    """Create criterion function that keeps at most ``k`` top eigenvalues.

    All eigenvalues must exceed ``must_exceed``.
    """

    def criterion(evals):
        """
        Args:
            evals (torch.Tensor): Eigenvalues.

        Returns:
            [int]: Indices of two leading eigenvalues.
        """
        num_evals = len(evals)

        if num_evals <= k:
            shift = 0
            candidates = evals
        else:
            shift = num_evals - 1 - k
            candidates = evals[shift:]

        return [idx + shift for idx, ev in enumerate(candidates) if ev > must_exceed]

    return criterion


TOP_K = [1, 10]
TOP_K_IDS = [f"top_k={k}" for k in TOP_K]
TOP_K = [make_criterion(k) for k in TOP_K]

SUBSAMPLINGS_DIRECTIONS = [None, [0, 0, 1, 0, 1]]
SUBSAMPLINGS_DIRECTIONS_IDS = [
    f"subsampling_directions={sub}" for sub in SUBSAMPLINGS_DIRECTIONS
]

SUBSAMPLINGS_FIRST = [None, [0, 0, 1, 0, 1]]
SUBSAMPLINGS_FIRST_IDS = [f"subsampling_first={sub}" for sub in SUBSAMPLINGS_FIRST]

SUBSAMPLINGS_SECOND = [None, [0, 0, 1, 0, 1]]
SUBSAMPLINGS_SECOND_IDS = [f"subsampling_second={sub}" for sub in SUBSAMPLINGS_SECOND]

PARAM_BLOCKS_FN = []
PARAM_BLOCKS_FN_IDS = []


def one_group(parameters):
    """All parameters in all group."""
    return [{"params": list(parameters)}]


PARAM_BLOCKS_FN.append(one_group)
PARAM_BLOCKS_FN_IDS.append("param_groups=one")


def per_param(parameters):
    """One parameter group for each parameter. Only group last two parameters.

    Grouping the last two parameters is a fix to avoid degenerate eigenspaces
    (which will then result in arbitrary directions and differing directional
    derivatives). Consider for instance a last linear layer in a neural net
    with ``MSELoss``. Then, the GGN w.r.t. only the last bias is proportional
    to the identity matrix, hence its eigenspace is degenerate.
    """
    parameters = list(parameters)
    num_params = len(parameters)

    if num_params <= 2:
        return one_group(parameters)
    else:
        return [{"params": list(parameters)[-2:]}] + [
            {"params": [p]} for p in list(parameters)[: num_params - 2]
        ]


PARAM_BLOCKS_FN.append(per_param)
PARAM_BLOCKS_FN_IDS.append("param_groups=per_param")


def weights_and_biases(parameters):
    """Group weights in one, biases in other group."""
    parameters = list(parameters)

    def is_bias(param):
        return param.dim() == 1

    weights = {"params": [p for p in parameters if is_bias(p)]}
    biases = {"params": [p for p in parameters if not is_bias(p)]}

    if len(biases["params"]) == 1:
        raise ValueError(
            "The GGN w.r.t. to a single bias is known to exhibit"
            + " eigen-space degeneracies which lead to failing tests."
            + " Use a network with multiple biases for this test."
        )

    return [weights, biases]


PARAM_BLOCKS_FN.append(weights_and_biases)
PARAM_BLOCKS_FN_IDS.append("param_groups=weights_and_biases")


def insert_criterion(param_groups, criterion):
    """Add 'criterion' entry for each parameter group."""
    criterion_entry = {"criterion": criterion}
    for group in param_groups:
        group.update(criterion_entry)
