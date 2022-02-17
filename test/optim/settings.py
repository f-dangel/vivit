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
CRITERIA_IDS = [f"criterion=top_{k}" for k in TOP_K]
CRITERIA = [make_criterion(k) for k in TOP_K]

SUBSAMPLINGS_GGN = [None, [0, 0, 1, 0, 1]]
SUBSAMPLINGS_GGN_IDS = [f"subsampling_ggn={sub}" for sub in SUBSAMPLINGS_GGN]

SUBSAMPLINGS_GRAD = [None, [0, 0, 1, 0, 1]]
SUBSAMPLINGS_GRAD_IDS = [f"subsampling_grad={sub}" for sub in SUBSAMPLINGS_GRAD]

PARAM_BLOCKS_FN = []
PARAM_BLOCKS_FN_IDS = []


def one_group(named_parameters, criterion):
    """All parameters in all group."""
    return [{"params": [p for (_, p) in named_parameters], "criterion": criterion}]


PARAM_BLOCKS_FN.append(one_group)
PARAM_BLOCKS_FN_IDS.append("param_groups=one")


def weights_and_biases(parameters, criterion):
    """Group weights in one, biases in other group."""
    parameters = list(parameters)

    def is_bias(name, param):
        if "bias" in name:
            return True
        elif "weight" in name:
            return False
        else:
            return param.dim() == 1

    weights = {
        "params": [p for (n, p) in parameters if is_bias(n, p)],
        "criterion": criterion,
    }
    biases = {
        "params": [p for (n, p) in parameters if not is_bias(n, p)],
        "criterion": criterion,
    }

    if len(biases["params"]) == 1:
        raise ValueError(
            "The GGN w.r.t. to a single bias is known to exhibit"
            + " eigen-space degeneracies which lead to failing tests."
            + " Use a network with multiple biases for this test."
        )

    return [weights, biases]


PARAM_BLOCKS_FN.append(weights_and_biases)
PARAM_BLOCKS_FN_IDS.append("param_groups=weights_and_biases")
