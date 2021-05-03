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
