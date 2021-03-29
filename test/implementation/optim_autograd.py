"""Autograd implementation of operations in ``lowrank.optim``."""

from test.implementation.autograd import AutogradExtensions


class AutogradOptimExtensions(AutogradExtensions):
    def gammas_ggn(self, k):
        """First-order directional derivatives via ``lowrank.optim.computations``.

        Use top-k eigenvectors as directions.
        """
        assert k == 1, "Currently only supports k=1"
        return super().gammas_ggn(0.0)

    def lambdas_ggn(self, k):
        """First-order directional derivatives via ``lowrank.optim.computations``.

        Use top-k eigenvectors as directions.
        """
        assert k == 1, "Currently only supports k=1"
        return super().lambdas_ggn(0.0)
