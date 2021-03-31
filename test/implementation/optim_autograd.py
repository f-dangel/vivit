"""Autograd implementation of operations in ``lowrank.optim``."""

from test.implementation.autograd import AutogradExtensions


class AutogradOptimExtensions(AutogradExtensions):
    """Autograd implementation of optimizer functionality with similar API."""

    def gammas_ggn(self, top_k):
        """First-order directional derivatives along the top-k GGN eigenvectors.

        Args:
            top_k (int): Number of leading eigenvectors used as directions. Will be
                clipped to ``[1, max]`` with ``max`` the maximum number of nontrivial
                eigenvalues.
        """
        return super().gammas_ggn(top_k)

    def lambdas_ggn(self, top_k):
        """Second-order directional derivatives along the top-k GGN eigenvectors.

        Args:
            top_k (int): Number of leading eigenvectors used as directions. Will be
                clipped to ``[1, max]`` with ``max`` the maximum number of nontrivial
                eigenvalues.
        """
        return super().lambdas_ggn(top_k)
