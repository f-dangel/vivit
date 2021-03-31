"""Autograd implementation of operations in ``lowrank.optim``."""

from test.implementation.autograd import AutogradExtensions


class AutogradOptimExtensions(AutogradExtensions):
    """Autograd implementation of optimizer functionality with similar API."""

    def gammas_ggn(self, top_k, subsampling_directions=None, subsampling_first=None):
        """First-order directional derivatives along the top-k GGN eigenvectors.

        Args:
            top_k (int): Number of leading eigenvectors used as directions. Will be
                clipped to ``[1, max]`` with ``max`` the maximum number of nontrivial
                eigenvalues.
            subsampling_directions ([int] or None): Indices of samples used to compute
                Newton directions. If ``None``, all samples in the batch will be used.
            subsampling_first ([int], optional): Sample indices used for individual
                gradients.
        """
        return super().gammas_ggn(
            top_k,
            ggn_subsampling=subsampling_directions,
            grad_subsampling=subsampling_first,
        )

    def lambdas_ggn(self, top_k, subsampling_directions=None, subsampling_second=None):
        """Second-order directional derivatives along the top-k GGN eigenvectors.

        Args:
            top_k (int): Number of leading eigenvectors used as directions. Will be
                clipped to ``[1, max]`` with ``max`` the maximum number of nontrivial
                eigenvalues.
            subsampling_directions ([int] or None): Indices of samples used to compute
                Newton directions. If ``None``, all samples in the batch will be used.
            subsampling_second ([int], optional): Sample indices used for individual
                curvature matrices.
        """
        return super().lambdas_ggn(
            top_k,
            ggn_subsampling=subsampling_directions,
            lambda_subsampling=subsampling_second,
        )
