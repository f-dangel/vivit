"""Autograd implementation of operations in ``vivit.optim``."""

from test.implementation.autograd import AutogradExtensions


class AutogradOptimExtensions(AutogradExtensions):
    """Autograd implementation of optimizer functionality with similar API."""

    def directional_derivatives(
        self, param_groups, subsampling_grad=None, subsampling_ggn=None
    ):
        gammas = self.gammas_ggn(
            param_groups,
            grad_subsampling=subsampling_grad,
            ggn_subsampling=subsampling_ggn,
            directions=False,
        )

        lambdas = self.lambdas_ggn(
            param_groups,
            ggn_subsampling=subsampling_ggn,
            lambda_subsampling=subsampling_ggn,
        )

        return gammas, lambdas
