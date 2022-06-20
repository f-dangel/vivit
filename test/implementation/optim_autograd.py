"""Autograd implementation of operations in ``vivit.optim``."""

from test.implementation.autograd import AutogradExtensions

from torch import einsum


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

    def directional_damped_newton(
        self, param_groups, subsampling_grad=None, subsampling_ggn=None
    ):

        group_gammas, group_evecs = self.gammas_ggn(
            param_groups,
            ggn_subsampling=subsampling_ggn,
            grad_subsampling=subsampling_grad,
            directions=True,
        )
        group_lambdas = self.lambdas_ggn(
            param_groups,
            ggn_subsampling=subsampling_ggn,
            lambda_subsampling=subsampling_ggn,
        )

        newton_steps = []

        for group, gammas, lambdas, evecs in zip(
            param_groups, group_gammas, group_lambdas, group_evecs
        ):
            dummy_gram_evecs = None
            dummy_evals = None
            deltas = group["damping"](dummy_gram_evecs, dummy_evals, gammas, lambdas)

            coefficients = -gammas.mean(0) / (lambdas.mean(0) + deltas)
            newton = einsum("id,d->i", evecs, coefficients)

            newton_steps.append(newton)

        return newton_steps
