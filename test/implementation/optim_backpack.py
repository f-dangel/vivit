"""BackPACK implementation of operations used in ``vivit.optim``."""

from test.implementation.backpack import BackpackExtensions

from backpack import backpack
from torch import cat

from vivit.optim import (
    DirectionalDampedNewtonComputation,
    DirectionalDerivativesComputation,
)


class BackpackOptimExtensions(BackpackExtensions):
    def directional_derivatives(
        self,
        param_groups,
        subsampling_grad=None,
        subsampling_ggn=None,
        mc_samples_ggn=0,
    ):
        """Compute 1st and 2nd-order directional derivatives along GGN eigenvectors."""
        computations = DirectionalDerivativesComputation(
            subsampling_grad=subsampling_grad,
            subsampling_ggn=subsampling_ggn,
            mc_samples_ggn=mc_samples_ggn,
        )

        _, _, loss = self.problem.forward_pass()

        with backpack(
            *computations.get_extensions(),
            extension_hook=computations.get_extension_hook(param_groups),
        ):
            loss.backward()

        gammas, lambdas = [], []

        for group in param_groups:
            group_gammas, group_lambdas = computations.get_result(group)
            gammas.append(group_gammas)
            lambdas.append(group_lambdas)

        return gammas, lambdas

    def directional_damped_newton(
        self,
        param_groups,
        subsampling_grad=None,
        subsampling_ggn=None,
        mc_samples_ggn=0,
    ):
        computations = DirectionalDampedNewtonComputation(
            subsampling_grad=subsampling_grad,
            subsampling_ggn=subsampling_ggn,
            mc_samples_ggn=mc_samples_ggn,
        )

        _, _, loss = self.problem.forward_pass()

        with backpack(
            *computations.get_extensions(),
            extension_hook=computations.get_extension_hook(param_groups),
        ):
            loss.backward()

        newton_steps = []

        for group in param_groups:
            group_newton_step = computations.get_result(group)
            # flatten and concatenate over parameters in group
            group_newton_step = cat([n.flatten() for n in group_newton_step])
            newton_steps.append(group_newton_step)

        return newton_steps
