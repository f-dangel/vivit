"""BackPACK implementation of operations used in ``vivit.optim``."""

from test.implementation.backpack import BackpackExtensions
from typing import Any, Dict, List, Optional

from backpack import backpack
from torch import Tensor

from vivit.optim import GramComputations
from vivit.optim.computations import BaseComputations
from vivit.optim.damped_newton import DampedNewton
from vivit.optim.damping import _DirectionalCoefficients


class BackpackOptimExtensions(BackpackExtensions):
    def gammas_ggn(
        self, param_groups, subsampling_directions=None, subsampling_first=None
    ):
        """First-order directional derivatives along leading GGN eigenvectors via
        ``vivit.optim.computations``.

        Args:
            param_groups ([dict]): Parameter groups like for ``torch.nn.Optimizer``s.
            subsampling_directions ([int] or None): Indices of samples used to compute
                Newton directions. If ``None``, all samples in the batch will be used.
            subsampling_first ([int], optional): Sample indices used for individual
                gradients.
        """
        computations = GramComputations(
            subsampling_directions=subsampling_directions,
            subsampling_first=subsampling_first,
        )

        _, _, loss = self.problem.forward_pass()

        with backpack(
            *computations.get_extensions(param_groups),
            extension_hook=computations.get_extension_hook(
                param_groups,
                keep_backpack_buffers=False,
                keep_gram_mat=False,
                keep_gram_evecs=False,
                keep_batch_size=False,
                keep_gammas=True,
                keep_lambdas=False,
                keep_gram_evals=False,
            ),
        ):
            loss.backward()

        return [computations._gammas[id(group)] for group in param_groups]

    def lambdas_ggn(
        self, param_groups, subsampling_directions=None, subsampling_second=None
    ):
        """Second-order directional derivatives along leading GGN eigenvectors via
        ``vivit.optim.computations``.

        Args:
            param_groups ([dict]): Parameter groups like for ``torch.nn.Optimizer``s.
            subsampling_directions ([int] or None): Indices of samples used to compute
                Newton directions. If ``None``, all samples in the batch will be used.
            subsampling_second ([int], optional): Sample indices used for individual
                curvature matrices.
        """
        computations = GramComputations(
            subsampling_directions=subsampling_directions,
            subsampling_second=subsampling_second,
        )

        _, _, loss = self.problem.forward_pass()

        with backpack(
            *computations.get_extensions(param_groups),
            extension_hook=computations.get_extension_hook(
                param_groups,
                keep_backpack_buffers=False,
                keep_gram_mat=False,
                keep_gram_evecs=False,
                keep_batch_size=False,
                keep_gammas=False,
                keep_lambdas=True,
                keep_gram_evals=False,
            ),
        ):
            loss.backward()

        return [computations._lambdas[id(group)] for group in param_groups]

    def newton_step(
        self,
        param_groups,
        damping,
        subsampling_directions=None,
        subsampling_first=None,
        subsampling_second=None,
    ):
        """Directionally-damped Newton step along the top-k GGN eigenvectors.

        Args:
            param_groups ([dict]): Parameter groups like for ``torch.nn.Optimizer``s.
            damping (vivit.optim.damping._Damping): Policy for selecting
                dampings along a direction from first- and second- order directional
                derivatives.
            subsampling_directions ([int] or None): Indices of samples used to compute
                Newton directions. If ``None``, all samples in the batch will be used.
            subsampling_first ([int], optional): Sample indices used for individual
                gradients.
            subsampling_second ([int], optional): Sample indices used for individual
                curvature matrices.
        """
        computations = BaseComputations(
            subsampling_directions=subsampling_directions,
            subsampling_first=subsampling_first,
            subsampling_second=subsampling_second,
        )

        _, _, loss = self.problem.forward_pass()

        savefield = "test_newton_step"

        with backpack(
            *computations.get_extensions(param_groups),
            extension_hook=computations.get_extension_hook(
                param_groups,
                damping,
                savefield,
                keep_gram_mat=False,
                keep_gram_evals=False,
                keep_gram_evecs=False,
                keep_gammas=False,
                keep_lambdas=False,
                keep_batch_size=False,
                keep_coefficients=False,
                keep_newton_step=False,
                keep_backpack_buffers=False,
            ),
        ):
            loss.backward()

        newton_step = [
            [getattr(param, savefield) for param in group["params"]]
            for group in param_groups
        ]

        return newton_step

    def optim_newton_step(
        self,
        param_groups: List[Dict[str, Any]],
        damping: _DirectionalCoefficients,
        subsampling_directions: Optional[List[int]] = None,
        subsampling_first: Optional[List[int]] = None,
        subsampling_second: Optional[List[int]] = None,
        use_closure: bool = False,
    ):
        """Directionally-damped Newton step along the top-k GGN eigenvectors.

        Uses the ``DampedNewton`` optimizer to compute Newton steps.

        Args:
            param_groups: Parameter groups like for ``torch.nn.Optimizer``s.
            damping: Computes Newton coefficients from first- and second- order
                directional derivatives.
            subsampling_directions: Indices of samples used to compute
                Newton directions. If ``None``, all samples in the batch will be used.
            subsampling_first: Sample indices used for individual gradients.
            subsampling_second: Sample indices used for individual curvature matrices.
            use_closure: Whether to use a closure in the optimizer. Default: ``False``.
        """
        computations = BaseComputations(
            subsampling_directions=subsampling_directions,
            subsampling_first=subsampling_first,
            subsampling_second=subsampling_second,
        )

        opt = DampedNewton(
            param_groups,
            coefficients=damping,
            computations=computations,
            criterion=None,
        )

        savefield = "test_newton_step"
        DampedNewton.SAVEFIELD = savefield

        if use_closure:

            def closure() -> Tensor:
                """Evaluate the loss on a fixed mini-batch.

                Returns:
                    Mini-batch loss.
                """
                _, _, loss = self.problem.forward_pass()
                return loss

            opt.step(closure=closure)

        else:
            _, _, loss = self.problem.forward_pass()

            with backpack(
                *opt.get_extensions(),
                extension_hook=opt.get_extension_hook(
                    keep_gram_mat=False,
                    keep_gram_evals=False,
                    keep_gram_evecs=False,
                    keep_gammas=False,
                    keep_lambdas=False,
                    keep_batch_size=False,
                    keep_coefficients=False,
                    keep_newton_step=False,
                    keep_backpack_buffers=False,
                ),
            ):
                loss.backward()
                opt.step()

        newton_step = [
            [getattr(param, savefield) for param in group["params"]]
            for group in param_groups
        ]

        return newton_step
