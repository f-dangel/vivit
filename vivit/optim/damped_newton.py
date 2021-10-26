"""PyTorch optimizer with damped Newton updates."""

from typing import Callable, List

from backpack import backpack
from backpack.extensions.backprop_extension import BackpropExtension
from torch import Tensor
from torch.optim import Optimizer

from vivit.optim.computations import BaseComputations
from vivit.optim.damping import _DirectionalCoefficients


class DampedNewton(Optimizer):
    """
    Newton optimizer damped via bootstrapped 1st- and 2nd-order directional derivatives.

    Attributes:
        SAVEFIELD: Field under which the damped Newton update is stored in a parameter.
    """

    SAVEFIELD: str = "damped_newton_step"

    def __init__(
        self,
        parameters: List[Tensor],
        coefficients: _DirectionalCoefficients,
        computations: BaseComputations,
        criterion: Callable[[Tensor], List[int]],
    ):
        """Initialize the optimizer, specifying the damping damping and sample split.

        Args:
            parameters: List of parameters to be trained.
            coefficients: Policy for computing Newton step coefficients from first-
                and second- order directional derivatives.
            computations: Assignment of mini-batch samples to the different
                computational tasks (finding directions, computing first- and
                second-order derivatives along them).
            criterion: Maps eigenvalues to indices of eigenvalues that are
                kept as directions. Assumes eigenvalues to be sorted in ascending order.
        """
        defaults = {"criterion": criterion}
        super().__init__(parameters, defaults=defaults)

        self._coefficients = coefficients
        self._computations = computations

    def get_extensions(self) -> List[BackpropExtension]:
        """Return the required extensions for BackPACK.

        They can directly be placed inside a ``with backpack(...)`` context.

        Returns:
            List of extensions that can be handed into a ``with backpack(...)`` context.
        """
        return self._computations.get_extensions(self.param_groups)

    def get_extension_hook(
        self,
        keep_gram_mat=False,
        keep_gram_evals=False,
        keep_gram_evecs=False,
        keep_gammas=False,
        keep_lambdas=False,
        keep_batch_size=False,
        keep_coefficients: bool = False,
        keep_newton_step=False,
        keep_backpack_buffers=False,
    ):
        """Return hook to be executed right after a BackPACK extension during backprop.

        Args:
            keep_gram_mat (bool, optional): Keep buffers for Gram matrix under group id
                in ``self._computations._gram_computation._gram_mat``.
                Default: ``False``
            keep_gram_evals (bool, optional): Keep buffers for filtered Gram matrix
                eigenvalues under group id in
                ``self._computations._gram_computation._gram_evals``. Default: ``False``
            keep_gram_evecs (bool, optional): Keep buffers for filtered Gram matrix
                eigenvectors under group id in
                ``self._computations._gram_computation._gram_evecs``. Default: ``False``
            keep_gammas (bool, optional): Keep buffers for first-order directional
                derivatives under group id in
                ``self._computations._gram_computation._gammas``. Default: ``False``
            keep_lambdas (bool, optional): Keep buffers for second-order directional
                derivatives under group id in
                ``self._computations._gram_computation._lambdas``. Default: ``False``
            keep_batch_size (bool, optional): Keep batch size for under group id
                in ``self._computations._gram_computation._lambdas``. Default: ``False``
            keep_coefficients: Keep Newton step coefficients under group id in
                ``self._computations._coefficients``. Default: ``False``.
            keep_newton_step (bool, optional): Keep damped Newton step under group id
                in ``self._computations._newton_step``. Default: ``False``.
            keep_backpack_buffers (bool, optional): Keep buffers from used BackPACK
                extensions during backpropagation. Default: ``False``.

        Returns:
            callable or None: Hook function that can be handed into a
                ``with backpack(...)`` context. ``None`` signifies no action will be
                performed.
        """
        return self._computations.get_extension_hook(
            self.param_groups,
            self._coefficients,
            self.SAVEFIELD,
            keep_gram_mat=keep_gram_mat,
            keep_gram_evals=keep_gram_evals,
            keep_gram_evecs=keep_gram_evecs,
            keep_gammas=keep_gammas,
            keep_lambdas=keep_lambdas,
            keep_batch_size=keep_batch_size,
            keep_coefficients=keep_coefficients,
            keep_newton_step=keep_newton_step,
            keep_backpack_buffers=keep_backpack_buffers,
        )

    def step(
        self,
        closure: Callable[[], Tensor] = None,
        lr: float = 1.0,
        keep_gram_mat: bool = False,
        keep_gram_evals: bool = False,
        keep_gram_evecs: bool = False,
        keep_gammas: bool = False,
        keep_lambdas: bool = False,
        keep_batch_size: bool = False,
        keep_coefficients: bool = False,
        keep_newton_step: bool = False,
        keep_backpack_buffers: bool = False,
    ):
        """Apply damped Newton step to all parameters.

        Modifies the ``.data`` entry of each parameter.

        Args:
            closure: Function to reevaluate the model and return the loss. This
                function should only perform the forward pass, BUT NOT the additional
                steps outlined in https://pytorch.org/docs/stable/optim.html.
            lr: Learning rate. The Newton step is scaled by this value before
                it is applied to the network parameters. The default value is ``1.0``.
            keep_gram_mat: (only relevant if closure us passed) Keep buffers for Gram
                matrix under group id in
                ``self._computations._gram_computation._gram_mat``. Default: ``False``
            keep_gram_evals: (only relevant if closure us passed) Keep buffers for
                filtered Gram matrix eigenvalues under group id in
                ``self._computations._gram_computation._gram_evals``. Default: ``False``
            keep_gram_evecs: (only relevant if closure us passed) Keep buffers for
                filtered Gram matrix eigenvectors under group id in
                ``self._computations._gram_computation._gram_evecs``. Default: ``False``
            keep_gammas: (only relevant if closure us passed) Keep buffers for
                first-order directional derivatives under group id in
                ``self._computations._gram_computation._gammas``. Default: ``False``
            keep_lambdas: (only relevant if closure us passed) Keep buffers for
                second-order directional derivatives under group id in
                ``self._computations._gram_computation._lambdas``. Default: ``False``
            keep_batch_size: (only relevant if closure us passed) Keep batch size for
                under group id in ``self._computations._gram_computation._lambdas``.
                Default: ``False``
            keep_coefficients: Keep Newton step coefficients under group id in
                ``self._computations._coefficients``. Default: ``False``.
            keep_newton_step: (only relevant if closure us passed) Keep damped Newton
                step under group id in ``self._computations._newton_step``.
                Default: ``False``.
            keep_backpack_buffers: (only relevant if closure us passed) Keep buffers
                from used BackPACK extensions during backpropagation. Default:
                ``False``.
        """
        if closure is not None:
            self.zero_grad()
            self.zero_newton()
            loss = closure()
            extensions = self.get_extensions()
            hook = self.get_extension_hook(
                keep_gram_mat=keep_gram_mat,
                keep_gram_evals=keep_gram_evals,
                keep_gram_evecs=keep_gram_evecs,
                keep_gammas=keep_gammas,
                keep_lambdas=keep_lambdas,
                keep_batch_size=keep_batch_size,
                keep_coefficients=keep_coefficients,
                keep_newton_step=keep_newton_step,
                keep_backpack_buffers=keep_backpack_buffers,
            )
            with backpack(*extensions, extension_hook=hook):
                loss.backward()

        for group in self.param_groups:
            self.step_group(group, lr)

    def step_group(self, group, lr=1.0):
        """Apply damped Newton step to a parameter group.

        Modifies the ``.data`` entry of each group parameter.

        Args:
            group (dict): Parameter group. Entry of a ``torch.optim.Optimizer``'s
                ``param_groups`` list.
            lr (float): Learning rate. The Newton step is scaled by this value before
                it is applied to the network parameters. The default value is ``1.0``.
        """
        for param in group["params"]:
            param.data.add_(getattr(param, self.SAVEFIELD), alpha=lr)

    def zero_newton(self):
        """Delete the parameter attributes used to store the Newton steps."""
        for group in self.param_groups:
            for param in group["params"]:
                if hasattr(param, self.SAVEFIELD):
                    delattr(param, self.SAVEFIELD)
