"""PyTorch optimizer with damped Newton updates."""

import torch


class DampedNewton(torch.optim.Optimizer):
    """
    Newton optimizer damped via bootstrapped 1st- and 2nd-order directional derivatives.

    Attributes:
        SAVEFIELD (str): Field under which the damped Newton update is stored in a
            parameter.
    """

    SAVEFIELD = "damped_newton_step"

    def __init__(self, parameters, damping, computations, criterion):
        """Initialize the optimizer, specifying the damping damping and sample split.

        Args:
            parameters ([torch.nn.Parameters]): List of parameters to be trained.
            damping (vivit.optim.damping.BaseDamping): Policy for selecting
                dampings along a direction from first- and second- order directional
                derivatives.
            computations (vivit.optim.computations.BaseComputations): Assignment of
                mini-batch samples to the different computational tasks (finding
                directions, computing first- and second-order derivatives along them).
            criterion (callable): Maps eigenvalues to indices of eigenvalues that are
                kept as directions. Assumes eigenvalues to be sorted in ascending order.
        """
        defaults = {"criterion": criterion}
        super().__init__(parameters, defaults=defaults)

        self._damping = damping
        self._computations = computations

    def get_extensions(self):
        """Return the required extensions for BackPACK.

        They can directly be placed inside a ``with backpack(...)`` context.

        Returns:
            [backpack.extensions.backprop_extension.BackpropExtension]: List of
                extensions that can be handed into a ``with backpack(...)`` context.
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
        keep_deltas=False,
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
            keep_deltas (bool, optional): Keep directional dampings under group id in
                ``self._deltas``. Default: ``False``.
            keep_newton_step (bool, optional): Keep damped Newton step under group id
                in ``self._newton_step``. Default: ``False``.
            keep_backpack_buffers (bool, optional): Keep buffers from used BackPACK
                extensions during backpropagation. Default: ``False``.

        Returns:
            callable or None: Hook function that can be handed into a
                ``with backpack(...)`` context. ``None`` signifies no action will be
                performed.
        """
        return self._computations.get_extension_hook(
            self.param_groups,
            self._damping,
            self.SAVEFIELD,
            keep_gram_mat=keep_gram_mat,
            keep_gram_evals=keep_gram_evals,
            keep_gram_evecs=keep_gram_evecs,
            keep_gammas=keep_gammas,
            keep_lambdas=keep_lambdas,
            keep_batch_size=keep_batch_size,
            keep_deltas=keep_deltas,
            keep_newton_step=keep_newton_step,
            keep_backpack_buffers=keep_backpack_buffers,
        )

    def step(self, closure=None, lr=1.0):
        """Apply damped Newton step to all parameters.

        Modifies the ``.data`` entry of each parameter.

        Args:
            closure (callable): Function to reevaluate the model and return the loss.
            lr (float): Learning rate. The Newton step is scaled by this value before
                it is applied to the network parameters. The default value is ``1.0``.
        """
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
