"""Interface for computing the individual gradient Gram matrix."""

import torch

from lowrank.utils.hooks import ParameterHook


class GramBatchGrad(ParameterHook):
    """BackPACK extension hook that computes the gradient Gram matrix.

    Can be used as extension hook in ``with backpack(BatchGrad()):``. It is
    obligatory that ``backpack``'s ``BatchGrad`` extension is active.

    Note:
        BackPACK computes individual gradients with a potential factor stemming
        from the loss function's ``reduction`` argument, the result differs by the
        actual gram matrix by a factor of ``1 / N ** 2`` (reduction mean) or by
        ``1 / N`` (reduction sum).

    Args:
        layerwise (bool): Whether layerwise Gram matrices should be kept. Otherwise
            they are discarded to save memory.
        free_individual_gradients (bool) : Whether individual gradients should be freed
            during backpropagation to save memory.
    """

    def __init__(
        self,
        savefield="gram_grad_batch",
        layerwise=False,
        free_individual_gradients=False,
    ):
        super().__init__(savefield)

        self._gram_mat = None
        self._layerwise = layerwise
        self._free_individual_gradients = free_individual_gradients

        self._savefield_individual_gradients = "grad_batch"

    def param_hook(self, param):
        """Compute pairwise individual gradient dot products and update Gram matrix."""
        individual_gradients = getattr(param, self._savefield_individual_gradients)
        gram_param = self.pairwise_dot(individual_gradients).detach()
        self._update_result(gram_param)

        if self._free_individual_gradients:
            delattr(param, self._savefield_individual_gradients)

        if self._layerwise:
            return gram_param

    def get_result(self):
        """Return the gradient Gram matrix computed from a backward pass."""
        return self._gram_mat

    def _update_result(self, mat):
        if self._gram_mat is None:
            self._gram_mat = mat
        else:
            self._gram_mat += mat

    @staticmethod
    def pairwise_dot(tensor):
        """Compute pairwise scalar product. Pairs are determined by the leading dim."""
        # TODO Avoid flattening with more sophisticated einsum equation
        tensor_flat = tensor.flatten(start_dim=1)
        equation = "if,jf->ij"

        return torch.einsum(equation, tensor_flat, tensor_flat)
