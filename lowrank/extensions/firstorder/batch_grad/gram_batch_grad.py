"""Interface for computing the individual gradient Gram matrix."""

from lowrank.utils.gram import pairwise_dot
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
        free_grad_batch=False,
    ):
        super().__init__(savefield)

        self._gram_mat = None
        self._layerwise = layerwise
        self._free_grad_batch = free_grad_batch

        self._savefield_grad_batch = "grad_batch"

    def param_hook(self, param):
        """Compute pairwise individual gradient dot products and update Gram matrix."""
        grad_batch = getattr(param, self._savefield_grad_batch)
        gram_param = pairwise_dot(grad_batch, start_dim=1).detach()
        self._update_result(gram_param)

        if self._free_grad_batch:
            delattr(param, self._savefield_grad_batch)

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
