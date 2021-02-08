"""Interface for computing the individual gradient Gram matrix."""

from lowrank.utils.gram import pairwise_dot
from lowrank.utils.hooks import ParameterHook


class GramBatchGrad(ParameterHook):
    """BackPACK extension hook that computes the gradient Gram matrix.

    Can be used as extension hook in ``with backpack(BatchGrad()):``. It is
    obligatory that ``backpack``'s ``BatchGrad`` extension is active.

    The result, an ``[N x N]`` tensor, is collected by calling ``get_result``
    after a backward pass.

    Note: Single-use only

        The result buffer cannot be reset. Hence you need to create a new instance
        every backpropagation.

    Note: beware of scaling issue

        BackPACK computes individual gradients with a scaling factor stemming
        from the loss function's ``reduction`` argument.

        Let ``fᵢ`` be the loss of the ``i`` th sample, with gradient ``gᵢ``.
        The individual gradients computed by BackPACK are

        - ``[g₁, …, gₙ]`` if the loss is a sum, ``∑ᵢ₌₁ⁿ fᵢ``,
        - ``[¹/ₙ g₁, …, ¹/ₙ gₙ]`` if the loss is a mean, ``¹/ₙ ∑ᵢ₌₁ⁿ fᵢ``.

        The quantity computed by this hook is a matrix containing pairwise scalar
        products of those vectors, i.e. it has elements

        - ``⟨gᵢ, gⱼ⟩`` if the loss is a sum, ``∑ᵢ₌₁ⁿ fᵢ``,
        - ``⟨¹/ₙ gᵢ, ¹/ₙ gⱼ⟩`` if the loss is a mean, ``¹/ₙ ∑ᵢ₌₁ⁿ fᵢ``.

        This must be kept in mind as the object of interest is often given by a matrix
        with elements ``1/ₙ ⟨gᵢ, gⱼ⟩``.

    Args:
        layerwise (bool): Whether layerwise Gram matrices should be kept. Otherwise
            they are discarded to save memory. If ``True``, a Gram matrix constructed
            from the parameter-wise individual gradients will be stored in
            ``grad_grad_batch`` as an ``[N x N]`` tensor.
        free_grad_batch (bool) : Whether individual gradients, stored by the
            ``BatchGrad`` extension should be freed during backpropagation to save
            memory.
    """

    _SAVEFIELD_GRAD_BATCH = "grad_batch"

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

    def param_hook(self, param):
        """Compute pairwise individual gradient dot products and update Gram matrix."""
        grad_batch = getattr(param, self._SAVEFIELD_GRAD_BATCH)
        gram_param = pairwise_dot(grad_batch, start_dim=1).detach()
        self._update_result(gram_param)

        if self._free_grad_batch:
            delattr(param, self._SAVEFIELD_GRAD_BATCH)

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
