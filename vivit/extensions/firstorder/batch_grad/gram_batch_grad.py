"""Interface for computing the individual gradient Gram matrix."""

from vivit.utils.gram import pairwise_dot
from vivit.utils.hooks import ParameterHook


class CenteredBatchGrad(ParameterHook):
    """Extension hook that centers individual gradients by subtracting their mean.

    Stores the output in an ``[N, *]`` tensor, where ``*`` is the associated
    parameter's shape. The output is stored in an attribute determined by
    ``savefield``.

    Note: Single-use only

        You need to create a new instance of this object every backpropagation.
    """

    _SAVEFIELD_GRAD_BATCH = "grad_batch"

    def __init__(self, savefield="centered_grad_batch"):
        super().__init__(savefield)

    def param_hook(self, param):
        """Subtract individual gradient mean from individual gradients.

        Args:
            param (torch.nn.Parameter): Parameter whose individual gradients will
                be centered an saved under ``savefield``.

        Returns:
            torch.Tensor: Centered parameter-wise individual gradients.
        """
        N_axis = 0
        grad_batch = getattr(param, self._SAVEFIELD_GRAD_BATCH)

        return grad_batch - grad_batch.mean(N_axis)


class _GramBatchGradBase(ParameterHook):
    """Base class for Gram matrices based on individual gradients.

    Note: Single-use only

        The result buffer cannot be reset. Hence you need to create a new instance
        every backpropagation.

    Args:
        center (bool): Whether to center the individual gradients before computing
            pairwise scalar products.
        layerwise (bool): Whether layerwise Gram matrices should be kept. Otherwise
            they are discarded to save memory. If ``True``, a Gram matrix constructed
            from the parameter-wise individual gradients will be stored in
            ``savefield`` as an ``[N x N]`` tensor.
        free_grad_batch (bool) : Whether individual gradients, stored by the
            ``BatchGrad`` extension should be freed during backpropagation to save
            memory.
    """

    _SAVEFIELD_GRAD_BATCH = "grad_batch"

    def __init__(
        self,
        savefield,
        center,
        layerwise=False,
        free_grad_batch=False,
    ):
        super().__init__(savefield)

        self._center = center
        self._gram_mat = None
        self._layerwise = layerwise
        self._free_grad_batch = free_grad_batch

    def param_hook(self, param):
        """Compute pairwise individual gradient dot products and update Gram matrix.

        Optionally center individual gradients before the scalar product, depending
        on ``self._center``.

        Args:
            param (torch.nn.Parameter): Parameter whose individual gradients are used
                to compute pairwise dot products.

        Returns:
            torch.Tensor: ``[N x N]`` tensor containing the pairwise dot products of
                of ``param``'s individual gradients w.r.t samples ``1, ..., N``.
        """
        grad_batch = getattr(param, self._SAVEFIELD_GRAD_BATCH)

        if self._center:
            grad_batch -= grad_batch.mean(0)

        gram_param = pairwise_dot(grad_batch, start_dim=1).detach()
        self._update_result(gram_param)

        if self._free_grad_batch:
            delattr(param, self._SAVEFIELD_GRAD_BATCH)

        if self._layerwise:
            return gram_param

    def get_result(self):
        """Return (un-)centered gradient Gram matrix computed from a backward pass.

        Returns:
            torch.Tensor: ``[N x N]`` tensor containing the pairwise dot products of
                of individual gradients w.r.t samples ``1, ..., N``.
        """
        return self._gram_mat

    def _update_result(self, mat):
        if self._gram_mat is None:
            self._gram_mat = mat
        else:
            self._gram_mat += mat


class GramBatchGrad(_GramBatchGradBase):
    """BackPACK extension hook that computes the uncentered gradient Gram matrix.

    Can be used as extension hook in ``with backpack(BatchGrad()):``. It is
    obligatory that ``backpack``'s ``BatchGrad`` extension is active.

    The result, an ``[N x N]`` tensor where ``N`` is the batch size, is
    collected by calling ``get_result`` after a backward pass.

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

    """

    def __init__(
        self,
        savefield="gram_grad_batch",
        layerwise=False,
        free_grad_batch=False,
    ):
        center = False

        super().__init__(
            savefield,
            center,
            layerwise=layerwise,
            free_grad_batch=free_grad_batch,
        )


class CenteredGramBatchGrad(_GramBatchGradBase):
    """BackPACK extension hook that computes the centered gradient Gram matrix.

    Can be used as extension hook in ``with backpack(BatchGrad()):``. It is
    obligatory that ``backpack``'s ``BatchGrad`` extension is active.

    The result, an ``[N x N]`` tensor where ``N`` is the batch size, is
    collected by calling ``get_result`` after a backward pass.

    Note: beware of scaling issue

        BackPACK computes individual gradients with a scaling factor stemming
        from the loss function's ``reduction`` argument.

        Let ``fᵢ`` be the loss of the ``i`` th sample, with gradient ``gᵢ``.
        The individual gradients computed by BackPACK are

        - ``[g₁, …, gₙ]`` if the loss is a sum, ``∑ᵢ₌₁ⁿ fᵢ``,
        - ``[¹/ₙ g₁, …, ¹/ₙ gₙ]`` if the loss is a mean, ``¹/ₙ ∑ᵢ₌₁ⁿ fᵢ``.

        The quantity computed by this hook is a matrix containing pairwise scalar
        products of those vectors, i.e. it has elements

        - ``⟨gᵢ - (¹/ₙ ∑ₗ₌₁ⁿ gₗ) , gⱼ - (¹/ₙ ∑ₗ₌₁ⁿ gₗ)⟩`` if the loss is a sum,
          ``∑ᵢ₌₁ⁿ fᵢ``,
        - ``⟨¹/ₙ (gᵢ - (¹/ₙ ∑ₗ₌₁ⁿ gₗ)), ¹/ₙ (gⱼ- (¹/ₙ ∑ₗ₌₁ⁿ gₗ))⟩`` if the loss is a
          mean, ``¹/ₙ ∑ᵢ₌₁ⁿ fᵢ``.

        This must be kept in mind as the object of interest is often given by a matrix
        with elements ``1/ₙ ⟨gᵢ - (¹/ₙ ∑ₗ₌₁ⁿ gₗ), gⱼ - (¹/ₙ ∑ₗ₌₁ⁿ gₗ)⟩``.

    """

    def __init__(
        self,
        savefield="centered_gram_grad_batch",
        layerwise=False,
        free_grad_batch=False,
    ):
        center = True

        super().__init__(
            savefield,
            center,
            layerwise=layerwise,
            free_grad_batch=free_grad_batch,
        )
