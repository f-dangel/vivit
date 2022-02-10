"""Extension hook for computing the GGN Gram matrix."""

from backpack.extensions.secondorder.hbp import LossHessianStrategy

from vivit.utils.gram import pairwise_dot
from vivit.utils.hooks import ParameterHook


class GramSqrtGGN(ParameterHook):
    SQRT_GGN_SAVEFIELDS = {
        LossHessianStrategy.EXACT: "sqrt_ggn_exact",
        LossHessianStrategy.SAMPLING: "sqrt_ggn_mc",
    }

    def __init__(self, loss_hessian_strategy, savefield, layerwise, free_sqrt_ggn):
        """Initialize attributes.

        Args:
            loss_hessian_strategy (str, "exact" or "sampling"): String describing how
                to represent the loss Hessian's symmetric decomposition that is back-
                propagated.
            savefield (str): Name of the attribute in under which the quantity computed
                by this extension can be found in the model parameters.
            layerwise (bool): Controls whether results should be kept layer-wise. If
                ``False``, results are aggregated over all parameters, and can must be
                retrieved via ``get_result``.
            free_sqrt_ggn (bool): Controls whether the backpropagated symmetric loss
                Hessian should be freed immediately after the Gram matrix has been
                computed.
        """
        super().__init__(savefield)

        self._gram_mat = None
        self._layerwise = layerwise
        self._free_sqrt_ggn = free_sqrt_ggn
        self._savefield_sqrt_ggn = self.SQRT_GGN_SAVEFIELDS[loss_hessian_strategy]

    def param_hook(self, param):
        """Compute pairwise dot products of GGN square root decomposition.

        Args:
            param (torch.nn.Parameter): Parameter whose GGN vectors are used
                to compute pairwise dot products.

        Returns:
            torch.Tensor: ``[NC x NC]`` tensor containing the pairwise dot products of
                of GGN vectors of samples ``1, ..., N`` and classes ``1, ..., C``,
                for the parameter.
        """
        sqrt_ggn = getattr(param, self._savefield_sqrt_ggn)
        gram_param = pairwise_dot(sqrt_ggn, start_dim=2).detach()
        self._update_result(gram_param)

        if self._free_sqrt_ggn:
            delattr(param, self._savefield_sqrt_ggn)

        if self._layerwise:
            return gram_param

    def get_result(self):
        """Return the GGN Gram matrix computed from a backward pass.

        Returns:
            torch.Tensor: ``[NC x NC]`` tensor containing the pairwise dot products of
                of GGN vectors of samples ``1, ..., N`` and classes ``1, ..., C``,
                for all model parameters.
        """
        return self._gram_mat

    def _update_result(self, mat):
        if self._gram_mat is None:
            self._gram_mat = mat
        else:
            self._gram_mat += mat


class GramSqrtGGNExact(GramSqrtGGN):
    """
    BackPACK extension hook to compute the Generalized Gauss-Newton/Fisher Gram matrix.
    Uses the exact Hessian of the loss w.r.t. the model output.

    Can be used as extension hook in ``with backpack(SqrtGGNExact()):``. It is
    obligatory that ``vivit``'s ``SqrtGGNExact`` extension is active.

    The result, a ``[CN x CN]`` tensor where ``N`` is the batch size and ``C`` is the
    model prediction output (number of classes for classification problems), is
    collected by calling ``get_result`` after a backward pass.

    Note: Single-use only

        The result buffer cannot be reset. Hence you need to create a new instance
        every backpropagation.

    Args:
        layerwise (bool): Whether layerwise Gram matrices should be kept. Otherwise
            they are discarded to save memory. If ``True``, a Gram matrix constructed
            from the parameter-wise symmetric composition will be stored in
            ``gram_sqrt_ggn_exact`` as a ``[CN x CN]`` tensor.
        free_sqrt_ggn (bool) : Whether symmetric composition, stored by the
            ``SqrtGGNExact`` extension should be freed during backpropagation to save
            memory.
    """

    def __init__(
        self, savefield="gram_sqrt_ggn_exact", layerwise=False, free_sqrt_ggn=False
    ):
        super().__init__(LossHessianStrategy.EXACT, savefield, layerwise, free_sqrt_ggn)


class GramSqrtGGNMC(GramSqrtGGN):
    """
    BackPACK extension hook to compute the Generalized Gauss-Newton/Fisher Gram matrix.
    Uses a Monte-Carlo approximation of the Hessian of the loss w.r.t. the model output.

    Can be used as extension hook in ``with backpack(SqrtGGNMC()):``. It is
    obligatory that ``vivit``'s ``SqrtGGNMC`` extension is active.

    The result, a ``[MN x MN]`` tensor where ``N`` is the batch size and ``M`` is the
    number of Monte-Carlo samples used by the ``SqrtGGNMC`` extension, is
    collected by calling ``get_result`` after a backward pass.

    Note: Single-use only

        The result buffer cannot be reset. Hence you need to create a new instance
        every backpropagation.

    Args:
        layerwise (bool): Whether layerwise Gram matrices should be kept. Otherwise
            they are discarded to save memory. If ``True``, a Gram matrix constructed
            from the parameter-wise symmetric composition will be stored in
            ``gram_sqrt_ggn_mc`` as a ``[MN x MN]`` tensor.
        free_sqrt_ggn (bool) : Whether symmetric composition, stored by the
            ``SqrtGGNMC`` extension should be freed during backpropagation to save
            memory.
    """

    def __init__(
        self, savefield="gram_sqrt_ggn_mc", layerwise=False, free_sqrt_ggn=False
    ):
        super().__init__(
            LossHessianStrategy.SAMPLING, savefield, layerwise, free_sqrt_ggn
        )
