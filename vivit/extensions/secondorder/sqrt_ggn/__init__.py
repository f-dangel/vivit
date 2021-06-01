"""Symmetric decomposition for the generalized Gauss-Newton."""

from backpack.extensions.backprop_extension import BackpropExtension
from backpack.extensions.secondorder.hbp import LossHessianStrategy
from torch.nn import (
    ELU,
    SELU,
    AvgPool1d,
    AvgPool2d,
    AvgPool3d,
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
    CrossEntropyLoss,
    Dropout,
    Flatten,
    LeakyReLU,
    Linear,
    LogSigmoid,
    MaxPool1d,
    MaxPool2d,
    MaxPool3d,
    MSELoss,
    ReLU,
    Sigmoid,
    Tanh,
    ZeroPad2d,
)

from vivit.extensions.secondorder.sqrt_ggn import (
    activations,
    conv1d,
    conv2d,
    conv3d,
    convtranspose1d,
    convtranspose2d,
    convtranspose3d,
    dropout,
    flatten,
    linear,
    losses,
    padding,
    pooling,
)


class SqrtGGN(BackpropExtension):
    VALID_LOSS_HESSIAN_STRATEGIES = [
        LossHessianStrategy.EXACT,
        LossHessianStrategy.SAMPLING,
    ]

    def __init__(self, loss_hessian_strategy, savefield, subsampling):
        if loss_hessian_strategy not in self.VALID_LOSS_HESSIAN_STRATEGIES:
            raise ValueError(
                "Unknown hessian strategy: {}".format(loss_hessian_strategy)
                + "Valid strategies: [{}]".format(self.VALID_LOSS_HESSIAN_STRATEGIES)
            )

        self.loss_hessian_strategy = loss_hessian_strategy
        super().__init__(
            savefield=savefield,
            fail_mode="ERROR",
            module_exts={
                MSELoss: losses.SqrtGGNMSELoss(),
                CrossEntropyLoss: losses.SqrtGGNCrossEntropyLoss(),
                Linear: linear.SqrtGGNLinear(),
                MaxPool1d: pooling.SqrtGGNMaxPool1d(),
                MaxPool2d: pooling.SqrtGGNMaxPool2d(),
                AvgPool1d: pooling.SqrtGGNAvgPool1d(),
                MaxPool3d: pooling.SqrtGGNMaxPool3d(),
                AvgPool2d: pooling.SqrtGGNAvgPool2d(),
                AvgPool3d: pooling.SqrtGGNAvgPool3d(),
                ZeroPad2d: padding.SqrtGGNZeroPad2d(),
                Conv1d: conv1d.SqrtGGNConv1d(),
                Conv2d: conv2d.SqrtGGNConv2d(),
                Conv3d: conv3d.SqrtGGNConv3d(),
                ConvTranspose1d: convtranspose1d.SqrtGGNConvTranspose1d(),
                ConvTranspose2d: convtranspose2d.SqrtGGNConvTranspose2d(),
                ConvTranspose3d: convtranspose3d.SqrtGGNConvTranspose3d(),
                Dropout: dropout.SqrtGGNDropout(),
                Flatten: flatten.SqrtGGNFlatten(),
                ReLU: activations.SqrtGGNReLU(),
                Sigmoid: activations.SqrtGGNSigmoid(),
                Tanh: activations.SqrtGGNTanh(),
                LeakyReLU: activations.SqrtGGNLeakyReLU(),
                LogSigmoid: activations.SqrtGGNLogSigmoid(),
                ELU: activations.SqrtGGNELU(),
                SELU: activations.SqrtGGNSELU(),
            },
        )
        self._subsampling = subsampling

    def get_subsampling(self):
        """Return indices of samples whose GGN/Fisher factorizations are evaluated.

        Returns:
            None or [int]: Indices of samples considered by the computation. ``None``
                signifies that the full mini-batch is used.
        """
        return self._subsampling


class SqrtGGNExact(SqrtGGN):
    """
    Symmetric factorization of the Generalized Gauss-Newton/Fisher.
    Uses the exact Hessian of the loss w.r.t. the model output.

    Stores the output in :code:`sqrt_ggn_exact`,
    has the dimensions ``[C, N, *]``, where ``C`` is the model output dimension (number
    of classes for classification problems), ``N`` is the batch (or subset) size, and
    ``*`` denotes the parameter shape.

    For a faster but less precise alternative, see
    :py:meth:`vivit.extensions.SqrtGGNMC`.

    Args:
        subsampling ([int], optional): Indices of samples in the current mini-batch,
            that will be backpropagated. Default value ``None`` uses the entire batch.

    Details:
        The ``[CN, *]`` matrix view ``V`` of :code:`sqrt_ggn_exact` is the symmetric
        factorization of the exact parameter GGN, i.e. ``G(θ) = Vᵀ V``.
    """

    def __init__(self, subsampling=None):
        super().__init__(LossHessianStrategy.EXACT, "sqrt_ggn_exact", subsampling)


class SqrtGGNMC(SqrtGGN):
    """
    Symmetric composition of the Generalized Gauss-Newton/Fisher.
    Uses a Monte-Carlo approximation of the Hessian of the loss w.r.t. the model output.

    Stores the output in :code:`sqrt_ggn_mc`,
    has the dimensions ``[C, N, *]``, where ``C`` is the model output dimension (number
    of classes for classification problems), ``N`` is the batch (or subset) size, and
    ``*`` denotes the parameter shape.

    Args:
        subsampling ([int], optional): Indices of samples in the current mini-batch,
            that will be backpropagated. Default value ``None`` uses the entire batch.

    Details:
        The ``[CN, *]`` matrix view ``V`` of :code:`sqrt_ggn_mc` is the symmetric
        factorization of the approximate parameter GGN, i.e. ``G(θ) ≈ Vᵀ V``.
    """

    def __init__(self, mc_samples=1, subsampling=None):
        self._mc_samples = mc_samples
        super().__init__(LossHessianStrategy.SAMPLING, "sqrt_ggn_mc", subsampling)

    def get_num_mc_samples(self):
        return self._mc_samples
