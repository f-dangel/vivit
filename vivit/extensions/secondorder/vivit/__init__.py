"""Extension that provides functional access to ``V`` and its Gram matrix."""

from typing import List, Union

from backpack.custom_module.branching import SumModule
from backpack.custom_module.pad import Pad
from backpack.custom_module.scale_module import ScaleModule
from backpack.custom_module.slicing import Slicing
from backpack.extensions.secondorder.base import SecondOrderBackpropExtension
from backpack.extensions.secondorder.hbp import LossHessianStrategy
from backpack.extensions.secondorder.sqrt_ggn import (
    activations,
    custom_module,
    dropout,
    flatten,
    losses,
    pad,
    padding,
    pooling,
    slicing,
)
from torch import Tensor
from torch.nn import (
    ELU,
    SELU,
    AvgPool1d,
    AvgPool2d,
    AvgPool3d,
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
    CrossEntropyLoss,
    Dropout,
    Flatten,
    Identity,
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

from vivit.extensions.secondorder.vivit import (
    batchnormnd,
    convnd,
    convtransposend,
    linear,
)


class ViViTGGN(SecondOrderBackpropExtension):
    """Base class for functional access to the GGN's ``V``, ``Vᵀ``, and Gram matrix."""

    def __init__(
        self,
        loss_hessian_strategy: str,
        savefield: str,
        subsampling: Union[List[int], None],
    ):
        """Store approximation for backpropagated object and where to save the result.

        Args:
            loss_hessian_strategy: Which approximation is used for the backpropagated
                loss Hessian. Must be ``'exact'`` or ``'sampling'``.
            savefield: Attribute under which the quantity is saved in a parameter.
            subsampling: Indices of active samples. ``None`` uses the full mini-batch.
        """
        self.loss_hessian_strategy = loss_hessian_strategy
        super().__init__(
            savefield=savefield,
            fail_mode="ERROR",
            module_exts={
                MSELoss: losses.SqrtGGNMSELoss(),
                CrossEntropyLoss: losses.SqrtGGNCrossEntropyLoss(),
                Linear: linear.ViViTGGNLinear(),
                MaxPool1d: pooling.SqrtGGNMaxPool1d(),
                MaxPool2d: pooling.SqrtGGNMaxPool2d(),
                AvgPool1d: pooling.SqrtGGNAvgPool1d(),
                MaxPool3d: pooling.SqrtGGNMaxPool3d(),
                AvgPool2d: pooling.SqrtGGNAvgPool2d(),
                AvgPool3d: pooling.SqrtGGNAvgPool3d(),
                ZeroPad2d: padding.SqrtGGNZeroPad2d(),
                Conv1d: convnd.ViViTGGNConv1d(),
                Conv2d: convnd.ViViTGGNConv2d(),
                Conv3d: convnd.ViViTGGNConv3d(),
                ConvTranspose1d: convtransposend.ViViTGGNConvTranspose1d(),
                ConvTranspose2d: convtransposend.ViViTGGNConvTranspose2d(),
                ConvTranspose3d: convtransposend.ViViTGGNConvTranspose3d(),
                Dropout: dropout.SqrtGGNDropout(),
                Flatten: flatten.SqrtGGNFlatten(),
                ReLU: activations.SqrtGGNReLU(),
                Sigmoid: activations.SqrtGGNSigmoid(),
                Tanh: activations.SqrtGGNTanh(),
                LeakyReLU: activations.SqrtGGNLeakyReLU(),
                LogSigmoid: activations.SqrtGGNLogSigmoid(),
                ELU: activations.SqrtGGNELU(),
                SELU: activations.SqrtGGNSELU(),
                BatchNorm1d: batchnormnd.ViViTGGNBatchNormNd(),
                BatchNorm2d: batchnormnd.ViViTGGNBatchNormNd(),
                BatchNorm3d: batchnormnd.ViViTGGNBatchNormNd(),
                Identity: custom_module.SqrtGGNScaleModule(),
                ScaleModule: custom_module.SqrtGGNScaleModule(),
                SumModule: custom_module.SqrtGGNSumModule(),
                Pad: pad.SqrtGGNPad(),
                Slicing: slicing.SqrtGGNSlicing(),
            },
            subsampling=subsampling,
        )

    def get_loss_hessian_strategy(self) -> str:
        """Return the strategy used to represent the backpropagated loss Hessian.

        Returns:
            Loss Hessian strategy.
        """
        return self.loss_hessian_strategy

    def accumulate_backpropagated_quantities(
        self, existing: Tensor, other: Tensor
    ) -> Tensor:  # noqa: D102
        return existing + other


class ViViTGGNExact(ViViTGGN):
    """Functional access to the exact GGN/Fisher matrix square root.

    Uses the exact Hessian of the loss w.r.t. the model output.

    Stores a dictionary in :code:`vivit_ggn_exact` that contains a function to
    evaluate the Gram matrix & functions to apply ``V`` and ``Vᵀ`` to multiple vectors.
    """

    def __init__(self, subsampling: List[int] = None):
        """Use exact loss Hessian, store results under ``sqrt_ggn_exact``.

        Args:
            subsampling: Indices of active samples. Defaults to ``None`` (use all
                samples in the mini-batch).
        """
        super().__init__(LossHessianStrategy.EXACT, "vivit_ggn_exact", subsampling)


class ViViTGGNMC(ViViTGGN):
    """Functional access to the MC-sampled GGN/Fisher matrix square root.

    Uses the exact Hessian of the loss w.r.t. the model output.

    Stores a dictionary in :code:`vivit_ggn_mc` that contains a function to
    evaluate the Gram matrix & functions to apply ``V`` and ``Vᵀ`` to multiple vectors.
    """

    def __init__(self, mc_samples: int = 1, subsampling: List[int] = None):
        """Approximate loss Hessian via MC and set savefield to ``sqrt_ggn_mc``.

        Args:
            mc_samples: Number of Monte-Carlo samples. Default: ``1``.
            subsampling: Indices of active samples. Defaults to ``None`` (use all
                samples in the mini-batch).
        """
        self._mc_samples = mc_samples
        super().__init__(LossHessianStrategy.SAMPLING, "vivit_ggn_mc", subsampling)

    def get_num_mc_samples(self) -> int:
        """Return the number of MC samples used to approximate the loss Hessian.

        Returns:
            Number of Monte-Carlo samples.
        """
        return self._mc_samples
