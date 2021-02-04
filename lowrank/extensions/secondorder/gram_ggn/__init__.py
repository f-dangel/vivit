"""Gram matrix for the generalized Gauss-Newton."""

from torch.nn import (ELU, SELU, AvgPool1d, AvgPool2d, AvgPool3d, Conv1d,
                      Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d,
                      ConvTranspose3d, CrossEntropyLoss, Dropout, Flatten,
                      LeakyReLU, Linear, LogSigmoid, MaxPool1d, MaxPool2d,
                      MaxPool3d, MSELoss, ReLU, Sigmoid, Tanh, ZeroPad2d)

from backpack.extensions.backprop_extension import BackpropExtension
from backpack.extensions.secondorder.hbp import LossHessianStrategy
from lowrank.extensions.secondorder.gram_ggn import (activations, conv1d,
                                                     conv2d, conv3d,
                                                     convtranspose1d,
                                                     convtranspose2d,
                                                     convtranspose3d, dropout,
                                                     flatten, linear, losses,
                                                     padding, pooling)


class GramGGN(BackpropExtension):
    VALID_LOSS_HESSIAN_STRATEGIES = [
        LossHessianStrategy.EXACT,
        LossHessianStrategy.SAMPLING,
    ]

    def __init__(self, loss_hessian_strategy=LossHessianStrategy.EXACT, savefield=None):
        if savefield is None:
            savefield = "gram_ggn"
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
                MSELoss: losses.GramGGNMSELoss(),
                CrossEntropyLoss: losses.GramGGNCrossEntropyLoss(),
                Linear: linear.GramGGNLinear(),
                MaxPool1d: pooling.GramGGNMaxPool1d(),
                MaxPool2d: pooling.GramGGNMaxPool2d(),
                AvgPool1d: pooling.GramGGNAvgPool1d(),
                MaxPool3d: pooling.GramGGNMaxPool3d(),
                AvgPool2d: pooling.GramGGNAvgPool2d(),
                AvgPool3d: pooling.GramGGNAvgPool3d(),
                ZeroPad2d: padding.GramGGNZeroPad2d(),
                Conv1d: conv1d.GramGGNConv1d(),
                Conv2d: conv2d.GramGGNConv2d(),
                Conv3d: conv3d.GramGGNConv3d(),
                ConvTranspose1d: convtranspose1d.GramGGNConvTranspose1d(),
                ConvTranspose2d: convtranspose2d.GramGGNConvTranspose2d(),
                ConvTranspose3d: convtranspose3d.GramGGNConvTranspose3d(),
                Dropout: dropout.GramGGNDropout(),
                Flatten: flatten.GramGGNFlatten(),
                ReLU: activations.GramGGNReLU(),
                Sigmoid: activations.GramGGNSigmoid(),
                Tanh: activations.GramGGNTanh(),
                LeakyReLU: activations.GramGGNLeakyReLU(),
                LogSigmoid: activations.GramGGNLogSigmoid(),
                ELU: activations.GramGGNELU(),
                SELU: activations.GramGGNSELU(),
            },
        )
