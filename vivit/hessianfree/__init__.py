"""Contains functionality to analyze Hessian & GGN via matrix-free multiplication."""

from typing import Iterable, List, Tuple

from backpack.hessianfree.ggnvp import ggn_vector_product_from_plist
from backpack.hessianfree.hvp import hessian_vector_product
from backpack.utils.convert_parameters import vector_to_parameter_list
from numpy import float32, ndarray
from numpy.typing import DTypeLike
from scipy.sparse.linalg import LinearOperator
from torch import Tensor, device, from_numpy, tensor, zeros_like
from torch.autograd import grad
from torch.nn import Module
from torch.nn.utils import parameters_to_vector
from tqdm import tqdm


class _LinearOperator(LinearOperator):
    """Base class for linear operators of DNN curvature matrices.

    Can be used with SciPy.
    """

    def __init__(
        self,
        model: Module,
        loss_func: Module,
        data: Iterable[Tuple[Tensor, Tensor]],
        device: device,
        dtype: DTypeLike = float32,
        progressbar: bool = False,
    ):
        """Linear operator for DNN curvature matrices.

        Args:
            model: Neural network.
            loss_func: Loss function criterion.
            data: Source from which mini-batches can be drawn, for instance a list of
                mini-batches ``[(X, y), ...]`` or a torch ``DataLoader``.
            device: Device on which matrix multiplications are carried out.
            dtype: Matrix data type (for SciPy operations).
            progressbar: Show a progressbar during matrix-multiplication.
                Default: ``False``.
        """
        self._params = [p for p in model.parameters() if p.requires_grad]
        dim = sum(p.numel() for p in self._params)
        super().__init__(shape=(dim, dim), dtype=dtype)

        self._model = model.to(device)
        self._loss_func = loss_func.to(device)
        self._data = data
        self._device = device
        self._progressbar = progressbar

        self._N_data = sum(X.shape[0] for (X, _) in self._loop_over_data())

    def _matvec(self, x: ndarray) -> ndarray:
        """Loop over all batches in the data and apply the matrix to vector x.

        Args:
            x: Vector for multiplication.

        Returns:
            Matrix-multiplication result ``mat @ x``.
        """
        x_list = self._preprocess(x)
        out_list = [zeros_like(x) for x in x_list]

        for (X, y) in self._loop_over_data():
            normalization_factor = self._get_normalization_factor(X, y)

            for mat_x, current in zip(out_list, self._matvec_batch(X, y, x_list)):
                mat_x.add_(current, alpha=normalization_factor)

        return self._postprocess(out_list)

    def _matvec_batch(
        self, X: Tensor, y: Tensor, x_list: List[Tensor]
    ) -> Tuple[Tensor]:
        """Apply the mini-batch matrix to a vector.

        Args:
            X: Input to the DNN.
            y: Ground truth.
            x_list: Vector in list format (same shape as trainable model parameters).

        Returns: # noqa: D402
           Result of matrix-multiplication in list format.

        Raises:
            NotImplementedError: Must be implemented by descendants.
        """
        raise NotImplementedError

    def _preprocess(self, x: ndarray) -> List[Tensor]:
        """Convert flat numpy array to torch list format.

        Args:
            x: Vector for multiplication.

        Returns:
            Vector in list format.
        """
        x_torch = from_numpy(x).to(self._device)
        return vector_to_parameter_list(x_torch, self._params)

    @staticmethod
    def _postprocess(x_list: List[Tensor]) -> ndarray:
        """Convert torch list format to flat numpy array.

        Args:
            x_list: Vector in list format.

        Returns:
            Flat vector.
        """
        return parameters_to_vector([x.contiguous() for x in x_list]).cpu().numpy()

    def _loop_over_data(self) -> Iterable[Tuple[Tensor, Tensor]]:
        """Yield batches of the data set, loaded to the correct device.

        Yields:
            Mini-batches ``(X, y)``.
        """
        data_iter = iter(self._data)

        if self._progressbar:
            data_iter = tqdm(data_iter, desc="matvec")

        for (X, y) in data_iter:
            X, y = X.to(self._device), y.to(self._device)
            yield (X, y)

    def gradient_and_loss(self) -> Tuple[List[Tensor], Tensor]:
        """Evaluate the gradient and loss on the data.

        (Not really part of the LinearOperator interface.)

        Returns:
            Gradient and loss on the data set.
        """
        total_loss = tensor([0.0], device=self._device)
        total_grad = [zeros_like(p) for p in self._params]

        for (X, y) in self._loop_over_data():
            loss = self._loss_func(self._model(X), y)
            normalization_factor = self._get_normalization_factor(X, y)

            for grad_param, current in zip(total_grad, grad(loss, self._params)):
                grad_param.add_(current, alpha=normalization_factor)
            total_loss.add_(loss.detach(), alpha=normalization_factor)

        return total_grad, total_loss

    def _get_normalization_factor(self, X: Tensor, y: Tensor) -> float:
        """Return the correction factor for correct normalization over the data set.

        Args:
            X: Input to the DNN.
            y: Ground truth.

        Returns:
            Normalization factor

        Raises:
            ValueError: If loss function does not have a ``reduction`` attribute or
                it is not set to ``'mean'`` or ``'sum'``.
        """
        if not hasattr(self._loss_func, "reduction"):
            raise ValueError("Loss must have a 'reduction' attribute.")

        reduction = self._loss_func.reduction
        if reduction == "sum":
            return 1.0
        elif reduction == "mean":
            return X.shape[0] / self._N_data
        else:
            raise ValueError("Loss must have reduction 'mean' or 'sum'.")


class HessianLinearOperator(_LinearOperator):
    """Hessian as SciPy linear operator."""

    def _matvec_batch(
        self, X: Tensor, y: Tensor, x_list: List[Tensor]
    ) -> Tuple[Tensor, ...]:
        """Apply the mini-batch Hessian to a vector.

        Args:
            X: Input to the DNN.
            y: Ground truth.
            x_list: Vector in list format (same shape as trainable model parameters).

        Returns:
            Result of Hessian-multiplication in list format.
        """
        loss = self._loss_func(self._model(X), y)
        return hessian_vector_product(loss, self._params, x_list)


class GGNLinearOperator(_LinearOperator):
    """GGN as SciPy linear operator."""

    def _matvec_batch(
        self, X: Tensor, y: Tensor, x_list: List[Tensor]
    ) -> Tuple[Tensor, ...]:
        """Apply the mini-batch GGN to a vector.

        Args:
            X: Input to the DNN.
            y: Ground truth.
            x_list: Vector in list format (same shape as trainable model parameters).

        Returns:
            Result of GGN-multiplication in list format.
        """
        output = self._model(X)
        loss = self._loss_func(output, y)
        return ggn_vector_product_from_plist(loss, output, self._params, x_list)
