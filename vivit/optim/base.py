"""Base class for optimizers that use a closure with BackPACK extensions."""

from typing import Callable

from torch import Tensor
from torch.optim import Optimizer


class BackpackOptimizer(Optimizer):
    """Base class for optimizers that use a closure with BackPACK extensions.

    Note:
        For better control of the backward pass, the closure has different
        responsibilities in comparison to the official documentation
        (https://pytorch.org/docs/stable/optim.html): It only performs a forward
        pass and returns the loss. This optimizer class needs to take care of
        clearing the gradients performing the backward pass.
    """

    def step(self, closure: Callable[[], Tensor]):
        """Perform a singel optimization step (parameter update).

        Args:
            closure: Function that evaluates the model and returns the loss.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError
