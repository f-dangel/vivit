"""BackPACK implementation of operations used in ``vivit.linalg``."""

from test.implementation.backpack import BackpackExtensions
from typing import Dict, List, Tuple, Union

from backpack import backpack
from torch import Tensor

from vivit.linalg.eigh import EighComputation
from vivit.linalg.eigvalsh import EigvalshComputation


class BackpackLinalgExtensions(BackpackExtensions):
    """BackPACK implementation of linalg functionality with similar API."""

    def eigvalsh_ggn(
        self, param_groups: List[Dict], subsampling: Union[List[int], None]
    ) -> Dict[int, Tensor]:
        """Compute the GGN's eigenvalues via the Gram matrix.

        Uses ``EigvalshComputation``.

        Args:
            param_groups: Parameter groups implying the block-diagonal approximation.
            subsampling: Indices of samples used for the computation.

        Returns:
            Dictionary that stores the eigenvalues of block under the group id.
        """
        computation = EigvalshComputation(subsampling=subsampling)

        with backpack(
            computation.get_extension(),
            extension_hook=computation.get_extension_hook(param_groups),
        ):
            _, _, loss = self.problem.forward_pass()
            loss.backward()

        return computation._evals

    def eigh_ggn(
        self, param_groups: List[Dict], subsampling: Union[List[int], None]
    ) -> Tuple[Dict[int, Tensor], Dict[int, List[Tensor]]]:
        """Compute the GGN's eigenvalues and eigenvectors via the Gram matrix.

        Uses ``EighComputation``.

        Args:
            param_groups: Parameter groups implying the block-diagonal approximation.
            subsampling: Indices of samples used for the computation.

        Returns:
            Dictionary that stores eval and evecs (in parameter format) of block under
            the group id.
        """
        computation = EighComputation(subsampling=subsampling)

        with backpack(
            computation.get_extension(),
            extension_hook=computation.get_extension_hook(param_groups),
        ):
            _, _, loss = self.problem.forward_pass()
            loss.backward()

        return computation._evals, computation._evecs
