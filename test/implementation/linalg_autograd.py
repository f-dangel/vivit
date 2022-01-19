"""Autograd implementation of operations in ``vivit.linalg``."""

from test.implementation.autograd import AutogradExtensions
from typing import Dict, List, Tuple, Union

from backpack.utils.convert_parameters import vector_to_parameter_list
from torch import Tensor, stack


class AutogradLinalgExtensions(AutogradExtensions):
    """Autograd implementation of linalg functionality with similar API."""

    def eigvalsh_ggn(
        self, param_groups: List[Dict], subsampling: Union[List[int], None]
    ) -> Dict[int, Tensor]:
        """Compute the GGN's eigenvalues via the GGN matrix.

        Args:
            param_groups: Parameter groups implying the block-diagonal approximation.
            subsampling: Indices of samples used for the computation.

        Returns:
            Dictionary that stores the eigenvalues of block under the group id.
        """
        group_evals, _ = self.directions_ggn(param_groups, subsampling=subsampling)

        return {id(group): evals for group, evals in zip(param_groups, group_evals)}

    def eigh_ggn(
        self, param_groups: List[Dict], subsampling: Union[List[int], None]
    ) -> Tuple[Dict[int, Tensor], Dict[int, List[Tensor]]]:
        """Compute the GGN's eigenvalues and normalized eigenvectors via the GGN matrix.

        Args:
            param_groups: Parameter groups implying the block-diagonal approximation.
            subsampling: Indices of samples used for the computation.

        Returns:
            Dictionary that stores the eigenvalues and eigenvectors (in parameter
            format) of block under the group id. Eigenvectors are indexed through the
            leading axis
        """
        group_evals, group_evecs = self.directions_ggn(
            param_groups, subsampling=subsampling
        )

        evals = {id(group): e for group, e in zip(param_groups, group_evals)}
        evecs = {}

        for group, vecs in zip(param_groups, group_evecs):
            # make eigenvectors selectable via first axis
            vecs = vecs.transpose(0, 1)

            # nested list with eigenvectors in parameter format
            vec_list = [vector_to_parameter_list(vec, group["params"]) for vec in vecs]
            vec_list = list(zip(*vec_list))
            vecs_list = [stack(vec) for vec in vec_list]

            evecs[id(group)] = vecs_list

        return evals, evecs
