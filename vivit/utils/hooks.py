"""Utility functions and base classes for implementing extension hooks."""

import types

import torch


class ModuleHook:
    """Hook class to perform actions on parameters right after BackPACK's extension.

    Hook has access to the parameter and its module. Use this hook if information from
    a module needs to be stored inside a parameter.

    To inherit from this class:
    - Implement the ``module_hook`` function.
    """

    def module_hook(self, param, module):
        """Extract info from a parameter during backpropagation with BackPACK.

        Args:
            param (torch.Tensor): Parameter of a neural net.
            module (torch.nn.Module): Layer that `param` is part of.

        Return:
            Arbitrary output which will be stored in ``param``'s attribute
            ``self.savefield``.

        Raises:
            NotImplementedError: Must be implemented by child classes.
        """
        raise NotImplementedError

    def __init__(self, savefield=None):
        """Store the attribute under which results are attached to parameters.

        Args:
            savefield (str, optional): Attribute name under which results can be saved.
                ``None`` means that the hook has side effects, but no results will be
                saved in parameters. Default value: ``None``.
        """
        self.savefield = savefield
        self.processed = set()

    def __call__(self, module):
        """Execute hook on all module parameters. Skip already processes parameters.

        Args:
            module (torch.nn.Module): Hook is applied to all parameters in module.
        """
        for param in module.parameters():
            if self.should_run_hook(param, module):
                self.run_hook(param, module)

    def should_run_hook(self, param, module):
        """Check if hooks should be executed on a parameter.

        Hooks are only executed once on every trainable parameter.
        ``torch.nn.Sequential``s are being skipped.

        Args:
            param (torch.Tensor): Parameter of a neural net.
            module (torch.nn.Module): Layer that `param` is part of.

        Returns:
            bool: Whether the hook should be executed on the parameter.
        """
        if isinstance(module, torch.nn.Sequential):
            return False
        else:
            return id(param) not in self.processed and param.requires_grad

    def run_hook(self, param, module):
        """Execute the hook on parameter, add it to processed items and store result.

        Args:
            param (torch.nn.Parameter): Parameter to execute the hook on.
            module (torch.nn.Module): Module that contains ``param``.
        """
        value = self.module_hook(param, module)
        self._save(value, param)
        self.processed.add(id(param))

    def _save(self, value, param):
        """Store value in parameter's ``savefield`` argument if necessary.

        Args:
            value (any): Arbitrary object that will be stored.
            param (torch.nn.Parameter): Parameter the value is attached to.

        Raises:
            ValueError: If the hook produced an output, but the savefield is empty.
        """
        should_save = self.savefield is not None

        if value is not None and not should_save:
            raise ValueError(
                f"Hook has no savefield, but produced output of type {type(value)}."
            )

        if should_save:
            setattr(param, self.savefield, value)


class ParameterHook(ModuleHook):
    """Hook class to perform actions on parameters right after BackPACK's extension.

    Hook has access to the parameter.

    To inherit from this class:
    - Implement the ``param_hook`` function.
    """

    def param_hook(self, param):
        """Extract info from a parameter during backpropagation with BackPACK.

        Args:
            param (torch.Tensor): Parameter of a neural net.

        Return:
            Arbitrary output which will be stored in ``param``'s attribute
            ``self.savefield``.

        Raises:
            NotImplementedError: Must be implemented by child classes.
        """
        raise NotImplementedError

    def module_hook(self, param, module):
        """Extract info from a parameter during backpropagation with BackPACK.

        Args:
            param (torch.Tensor): Parameter of a neural net.
            module (torch.nn.Module): Layer that `param` is part of.

        Returns:
            any: Arbitrary output which will be stored in ``param``'s attribute
            ``self.savefield``.
        """
        return self.param_hook(param)


class ParameterGroupsHook(ParameterHook):
    """Handle computations during backpropagation parameterized by parameter groups.

    Computation results for parameters in the same group are accumulated, then further
    processed once the entire group has undergone backpropagation.

    To inherit from this class:
    - Implement the ``param_computation`` function.
    - Implement the ``group_hook`` function.
    - Implement the ``accumulate`` function.
    """

    def group_hook(self, accumulation, group):
        """Process accumulated results from parameter computations.

        Args:
            accumulation (any): Accumulated parameter computations from cache.
            group (dict): Parameter group of a ``torch.optim.Optimizer``.

        Returns: # noqa: DAR102
            Any: Result that will be saved under the group id.

        Raises:
            NotImplementedError: Must be implemented by child classes.
        """
        raise NotImplementedError

    def param_computation(self, param):
        """Compute partial result of group computation for a parameter.

        Args:
            param (torch.Tensor): Parameter of a neural net.

        Returns: # noqa: DAR102
            torch.Tensor: Result of parameter computation that will be accumulated
                group-wise.

        Raises:
            NotImplementedError: Must be implemented by child classes.
        """
        raise NotImplementedError

    def accumulate(self, existing, update):
        """Update the currently accumulated result with the update from a parameter.

        Args:
            existing (any): Cached accumulation for a group.
            update (any): Result from parameter computation.

        Returns: # noqa: DAR102
            any: Updated result that will be written to cache.

        Raises:
            NotImplementedError: Must be implemented by child classes.
        """
        raise NotImplementedError

    def __init__(self, param_groups):
        """Store parameter groups. Set up mappings between groups and parameters.

        Args:
            param_groups (list): Parameter group list from a ``torch.optim.Optimizer``.
        """
        super().__init__(None)
        self._processed_groups = set()

        self._check_param_groups(param_groups)
        self._param_groups = param_groups
        self._param_groups_ids = [id(group) for group in param_groups]

        self._param_to_group = {
            id(p): id(group) for group in param_groups for p in group["params"]
        }
        self._group_to_params = {
            id(group): [id(p) for p in group["params"]] for group in param_groups
        }

        # accumulate parameter results for each group under ``id(group)``
        self._accumulations = {}
        # store group result under ``id(group)``
        self._output = {}

    def get_output(self, group, pop=True):
        """Return the computation result for a specific parameter group.

        Args:
            group (dict): Parameter group of a ``torch.optim.Optimizer``.
            pop (bool, optional): Remove the result for that group from the
                internal buffer. Default: ``True``.

        Returns:
            Any: Computation result for the parameter group.

        Raises:
            ValueError: If a parameter of the group was not processed. This indicates
                the computation is incomplete, and thus the result may be wrong.
        """
        if not all(id(p) in self.processed for p in group["params"]):
            raise ValueError("Group contains unprocessed parameters.")
        else:
            group_id = id(group)
            if pop:
                return self._output.pop(group_id)
            else:
                return self._output[group_id]

    def param_hook(self, param):
        """Perform parameter computation. Accumulate result in ``self._accumulations``.

        Args:
            param (torch.Tensor): Parameter of a neural net.
        """
        param_id = id(param)
        group_id = self._param_to_group[param_id]

        result = self.param_computation(param)
        self._accumulate_param_computation(result, group_id)

        if self.should_run_group_hook(param):
            self.run_group_hook(group_id)

    def run_group_hook(self, group_id):
        """Execute group hook after results from parameters have been accumulated.

        Saves the result in ``self._output`` under the group id.

        Args:
            group_id (int): Parameter group id.
        """
        accumulation = self._accumulations.pop(group_id)
        group = self.get_group(group_id)
        group_result = self.group_hook(accumulation, group)
        self._output[group_id] = group_result
        self._processed_groups.add(group_id)

    def get_group(self, group_id):
        """Return the parameter group from its ID.

        Args:
            group_id (int): ID of parameter group.

        Returns:
            group (dict): Entry of a ``torch.optim.Optimizer``'s parameter group.
        """
        idx = self._param_groups_ids.index(group_id)

        return self._param_groups[idx]

    def should_run_hook(self, param, module):
        """Check if hooks should be executed on a parameter.

        In addition to the parent class conditions, only execute the hook on a
        parameter that is contained in one of the parameter groups.

        Args:
            param (torch.Tensor): Parameter of a neural net.
            module (torch.nn.Module): Layer that `param` is part of.

        Returns:
            bool: Whether the hook should be executed on the parameter.
        """
        param_in_groups = id(param) in self._param_to_group.keys()

        return param_in_groups and super().should_run_hook(param, module)

    def should_run_group_hook(self, param):
        """Check if hooks should be executed on the parameter's group.

        The earliest possible for a group hook to be executed is when all other
        parameters in ``param``'s group have already been processed.

        Args:
            param (torch.Tensor): Parameter of a neural net.

        Returns:
            bool: Whether the group hook should be executed.
        """
        param_id = id(param)
        group_id = self._param_to_group[param_id]

        group_param_ids = self._group_to_params[group_id]
        other_param_ids = [p_id for p_id in group_param_ids if p_id != param_id]

        last_missing = param_id not in self.processed and all(
            p_id in self.processed for p_id in other_param_ids
        )
        return last_missing

    def _accumulate_param_computation(self, result, group_id):
        """Accumulate output of parameter computation in the group cache.

        Args:
            result (torch.Tensor): Result from parameter computation.
            group_id (int): Parameter group id.
        """
        if group_id not in self._accumulations.keys():
            updated = result
        else:
            existing = self._accumulations[group_id]
            updated = self.accumulate(existing, result)

        self._accumulations[group_id] = updated

    def _check_param_groups(self, param_groups):
        """Check parameter groups.

        Args:
            param_groups (list): Parameter group list from a ``torch.optim.Optimizer``.

        Raises:
            ValueError: If parameters occur in multiple groups.
        """
        param_ids = [id(p) for group in param_groups for p in group["params"]]

        if len(param_ids) != len(set(param_ids)):
            raise ValueError("Same parameters occur in different groups")

    @classmethod
    def from_functions(
        cls, param_groups, param_computation_fn, group_hook_fn, accumulate_fn
    ):
        """Generate parameter group hook by specifying the child class methods.

        Args:
            param_groups (list): Parameter group list from a ``torch.optim.Optimizer``.
            param_computation_fn (function): Function with same signature as
                ``param_computation``. Represents the computation that is carried out
                for every individual parameter.
            group_hook_fn (function): Function with same signature as ``group_hook``.
                Represents the computation that is carried out for every group.
            accumulate_fn (function): Function with same signature as ``accumulate``.
                Represents how parameter results are accumulated for a group.

        Returns:
            ParameterGroupsHook: Handles parameter-wise accumulation and group-wise
                post-processing of results during backpropagation.
        """
        hook = cls(param_groups)

        # bind functions to instance's unimplemented methods
        hook.param_computation = _bind(param_computation_fn, hook)
        hook.group_hook = _bind(group_hook_fn, hook)
        hook.accumulate = _bind(accumulate_fn, hook)

        return hook


def _bind(method, instance):
    """Bind a method to an instance.

    For details, see https://stackoverflow.com/a/37455782.

    Args:
        method (function): Method to be bound. Must accept ``self`` as first argument.
        instance (SomeClass): An instantiated class.

    Returns:
        MethodType: Bound method.
    """
    return types.MethodType(method, instance)
