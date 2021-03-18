"""Utility functions and base classes for implementing extension hooks.

Base classes for hooks.

TODO This is (partly) a copy of
    https://github.com/f-dangel/cockpit-paper/blob/hooks/backboard/quantities/hooks/base.py#L1-L100 # noqa: B950
and should use that code once it's packaged.
"""

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

    def __init__(self, savefield):
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
        """Store value in parameter's ``savefield`` argument.

        Args:
            value (any): Arbitrary object that will be stored.
            param (torch.nn.Parameter): Parameter the value is attached to.
        """
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


class ExtensionHookManager:
    """Manages execution of multiple hooks during one backpropagation.

    This file is (almost) a copy of
        https://github.com/f-dangel/cockpit-paper/blob/hooks/backboard/hook_manager.py#L1-L50  # noqa: B950
    """

    def __init__(self, *hooks):
        """Store parameter hooks.

        Args:
            hooks ([callable]): List of functions that accept a tensor and perform
                a side effect. The signature is ``torch.Tensor -> None``.
        """
        self.hooks = self._remove_duplicates(hooks)

    def _remove_duplicates(self, hooks):
        """Remove hook instances from the same class.

        Args:
            hooks ([callable]): List of functions that accept a tensor and perform
                a side effect. The signature is ``torch.Tensor -> None``.

        Returns:
            [callable]: Unified list of hook callables.
        """
        self._check(hooks)

        hook_cls = set()
        filtered = []

        for hook in hooks:
            if hook.__class__ not in hook_cls:
                filtered.append(hook)
                hook_cls.add(hook.__class__)

        return filtered

    @staticmethod
    def _check(hooks):
        for hook in hooks:
            if not isinstance(hook, (ModuleHook, ParameterHook)):
                raise ValueError(
                    f"Hooks must be 'Module/ParameterHook' instances . Got {hook}"
                )

    def __call__(self, module):
        """Apply every hook to the module parameters. Skip if already performed.

        This function is handed to the ``backpack`` context manager.

        Args:
            module (torch.nn.Module): The neural network layer that all parameter
                hooks will be applied to.
        """
        for hook in self.hooks:
            hook(module)
