"""Base classes for hooks.

TODO This is a copy of
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
        """
        raise NotImplementedError

    def __init__(self, savefield):
        self.savefield = savefield
        self.processed = set()

    def __call__(self, module):
        """Execute hook on all module parameters. Skip already processes parameters."""
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
        """Execute the hook on parameter, add it to processed items and store result."""
        value = self.module_hook(param, module)
        self._save(value, param)
        self.processed.add(id(param))

    def _save(self, value, param):
        """Store value in parameter's ``savefield`` argument."""
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
        """
        raise NotImplementedError

    def module_hook(self, param, module):
        """Extract info from a parameter during backpropagation with BackPACK.

        Args:
            param (torch.Tensor): Parameter of a neural net.
            module (torch.nn.Module): Layer that `param` is part of.

        Return:
            Arbitrary output which will be stored in ``param``'s attribute
            ``self.savefield``.
        """
        return self.param_hook(param)
