Computations
============


GGN eigenvalues
---------------

.. autoclass:: vivit.EigvalshComputation
   :members: __init__, get_extension, get_extension_hook, get_result

GGN eigenpairs (eigenvalues + eigenvector)
------------------------------------------

.. autoclass:: vivit.EighComputation
   :members: __init__, get_extension, get_extension_hook, get_result

1ˢᵗ- and 2ⁿᵈ-order directional derivatives along GGN eigenvectors
-----------------------------------------------------------------

.. autoclass:: vivit.DirectionalDerivativesComputation
   :members: __init__, get_extensions, get_extension_hook, get_result

Directionally damped Newton steps
---------------------------------

.. autoclass:: vivit.DirectionalDampedNewtonComputation
   :members: __init__, get_extensions, get_extension_hook, get_result
