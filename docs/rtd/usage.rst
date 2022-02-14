How to use ViViT
================

Preliminaries
-------------

ViViT's computational tricks are performed during a backward pass with `BackPACK
<http://www.github.com/f-dangel/backpack>`_. Hence, you must first `extend your
model and loss function
<https://docs.backpack.pt/en/master/main-api.html#extending-the-model-and-loss-function>`_.


Integration with BackPACK
-------------------------

Starting from a working backward pass with BackPACK, you can integrate ViViT as
follows:

1. Instantiate the :code:`...Computations` object for your quantity of interest
   (see :ref:`available computations <Computations>`).

2. Use that object to create the extension and extension hook for BackPACK. Pass
   them as arguments to your :code:`with backpack(...)` context.

3. Request the results from the :code:`...Computations` object after the
   backward pass.

For examples see :ref:`Code samples`.
