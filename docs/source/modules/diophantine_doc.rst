Hermite and Smith Normal Forms
==============================

desr uses the diophantine package, which in turn uses the methods found in :cite:`Havas1998`, to calculate the Hermite normal form of matrices.

hermite_helper.py includes some useful wrapper functions for this module in hermite_helper.py. We never call the diophantine package directly from other parts of the library.

This is also where the Smith normal form functions live, which use Hermite normal forms at their core.

hermite_helper.py
-----------------

.. automodule:: hermite_helper
    :members:
    :undoc-members:
    :show-inheritance:

diophantine.py
--------------

.. automodule:: diophantine
    :members:
    :undoc-members:
    :show-inheritance:

.. bibliography:: ../desr.bib