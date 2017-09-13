Hermite and Smith Normal Forms
==============================

desr uses the diophantine package, which in turn uses the methods found in :cite:`Havas1998`, to calculate the Hermite normal form of matrices.

matrix_normal_forms should be used for all normal form calculations - we never call the diophantine package directly from other parts of desr.

This is also where the Smith normal form functions live, which use Hermite normal forms at their core.

matrix_normal_forms.py
----------------------

.. automodule:: desr.matrix_normal_forms
    :members:
    :undoc-members:
    :show-inheritance:

diophantine.py
--------------

.. automodule:: desr.diophantine
    :members:
    :undoc-members:
    :show-inheritance:
