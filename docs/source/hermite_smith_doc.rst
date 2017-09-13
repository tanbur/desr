Hermite and Smith Normal Forms
==============================

desr uses the diophantine package, which in turn uses the methods found in :cite:`Havas1998`, to calculate the Hermite normal form of matrices.

matrix_normal_forms should be used for all normal form calculations - we never call the diophantine package directly from other parts of desr.

This is also where the Smith normal form functions live, which use Hermite normal forms at their core.

Hermite Normal Forms
--------------------

.. autofunction:: desr.matrix_normal_forms.is_hnf_row
    :noindex:

.. autofunction:: desr.matrix_normal_forms.is_hnf_col
    :noindex:

.. autofunction:: desr.matrix_normal_forms.hnf_row
    :noindex:

.. autofunction:: desr.matrix_normal_forms.hnf_col
    :noindex:

.. autofunction:: desr.matrix_normal_forms.normal_hnf_row
    :noindex:

.. autofunction:: desr.matrix_normal_forms.normal_hnf_col
    :noindex:

Smith Normal Form
-----------------

.. autofunction:: desr.matrix_normal_forms.is_smf
    :noindex:

.. autofunction:: desr.matrix_normal_forms.smf
    :noindex:



