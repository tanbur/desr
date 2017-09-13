ODETranslation and Reduction of ODESystems
==========================================

The :class:`~desr.ode_translation.ODETranslation` is the class used to represent the scaling symmetries of a system.

Creating ODE Translations
-------------------------

.. autoclass:: desr.ode_translation.ODETranslation
.. automethod:: desr.ode_translation.ODETranslation.from_ode_system

Reduction Methods
-----------------

.. automethod:: desr.ode_translation.ODETranslation.translate
.. automethod:: desr.ode_translation.ODETranslation.translate_parameter
.. automethod:: desr.ode_translation.ODETranslation.translate_dep_var
.. automethod:: desr.ode_translation.ODETranslation.translate_general

Reverse Translation
-------------------

Reverse translation is the process of taking solutions of the reduced system and recovering solutions of the original system.

.. automethod:: desr.ode_translation.ODETranslation.reverse_translate
.. automethod:: desr.ode_translation.ODETranslation.reverse_translate_parameter
.. automethod:: desr.ode_translation.ODETranslation.reverse_translate_dep_var
.. automethod:: desr.ode_translation.ODETranslation.reverse_translate_general

Extending from a set of Invariants
----------------------------------

.. automethod:: desr.ode_translation.ODETranslation.extend_from_invariants

Useful Attributes
-----------------

.. automethod:: desr.ode_translation.ODETranslation.invariants
.. automethod:: desr.ode_translation.ODETranslation.auxiliaries
.. automethod:: desr.ode_translation.ODETranslation.rewrite_rules


Output Functions
----------------

.. automethod:: desr.ode_translation.ODETranslation.to_tex

Advanced Methods
----------------

These methods will be familiar to those who use Lie groups to analyse more general symmetries of differential equations.
For more information, see :cite:`Fels1999` or :cite:`Hubert2007`.

.. automethod:: desr.ode_translation.ODETranslation.moving_frame
.. automethod:: desr.ode_translation.ODETranslation.rational_section