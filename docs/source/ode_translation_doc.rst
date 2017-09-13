ODETranslation and Reduction of ODESystems
==========================================

The :class:`~desr.ode_translation.ODETranslation` is the class used to represent the scaling symmetries of a system.

Creating ODE Translations
-------------------------

.. autoclass:: desr.ode_translation.ODETranslation
    :noindex:

.. automethod:: desr.ode_translation.ODETranslation.from_ode_system
    :noindex:

Reduction Methods
-----------------

.. automethod:: desr.ode_translation.ODETranslation.translate
    :noindex:

.. automethod:: desr.ode_translation.ODETranslation.translate_parameter
    :noindex:

.. automethod:: desr.ode_translation.ODETranslation.translate_dep_var
    :noindex:

.. automethod:: desr.ode_translation.ODETranslation.translate_general
    :noindex:

Reverse Translation
-------------------

Reverse translation is the process of taking solutions of the reduced system and recovering solutions of the original system.

.. automethod:: desr.ode_translation.ODETranslation.reverse_translate
    :noindex:

.. automethod:: desr.ode_translation.ODETranslation.reverse_translate_parameter
    :noindex:

.. automethod:: desr.ode_translation.ODETranslation.reverse_translate_dep_var
    :noindex:

.. automethod:: desr.ode_translation.ODETranslation.reverse_translate_general
    :noindex:


Extending from a set of Invariants
----------------------------------

.. automethod:: desr.ode_translation.ODETranslation.extend_from_invariants
    :noindex:


Useful Attributes
-----------------

.. automethod:: desr.ode_translation.ODETranslation.invariants
    :noindex:

.. automethod:: desr.ode_translation.ODETranslation.auxiliaries
    :noindex:

.. automethod:: desr.ode_translation.ODETranslation.rewrite_rules
    :noindex:



Output Functions
----------------

.. automethod:: desr.ode_translation.ODETranslation.to_tex
    :noindex:

Advanced Methods
----------------

These methods will be familiar to those who use Lie groups to analyse more general symmetries of differential equations.
For more information, see :cite:`Fels1999` or :cite:`Hubert2007`.

.. automethod:: desr.ode_translation.ODETranslation.moving_frame
    :noindex:

.. automethod:: desr.ode_translation.ODETranslation.rational_section
    :noindex:
