Systems of ODEs
===============

The :class:`~desr.ode_system.ODESystem` class is what we use to represent our system of ordinary differential equations.

Creating ODESystems
-------------------

There are a number of different ways of creating an :class:`~desr.ode_system.ODESystem`.

.. autoclass:: desr.ode_system.ODESystem
    :noindex:

.. automethod:: desr.ode_system.ODESystem.from_equations
    :noindex:
.. automethod:: desr.ode_system.ODESystem.from_dict
    :noindex:
.. automethod:: desr.ode_system.ODESystem.from_tex
    :noindex:

Finding Scaling Actions
-----------------------

.. automethod:: desr.ode_system.ODESystem.power_matrix
    :noindex:
.. automethod:: desr.ode_system.ODESystem.maximal_scaling_matrix
    :noindex:

Output Functions
----------------

There are a number of useful ways to output the system.

.. autoattribute:: desr.ode_system.ODESystem.derivative_dict
    :noindex:
.. automethod:: desr.ode_system.ODESystem.to_tex
    :noindex:
