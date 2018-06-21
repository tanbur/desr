

Example Michael-Mentis
======================

We perform an example analysis of the Michael-Mentis equations :cite:`Segel1989`:

.. math::
    :nowrap:

    \begin{align}
    \frac{dE}{dt} &= - k_1 E S + k_{-1} C + k_2 C \\
    \frac{dS}{dt} &= - k_1 E S + k_{-1} C \\
    \frac{dC}{dt} &= k_1 E S - k_{-1} C - k_2 C \\
    \frac{dP}{dt} &= k_2 C
    \end{align}

First we initiate the system from LaTeX and find the maximal scaling matrix such that the system is invariant.
Note that negative subscripts are turned into 'm' so that they comply with :py:mod:`sympy`. The 'm's are turned back into
negatives when printint to LaTeX using :func:`desr.tex_tools`.

    >>> import sympy
    >>> from desr.matrix_normal_forms import smf
    >>> from desr.ode_system import ODESystem
    >>> from desr.ode_translation import ODETranslation, scale_action
    >>> from desr.tex_tools import expr_to_tex
    >>> system_tex = '''\frac{dE}{dt} &= - k_1 E S + k_{-1} C + k_2 C \\\\
    ...                 \frac{dS}{dt} &= - k_1 E S + k_{-1} C \\\\
    ...                 \frac{dC}{dt} &= k_1 E S - k_{-1} C - k_2 C \\\\
    ...                 \frac{dP}{dt} &= k_2 C'''
    >>> original_system = ODESystem.from_tex(system_tex)
    >>> max_scal1 = ODETranslation.from_ode_system(original_system)
    >>> print 'Variable order: ', max_scal1.variables_domain
    Variable order:  (t, C, E, P, S, k_1, k_2, k_m1)
    >>> print 'Scaling Matrix:\n', max_scal1.scaling_matrix.__repr__()
    Scaling Matrix:
    Matrix([
    [1, 0, 0, 0, 0, -1, -1, -1],
    [0, 1, 1, 1, 1, -1,  0,  0]])

Now we can inspect the invariants easily:

    >>> print 'Invariants: ', max_scal1.invariants()
    Invariants:  Matrix([[k_m1*t, C*k_1/k_m1, E*k_1/k_m1, P*k_1/k_m1, S*k_1/k_m1, k_2/k_m1]])

Finding the reduced system is also easy.
Since the Hermite multiplier and inverse are compatible with the simplest parameter reduction scheme,
:meth:`~desr.ode_translation.ODETranslation.translate` will automatically perform this reduction.

    >>> print 'Reduced system:\n', max_scal1.translate(original_system)
    Reduced system:
    dt/dt = 1
    dC/dt = -C*c0 - C + E*S
    dE/dt = C*c0 + C - E*S
    dP/dt = C*c0
    dS/dt = C - E*S
    dc0/dt = 0



Changing the variable order
---------------------------

In our previous example, we had :math:`k_{-1}` at the end of the variable order, so that the algorithm tries to normalise using :math:`k_{-1}`.
Instead, we can choose to normalise by :math:`k_2`, by swapping around the last two variables.
Note that we need to recalculate the :class:`~desr.ode_translation.ODETranslation` instance.

    >>> original_system_reorder = original_system.copy()
    >>> variable_order = list(original_system.variables)
    >>> variable_order[-1], variable_order[-2] = variable_order[-2], variable_order[-1]  # Swap the last two variables
    >>> original_system_reorder.reorder_variables(variable_order)
    >>> original_system_reorder.variables
    (t, C, E, P, S, k_1, k_m1, k_2)
    >>> max_scal1_reorder = ODETranslation.from_ode_system(original_system_reorder)
    >>> print 'Invariants:', ', '.join(map(str, max_scal1_reorder.invariants()))
    Invariants: k_2*t, C*k_1/k_2, E*k_1/k_2, P*k_1/k_2, S*k_1/k_2, k_m1/k_2

Now we can reduce to find another, equivalent system.

    >>> reduced_system = max_scal1_reorder.translate(original_system_reorder)
    >>> reduced_system
    dt/dt = 1
    dC/dt = -C*c0 - C + E*S
    dE/dt = C*c0 + C - E*S
    dP/dt = C
    dS/dt = C*c0 - E*S
    dc0/dt = 0

Extending a choice of invariants
--------------------------------

We return to our original variable order: :math:`t, C, E, P, S, k_1, k_2, k_{-1}`.

Suppose we wish to study the invariants :math:`\frac{k_1}{k_{2}}C` and :math:`\frac{k_1}{k_{-1}}P`.
Then we must create a matrix representing these invariants:

.. math::

    P = \left[\begin{matrix}0 & 0\\1 & 0\\0 & 0\\0 & 1\\0 & 0\\1 & 1\\-1 & 0\\0 & -1\end{matrix}\right].

We can easily check we have correct matrix:

    >>> invariant_choice = sympy.Matrix([[0, 1, 0, 0, 0, 1, -1, 0],
    ...                                  [0, 0, 0, 1, 0, 1, 0, -1]]).T
    >>> scale_action(max_scal1.variables_domain, invariant_choice)
    Matrix([[C*k_1/k_2, P*k_1/k_m1]])

Finding a maximal scaling matrix that can be used to rewrite the system in terms of these invariants is also simple.

    >>> max_scal2 = max_scal1.extend_from_invariants(invariant_choice=invariant_choice)
    >>> max_scal2
    A=
    Matrix([
    [1, 0, 0, 0, 0, -1, -1, -1],
    [0, 1, 1, 1, 1, -1,  0,  0]])
    V=
    Matrix([
    [ 0,  0,  0,  0, 1,  0,  0,  0],
    [ 0,  0,  1,  0, 0,  0,  0,  0],
    [ 0,  0,  0,  0, 0,  1,  0,  0],
    [ 0,  0,  0,  1, 0,  0,  0,  0],
    [ 0,  0,  0,  0, 0,  0,  1,  0],
    [ 0, -1,  1,  1, 0,  1,  1,  0],
    [ 0,  0, -1,  0, 0,  0,  0,  1],
    [-1,  1,  0, -1, 1, -1, -1, -1]])
    W=
    Matrix([
    [1, 0, 0, 0, 0, -1, -1, -1],
    [0, 1, 1, 1, 1, -1,  0,  0],
    [0, 1, 0, 0, 0,  0,  0,  0],
    [0, 0, 0, 1, 0,  0,  0,  0],
    [1, 0, 0, 0, 0,  0,  0,  0],
    [0, 0, 1, 0, 0,  0,  0,  0],
    [0, 0, 0, 0, 1,  0,  0,  0],
    [0, 1, 0, 0, 0,  0,  1,  0]])

For Python code that steps through this procedure, see :py:mod:`desr.examples.example_michael_mentis`.

Now, this transformation doesn't satisfy the conditions of the parameter reduction scheme, so if we try to reduce it
:meth:`~desr.ode_translation.ODETranslation.translate` will use the dependent reduction scheme implemented in
:meth:`~desr.ode_translation.ODETranslation.translate_dep_var`.

    >>> max_scal2.invariants()
    Matrix([[C*k_1/k_2, P*k_1/k_m1, k_m1*t, E*k_1/k_m1, S*k_1/k_m1, k_2/k_m1]])
    >>> max_scal2.translate(original_system)
    dt/dt = 1
    dx0/dt = 0
    dx1/dt = 0
    dy0/dt = y0*(-y2*y5 - y2 + y2*y3*y4/(y0*y5))/t
    dy1/dt = y0*y2*y5**2/t
    dy2/dt = y2/t
    dy3/dt = y3*(y0*y2*y5**2/y3 + y0*y2*y5/y3 - y2*y4)/t
    dy4/dt = y4*(y0*y2*y5/y4 - y2*y3)/t
    dy5/dt = 0

Here, :code:`x0` and :code:`x1` are auxiliary variables, which can be fixed at any value at all.
:code:`(y0, y1, y2, y3, y4) = (C*k_1/k_2, P*k_1/k_m1, k_m1*t, E*k_1/k_m1, S*k_1/k_m1)` are our new dependent invariants.
Finally, :code:`y5 = k_2/k_m1` is the single parameter of the reduced system.

However, we can see that after performing a permutation of the columns, we can satisfy the parameter reduction scheme.
While this isn't implemented yet, we can do it by hand for the moment. We must apply the cycle
:math:`\begin{pmatrix}0 & 1 & 3 & 2\end{pmatrix}`
to the last :math:`n-r` columns.

    >>> max_scal3 = max_scal2.herm_mult_n
    >>> max_scal3.col_swap(0, 1)
    >>> max_scal3.col_swap(0, 3)
    >>> max_scal3.col_swap(0, 2)
    >>> print 'Permuted Vn:\n', max_scal3.__repr__()
    Permuted Vn:
    Matrix([
    [1,  0,  0,  0,  0,  0],
    [0,  1,  0,  0,  0,  0],
    [0,  0,  1,  0,  0,  0],
    [0,  0,  0,  1,  0,  0],
    [0,  0,  0,  0,  1,  0],
    [0,  1,  1,  1,  1,  0],
    [0, -1,  0,  0,  0,  1],
    [1,  0, -1, -1, -1, -1]])
    >>> max_scal3 = sympy.Matrix.hstack(max_scal1.herm_mult_i, max_scal3)
    >>> max_scal3 = ODETranslation(max_scal1.scaling_matrix, hermite_multiplier=max_scal3)
    >>> print max_scal3.translate(original_system)
    dt/dt = 1
    dC/dt = -C*c0 - C + E*S/c0
    dE/dt = C*c0**2 + C*c0 - E*S
    dP/dt = C*c0**2
    dS/dt = C*c0 - E*S
    dc0/dt = 0

So we have found a third different reparametrization of the Michaelis-Menten equations.

.. todo::

    Add a method to :class:`~desr.ode_translation.ODETranslation` that will try and re-order the last :math:`n-r` columns so
    that the parameter reduction scheme can be applied.
