

Example Michaelis-Menten
========================

We derive the analysis of the Michaelis-Menten equations found in :cite:`Segel1989`: in a systematic manner.

After using the law of mass action, we are able to reduce the initial set of equations to

.. math::
    :nowrap:

    \begin{align}
    \frac{ds}{dt} &= - k_1 e_0 s + k_1 c s + k_{-1} c \\
    \frac{dc}{dt} &= k_1 e_0 s - k_1 c s - k_{-1} c - k_2 c \\
    s(0) &= s_0
    \end{align}

First we must create the system.
Before we perform any analysis, we set :math:`K = k_2 + k_{-1}` and eliminate :math:`k_{-1}`.
Why can/do we do this??

    >>> system_tex = '''\frac{ds}{dt} &= - k_1 e_0 s + k_1 c s + k_{-1} c \\\\
    ...          \frac{dc}{dt} &= k_1 e_0 s - k_1 c s - k_{-1} c - k_2 c'''
    >>> system_tex_reduced_km1 = system_tex.replace('k_{-1}', '(K - k_2)')
    >>> reduced_system_km1 = ODESystem.from_tex(system_tex_reduced_km1)

We reorder the variables, so that we try to normalise by later oners.


    >>> reduced_system_km1.reorder_variables(['t', 's', 'c', 'K', 'k_2', 'k_1', 'e_0'])
    >>> reduced_system_km1.variables
    (t, s, c, K, k_2, k_1, e_0)


We can then see the exponent (or power) matrix, maximal scaling matrix and corresponding invariants:

    >>> reduced_system_km1.power_matrix()
    Matrix([
    [ 1, 1,  1, 1, 1, 1,  1],
    [-1, 0, -1, 0, 0, 1,  1],
    [ 1, 0,  1, 1, 0, 0, -1],
    [ 0, 0,  1, 0, 1, 0,  0],
    [ 1, 0,  0, 0, 0, 0,  0],
    [ 0, 1,  0, 1, 0, 1,  1],
    [ 0, 1,  0, 0, 0, 0,  1]])
    >>> max_scal1 = ODETranslation.from_ode_system(reduced_system_km1)
    >>> max_scal1.scaling_matrix
    Matrix([
    [1, 0, 0, -1, -1, -1, 0],
    [0, 1, 1,  0,  0, -1, 1]])
    >>> max_scal1.invariants()
    Matrix([[e_0*k_1*t, s/e_0, c/e_0, K/(e_0*k_1), k_2/(e_0*k_1)]])

The reduced system is also computed:

    >>> max_scal1.translate(reduced_system_km1)
    dt/dt = 1
    dc/dt = -c*c0 - c*s + s
    ds/dt = c*c0 - c*c1 + c*s - s
    dc0/dt = 0
    dc1/dt = 0

However, we need to add in our initial condition for :math:`s`.

    >>> reduced_system_km1.update_initial_conditions({'s': 's_0'})
    >>> max_scal2 = ODETranslation.from_ode_system(reduced_system_km1)
    >>> max_scal2.scaling_matrix
    Matrix([
    [1, 0, 0, -1, -1, -1, 0, 0],
    [0, 1, 1,  0,  0, -1, 1, 1]])
    >>> max_scal2.invariants()
    Matrix([[k_1*s_0*t, s/s_0, c/s_0, K/(k_1*s_0), k_2/(k_1*s_0), e_0/s_0]])
    >>> max_scal2.translate(reduced_system_km1)
    dt/dt = 1
    dc/dt = -c*c0 - c*s + c2*s
    ds/dt = c*c0 - c*c1 + c*s - c2*s
    dc0/dt = 0
    dc1/dt = 0
    dc2/dt = 0
    d1/dt = 0
    s(0) = 1

Some elementary column operations give us equations 8

    >>> max_scal2.multiplier_add_columns(2, -1, 1)  # Scale time by e_0 not s_0
    >>> max_scal2.multiplier_add_columns(4, -1, -1)  # Scale c by e_0
    >>> max_scal2.invariants()
    Matrix([[e_0*k_1*t, s/s_0, c/e_0, K/(k_1*s_0), k_2/(k_1*s_0), e_0/s_0]])
    >>> max_scal2.translate(reduced_system_km1)
    dt/dt = 1
    dc/dt = -c*c0/c2 - c*s/c2 + s/c2
    ds/dt = c*c0 - c*c1 + c*s - s
    dc0/dt = 0
    dc1/dt = 0
    dc2/dt = 0
    d1/dt = 0
    s(0) = 1


We can also scale time by :math:`\epsilon` to get the "inner" equation 11:

    >>> max_scal2.multiplier_add_columns(2, -1, -1)  # Divide time through by epsilon
    >>> max_scal2.invariants()
    Matrix([[k_1*s_0*t, s/s_0, c/e_0, K/(k_1*s_0), k_2/(k_1*s_0), e_0/s_0]])
    >>> max_scal2.translate(reduced_system_km1)
    dt/dt = 1
    dc/dt = -c*c0 - c*s + s
    ds/dt = c*c0*c2 - c*c1*c2 + c*c2*s - c2*s
    dc0/dt = 0
    dc1/dt = 0
    dc2/dt = 0
    d1/dt = 0
    s(0) = 1

What is epsilon is not small?
We can find that $s_0 + K_m$ is an invariant systematically.
So we add a variable $L = s_0 + K_m$.

    >>> # Substitute K_m into the equations
    >>> system_tex_reduced_l = system_tex.replace('k_{-1}', '(K - k_2)').replace('K', 'K_m k_1')
    >>> reduced_system_l = ODESystem.from_tex(system_tex_reduced_l)
    >>> reduced_system_l
    dt/dt = 1
    dc/dt = -c*k_1*s - c*k_2 - c*(K_m*k_1 - k_2) + e_0*k_1*s
    ds/dt = c*k_1*s + c*(K_m*k_1 - k_2) - e_0*k_1*s
    dK_m/dt = 0
    de_0/dt = 0
    dk_1/dt = 0
    dk_2/dt = 0
    >>> reduced_system_l.update_initial_conditions({'s': 's_0'})
    >>> reduced_system_l.add_constraints('L', 's_0 + K_m')

Check that if we keep L at the end, we have the same reduced system as before

    >>> reduced_system_l.reorder_variables(['t', 's', 'c', 'K_m', 'k_2', 'k_1', 'e_0', 'L', 's_0'])
    >>> max_scal = ODETranslation.from_ode_system(reduced_system_l)
    >>> max_scal.scaling_matrix
    Matrix([
    [1, 0, 0, 0, -1, -1, 0, 0, 0],
    [0, 1, 1, 1,  0, -1, 1, 1, 1]])
    >>> max_scal.invariants()
    Matrix([[k_1*s_0*t, s/s_0, c/s_0, K_m/s_0, k_2/(k_1*s_0), e_0/s_0, L/s_0]])
    >>> max_scal.translate(reduced_system_l)
    dt/dt = 1
    dc/dt = -c*c0 - c*s + c2*s
    ds/dt = c*c0 - c*c1 + c*s - c2*s
    dc0/dt = 0
    dc1/dt = 0
    dc2/dt = 0
    d1/dt = 0
    dc3/dt = 0
    s(0) = 1
    c3 == c0 + 1

Now we put L into the mix:

    >>> reduced_system_l.reorder_variables(['t', 's', 'c', 'k_2', 'k_1', 'e_0', 's_0', 'L', 'K_m'])
    >>> max_scal3 = ODETranslation.from_ode_system(reduced_system_l)
    >>> # Scale t correctly to t/t_C = k_1 L t
    >>> max_scal3.multiplier_add_columns(2, -1, 1)
    >>> # Scale s correctly to s / s_0
    >>> max_scal3.multiplier_add_columns(3, -2, -1)
    >>> # Scale c correctly to c / (e_0 s_0 / L)
    >>> max_scal3.multiplier_add_columns(4, 6, -1)
    >>> max_scal3.multiplier_add_columns(4, 7, -1)
    >>> max_scal3.multiplier_add_columns(4, -1, 1)
    >>> # Find kappa = k_{-1} / k_2 = (K_m k_1 / k_2) - 1
    >>> max_scal3.multiplier_negate_column(5)
    >>> # Find epsilon = e_0 / L
    >>> max_scal3.multiplier_add_columns(6, -1, -1)
    >>> # Find sigma = s_0 / K_m
    >>> max_scal3.invariants()
    Matrix([[L*k_1*t, s/s_0, L*c/(e_0*s_0), K_m*k_1/k_2, e_0/L, s_0/K_m, L/K_m]])

We now have:

.. math::
    :nowrap:

    \begin{align}
    c_0 &= \kappa + 1 \\
    c_1 &= \epsilon \\
    c_2 &= \sigma \\
    c_3 &= \frac{L}{K_m}
    \end{align}


Which gives us exactly equations 24 from Segel, after some trivial rearrangement.

    >>> max_scal3.translate(reduced_system_l)
    dt/dt = 1
    dc/dt = -c*c2*s/c3 - c/c3 + s
    ds/dt = c*c1*c2*s/c3 + c*c1/c3 - c*c1/(c0*c3) - c1*s
    dc0/dt = 0
    dc1/dt = 0
    dc2/dt = 0
    dc3/dt = 0
    d1/dt = 0
    s(0) = 1
    c3 == c2 + 1

To get the equations on the other timescale, we need to multiply :math:`Lk_1t` by
:math:`\frac{e_0}{L} \frac{k_2}{K_m*k_1}\frac{K_m}{L}=\frac{c_1}{c_0c_3}`

    >>> max_scal3.multiplier_add_columns(2, 6, 1)
    >>> max_scal3.multiplier_add_columns(2, 5, -1)
    >>> max_scal3.multiplier_add_columns(2, -1, -1)
    >>> max_scal3.invariants()
    Matrix([[e_0*k_2*t/L, s/s_0, L*c/(e_0*s_0), K_m*k_1/k_2, e_0/L, s_0/K_m, L/K_m]])
    >>> max_scal3.translate(reduced_system_l)
    dt/dt = 1
    dc/dt = -c*c0*c2*s/c1 - c*c0/c1 + c0*c3*s/c1
    ds/dt = c*c0*c2*s + c*c0 - c - c0*c3*s
    dc0/dt = 0
    dc1/dt = 0
    dc2/dt = 0
    dc3/dt = 0
    d1/dt = 0
    s(0) = 1
    c3 == c2 + 1



Raw Michaelis-Menten Equation Analysis
--------------------------------------

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

Walkthroughs from Supplementary Information
===========================================

Matching Segel and Slemrod's analysis.

    >>> system_tex = '''\frac{ds}{dt} &= - k_1 e_0 s + k_1 c s + k_{-1} c \\\\
    ... \frac{dc}{dt} &= k_1 e_0 s - k_1 c s - k_{-1} c - k_2 c \\\\'''
    >>> system_mm = ODESystem.from_tex(system_tex)
    >>> system_mm.update_initial_conditions({'s': 's_0'})
    >>> system_mm.add_constraints('K_m', '(k_2 + k_m1) / k_1')
    >>> system_mm.add_constraints('epsilon', 'e_0 / (s_0 + K_m)')
    >>> system_mm.reorder_variables(['t', 's', 'c', 'epsilon', 'k_m1', 'k_2', 'k_1', 'K_m', 'e_0', 's_0'])
    >>> system_mm.variables
    (t, s, c, epsilon, k_m1, k_2, k_1, K_m, e_0, s_0)
    >>> max_scal1 = ODETranslation.from_ode_system(system_mm)
    >>> max_scal1.scaling_matrix
    Matrix([
    [1, 0, 0, 0, -1, -1, -1, 0, 0, 0],
    [0, 1, 1, 0,  0,  0, -1, 1, 1, 1]])
    >>> max_scal1.translate(system_mm)
    dt/dt = 1
    dc/dt = -c*c1 - c*c2 - c*s + c4*s
    ds/dt = c*c1 + c*s - c4*s
    dc1/dt = 0
    dc2/dt = 0
    dc4/dt = 0
    d1/dt = 0
    dc3/dt = 0
    dc0/dt = 0
    s(0) = 1
    c3 == c1 + c2
    c0 == c4/(c3 + 1)


First recreate the original system.


    >>> # Scale t correctly to t/t_C = k_1 L t = e_0 k_1 t / epsilon
    >>> max_scal1.multiplier_add_columns(2, 5, -1)
    >>> max_scal1.multiplier_add_columns(2, -1, 1)
    >>> # Scale s correctly to s / s_0
    >>> # Scale c correctly to c / (e_0 s_0 / L) = c / (s_0 epsilon)
    >>> max_scal1.multiplier_add_columns(4, 5, -1)
    >>> # Find epsilon = e_0 / L
    >>> # Find kappa = k_{-1} / k_2 = (K_m k_1 / k_2) - 1
    >>> max_scal1.multiplier_add_columns(6, 7, -1)
    >>> # Find sigma = s_0 / K_m
    >>> max_scal1.multiplier_negate_column(-2)

Inner equations (21)

    >>> max_scal1.invariants()
    Matrix([[e_0*k_1*t/epsilon, s/s_0, c/(epsilon*s_0), epsilon, k_m1/k_2, k_2/(k_1*s_0), s_0/K_m, e_0/s_0]])
    >>> system_mm_red = max_scal1.translate(system_mm)
    >>> system_mm_red = system_mm_red.diff_subs({sympy.sympify('c2'): sympy.sympify('1 / (sigma * (kappa + 1))'),
    ...                                          sympy.sympify('c4'): sympy.sympify('epsilon * (1 + 1 / sigma)'),
    ...                                          })
    >>> system_mm_red.diff_subs({'c0': 'epsilon',
    ...                          'c3': 'sigma',
    ...                          'c1': 'kappa',
    ...                          's': 'u', 'c': 'v'},
    ...                         subs_constraints=True,
    ...                         expand_after=True,
    ...                         factor_after=True)
    dt/dt = 1
    dc/dt = -(sigma*u*v - sigma*u - u + v)/(sigma + 1)
    ds/dt = epsilon*(kappa*sigma*u*v - kappa*sigma*u - kappa*u + kappa*v + sigma*u*v - sigma*u - u)/((kappa + 1)*(sigma + 1))
    dc0/dt = 0
    dc1/dt = 0
    dc2/dt = 0
    dc4/dt = 0
    d1/dt = 0
    dc3/dt = 0
    dkappa/dt = 0
    dsigma/dt = 0
    depsilon/dt = 0
    s(0) = 1
    1/sigma == c2*kappa + c2
    epsilon == c4/(1 + 1/sigma)


Outer equations (24)

    >>> # Scale t correctly to t/t_S = k_2 epsilon t
    >>> max_scal1.multiplier_add_columns(2, 7, 1)
    >>> max_scal1.multiplier_add_columns(2, -1, -1)
    >>> max_scal1.multiplier_add_columns(2, 5, 2)
    >>> max_scal1.invariants()
    Matrix([[epsilon*k_2*t, s/s_0, c/(epsilon*s_0), epsilon, k_m1/k_2, k_2/(k_1*s_0), s_0/K_m, e_0/s_0]])
    >>> system_mm_red = max_scal1.translate(system_mm)
    >>> system_mm_red = system_mm_red.diff_subs({sympy.sympify('c2'): sympy.sympify('1 / (sigma * (kappa + 1))'),
    ...                                          sympy.sympify('c4'): sympy.sympify('epsilon * (1 + 1 / sigma)'),
    ...                                          })
    >>> system_mm_red.diff_subs({'c0': 'epsilon',
    ...                          'c3': 'sigma',
    ...                          'c1': 'kappa',
    ...                          's': 'u', 'c': 'v'},
    ...                         subs_constraints=True,
    ...                         expand_after=True,
    ...                         factor_after=True)
    dt/dt = 1
    dc/dt = -(kappa + 1)*(sigma*u*v - sigma*u - u + v)/epsilon
    ds/dt = kappa*sigma*u*v - kappa*sigma*u - kappa*u + kappa*v + sigma*u*v - sigma*u - u
    dc0/dt = 0
    dc1/dt = 0
    dc2/dt = 0
    dc4/dt = 0
    d1/dt = 0
    dc3/dt = 0
    dkappa/dt = 0
    dsigma/dt = 0
    depsilon/dt = 0
    s(0) = 1
    1/sigma == c2*kappa + c2
    epsilon == c4/(1 + 1/sigma)
