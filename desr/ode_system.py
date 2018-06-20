import itertools
import re

import sympy
from sympy.abc import _clash1

from matrix_normal_forms import hnf_col, hnf_row, normal_hnf_col
from sympy_helper import expressions_to_variables, unique_array_stable, monomial_to_powers
from tex_tools import expr_to_tex, var_to_tex, tex_to_sympy

class ODESystem(object):
    '''
    A system of differential equations.

    The main attributes are :attr:`~desr.ode_system.ODESystem.variables` and :attr:`~desr.ode_system.ODESystem.derivatives`.
    :attr:`~desr.ode_system.ODESystem.variables` is an ordered tuple of variables, which includes the independent variable.
    :attr:`~desr.ode_system.ODESystem.derivatives` is an ordered tuple of the same length that contains the derivatives with respect to :attr:`~desr.ode_system.ODESystem.indep_var`.

    Args:
        variables (tuple of sympy.Symbol): Ordered tuple of variables.
        derivatives (tuple of sympy.Expression): Ordered tuple of derivatives.
        indep_var (sympy.Symbol, optional): Independent variable we are differentiating with respect to.
        initial_conditions (tuple of sympy.Symbol): The initial values of non-constant variables
    '''

    def __init__(self, variables, derivatives, indep_var=None, initial_conditions=None):
        self._variables = tuple(variables)
        self._derivatives = tuple(derivatives)

        self._indep_var = sympy.var('t') if indep_var is None else indep_var

        self._initial_conditions = {}

        assert len(self._variables) == len(self._derivatives)
        assert self.derivatives[self.indep_var_index] == sympy.sympify(1)

        if initial_conditions is not None:
            self.update_initial_conditions(initial_conditions=initial_conditions)


    def __eq__(self, other):
        if type(self) is not type(other):
            return False

        # Compare variables
        self_var = sorted(self.variables, key=str)
        other_var = sorted(other.variables, key=str)
        if self_var != other_var:
            return False

        # Compare derivatives
        self_der, other_der = self.derivative_dict, other.derivative_dict
        for var1, var2 in zip(self_var, other_var):
            der1 = self_der.get(var1)
            der2 = other_der.get(var2)
            if der1 is None:
                if der2 is not None:
                    return False
            else:
                if der2 is None:
                    return False
                if der1.expand() != der2.expand():
                    return False
        # Compare independent variables
        if self._indep_var != other._indep_var:
            return False

        return True

    def copy(self):
        '''
        Returns:
            ODESystem: A copy of the system.
        '''
        return ODESystem(self._variables, self._derivatives, indep_var=self._indep_var)

    @property
    def indep_var(self):
        """
        Return the independent variable.

        Returns:
            sympy.Symbol: The independent variable, which we are differentiating with respect to.
        """
        return self._indep_var

    @property
    def indep_var_index(self):
        """
        Return the independent variable index.

        Return:
             int: The index of :py:attr:`~indep_var` in :py:attr:`~self.variables`.
        """
        return self.variables.index(self.indep_var)

    @property
    def variables(self):
        '''
        Return the variables appearing in the system.

        Returns:
            tuple: Ordered tuple of variables appearing in the system.
        '''
        return self._variables

    @property
    def constant_variables(self):
        '''
        Return the constant variables - specifically those which have a None derivative.

        Returns:
            tuple: The constant variables.
        '''
        return tuple(var for var, deriv in zip(self.variables, self._derivatives) if deriv is None)

    @property
    def non_constant_variables(self):
        '''
        Return the non-constant variables - specifically those which have a derivative that isn't None or 1.

        Returns:
            tuple: The constant variables.

        >>> _input = {'x': 'c_0*x*y', 'y': 'c_1*(1-x)*(1-y)*t'}
        >>> _input = {sympy.Symbol(k): sympy.sympify(v) for k, v in _input.iteritems()}
        >>> system = ODESystem.from_dict(_input)
        >>> system.non_constant_variables
        (x, y)
        '''
        return tuple(var for var, deriv in zip(self.variables, self._derivatives) if
                     ((deriv is not None) and (deriv != 1)))

    @property
    def num_constants(self):
        '''
        Return the number of constant variables - specifically those which have a :const:`None` derivative

        Returns:
            int: Number of non-constant variables.
        '''
        return len(self.constant_variables)

    @property
    def derivatives(self):
        ''' Getter for an ordered tuple of expressions representing the derivatives of self.variables.

        Returns:
            tuple: Ordered tuple of sympy.Expressions.
        '''
        return [expr if expr is not None else sympy.sympify(0) for expr in self._derivatives]

    @property
    def derivative_dict(self):
        '''
        Return a variable: expr mapping, filtering out the :const:`None`'s in expr.

        Returns:
            dict: Keys are non-constant variables, value is the derivative with respect to the independent variable.
        '''
        return dict(filter(lambda x: x[1] is not None, zip(self.variables, self._derivatives)))

    @property
    def initial_conditions(self):
        '''
        Return a variable: initial-value mapping.

        Returns:
            dict: Keys are non-constant variables, value is the constant representing their initial condition.
        '''
        return self._initial_conditions.copy()

    def update_initial_conditions(self, initial_conditions):
        '''
        Update the internal record of initial conditions.

        Args:
            initial_conditions (dict): non-constant variable: initial value constant.

        >>> _input = {'x': 'c_0*x*y', 'y': 'c_1*(1-x)*(1-y)*t'}
        >>> _input = {sympy.Symbol(k): sympy.sympify(v) for k, v in _input.iteritems()}
        >>> system = ODESystem.from_dict(_input)
        >>> system.update_initial_conditions({'x': 'x_0'})
        >>> system.initial_conditions
        {x: x_0}

        >>> system.update_initial_conditions({'c_0': 'k'})
        Traceback (most recent call last):
            ...
        ValueError: Cannot set initial condition k for variable c_0 with derivative None.

        >>> system
        dt/dt = 1
        dx/dt = c_0*x*y
        dy/dt = c_1*t*(-x + 1)*(-y + 1)
        dc_0/dt = 0
        dc_1/dt = 0
        dx_0/dt = 0
        x(0) = x_0
        '''
        non_const_var = self.non_constant_variables
        for variable, init_cond in initial_conditions.items():
            if not isinstance(variable, sympy.Symbol):
                variable = sympy.Symbol(variable)
            if not isinstance(init_cond, sympy.Symbol):
                init_cond = sympy.Symbol(init_cond)
            # We can only set initial conditions of non-constant variables we already know about.
            if variable not in non_const_var:
                raise ValueError('Cannot set initial condition {} for variable {} with derivative {}.'.format(init_cond,
                                                                                                              variable,
                                                                                                              self.derivative_dict.get(variable)))
            if init_cond not in self.variables:
                self._variables = tuple(list(self._variables) + [init_cond])
                self._derivatives = tuple(list(self._derivatives) + [None])
            self._initial_conditions[variable] = init_cond

    @classmethod
    def from_equations(cls, equations, indep_var=sympy.var('t'), initial_conditions=None):
        '''
        Instantiate from multiple equations.

        Args:
            equations (str, iter of str): Equations of the form "dx/dt = expr", optionally seperated by :code:`\\n`.
            indep_var (sympy.Symbol): The independent variable, usually :code:`t`.
            initial_conditions (tuple of sympy.Symbol): The initial values of non-constant variables

        Returns:
            ODESystem: System of equations.

        >>> eqns = ['dx/dt = c_0*x*y', 'dy/dt = c_1*(1-x)*(1-y)']
        >>> ODESystem.from_equations(eqns)
        dt/dt = 1
        dx/dt = c_0*x*y
        dy/dt = c_1*(-x + 1)*(-y + 1)
        dc_0/dt = 0
        dc_1/dt = 0
        >>> eqns = '\\n'.join(['dy/dx = c_0*x*y', 'dz/dx = c_1*(1-y)*z**2'])
        >>> ODESystem.from_equations(eqns, indep_var=sympy.Symbol('x'))
        dx/dx = 1
        dy/dx = c_0*x*y
        dz/dx = c_1*z**2*(-y + 1)
        dc_0/dx = 0
        dc_1/dx = 0
        '''
        if isinstance(equations, str):
            equations = equations.strip().split('\n')

        deriv_dict = dict(map(lambda x: parse_de(x, indep_var=str(indep_var)), equations))
        system = cls.from_dict(deriv_dict=deriv_dict, indep_var=indep_var, initial_conditions=initial_conditions)
        system.default_order_variables()
        return system

    @classmethod
    def from_dict(cls, deriv_dict, indep_var=sympy.var('t'), initial_conditions=None):
        '''
        Instantiate from a text of equations.

        Args:
            deriv_dict (dict): {variable: derivative} mapping.
            indep_var (sympy.Symbol): Independent variable, that the derivatives are with respect to.
            initial_conditions (tuple of sympy.Symbol): The initial values of non-constant variables


        Returns:
            ODESystem: System of ODEs.

        >>> _input = {'x': 'c_0*x*y', 'y': 'c_1*(1-x)*(1-y)'}
        >>> _input = {sympy.Symbol(k): sympy.sympify(v) for k, v in _input.iteritems()}
        >>> ODESystem.from_dict(_input)
        dt/dt = 1
        dx/dt = c_0*x*y
        dy/dt = c_1*(-x + 1)*(-y + 1)
        dc_0/dt = 0
        dc_1/dt = 0

        >>> _input = {'y':  'c_0*x*y', 'z': 'c_1*(1-y)*z**2'}
        >>> _input = {sympy.Symbol(k): sympy.sympify(v) for k, v in _input.iteritems()}
        >>> ODESystem.from_dict(_input, indep_var=sympy.Symbol('x'))
        dx/dx = 1
        dy/dx = c_0*x*y
        dz/dx = c_1*z**2*(-y + 1)
        dc_0/dx = 0
        dc_1/dx = 0
        '''
        # Make a tuple of all variables.
        variables = set(expressions_to_variables(deriv_dict.values())).union(set(deriv_dict.keys()))
        if initial_conditions is not None:
            variables.update(map(expressions_to_variables, initial_conditions.values()))
        variables = tuple(variables.union(set([indep_var])))

        assert ((deriv_dict.get(indep_var) is None) or (deriv_dict.get(indep_var) == 1))
        deriv_dict[indep_var] = sympy.sympify(1)

        system = cls(variables,
                     tuple([deriv_dict.get(var) for var in variables]),
                     indep_var=indep_var,
                     initial_conditions=initial_conditions)
        system.default_order_variables()
        return system

    def __repr__(self):
        lines = ['d{}/d{} = {}'.format(var, self.indep_var, expr) for var, expr in zip(self.variables, self.derivatives)]
        for v in self.non_constant_variables:
            init_cond = self.initial_conditions.get(v)
            if init_cond is not None:
                lines.append('{}(0) = {}'.format(v, init_cond))
        return '\n'.join(lines)

    def to_tex(self):
        '''
        Returns:
            str: TeX representation.


        >>> eqns = ['dC/dt = -C*k_2 - C*k_m1 + E*S*k_1',
        ... 'dE/dt = C*k_2 + C*k_m1 - E*S*k_1',
        ... 'dP/dt = C*k_2',
        ... 'dS/dt = C*k_m1 - E*S*k_1']
        >>> system = ODESystem.from_equations('\\n'.join(eqns))
        >>> print system.to_tex()
        \\frac{dt}{dt} &= 1 \\\\
        \\frac{dC}{dt} &= - C k_{2} - C k_{-1} + E S k_{1} \\\\
        \\frac{dE}{dt} &= C k_{2} + C k_{-1} - E S k_{1} \\\\
        \\frac{dP}{dt} &= C k_{2} \\\\
        \\frac{dS}{dt} &= C k_{-1} - E S k_{1} \\\\
        \\frac{dk_{1}}{dt} &= 0 \\\\
        \\frac{dk_{2}}{dt} &= 0 \\\\
        \\frac{dk_{-1}}{dt} &= 0

        >>> system.update_initial_conditions({'C': 'C_0'})
        >>> print system.to_tex()
        \\frac{dt}{dt} &= 1 \\\\
        \\frac{dC}{dt} &= - C k_{2} - C k_{-1} + E S k_{1} \\\\
        \\frac{dE}{dt} &= C k_{2} + C k_{-1} - E S k_{1} \\\\
        \\frac{dP}{dt} &= C k_{2} \\\\
        \\frac{dS}{dt} &= C k_{-1} - E S k_{1} \\\\
        \\frac{dk_{1}}{dt} &= 0 \\\\
        \\frac{dk_{2}}{dt} &= 0 \\\\
        \\frac{dk_{-1}}{dt} &= 0 \\\\
        \\frac{dC_{0}}{dt} &= 0 \\\\
        C\\left(0\\right) &= C_{0}
        '''
        line_template = '\\frac{{d{}}}{{d{}}} &= {}'
        lines = [line_template.format(var_to_tex(var), var_to_tex(self.indep_var), expr_to_tex(expr))
                 for var, expr in zip(self.variables, self.derivatives)]
        for v in self.non_constant_variables:
            init_cond = self.initial_conditions.get(v)
            if init_cond is not None:
                lines.append('{}\\left(0\\right) &= {}'.format(var_to_tex(v), expr_to_tex(init_cond)))
        return ' \\\\\n'.join(lines)

    @classmethod
    def from_tex(cls, tex):
        """
        Given the LaTeX of a system of differential equations, return a ODESystem of it.

        Args:
            tex (str): LaTeX

        Returns:
            ODESystem: System of ODEs.

        >>> eqns = ['\\frac{dE}{dt} &= - k_1 E S + k_{-1} C + k_2 C \\\\',
        ... '\\frac{dS}{dt} &= - k_1 E S + k_{-1} C \\\\',
        ... '\\frac{dC}{dt} &= k_1 E S - k_{-1} C - k_2 C \\\\',
        ... '\\frac{dP}{dt} &= k_2 C']
        >>> ODESystem.from_tex('\\n'.join(eqns))
        dt/dt = 1
        dC/dt = -C*k_2 - C*k_m1 + E*S*k_1
        dE/dt = C*k_2 + C*k_m1 - E*S*k_1
        dP/dt = C*k_2
        dS/dt = C*k_m1 - E*S*k_1
        dk_1/dt = 0
        dk_2/dt = 0
        dk_m1/dt = 0

        Todo:
            * Allow initial conditions to be set from tex.
        """
        sympification = tex_to_sympy(tex)
        derivative_dict = {}
        indep_var = None
        for sympy_eq in sympification:
            if not isinstance(sympy_eq.lhs, sympy.Derivative):
                raise ValueError('Invalid sympy equation: {}'.format(sympy_eq))
            derivative_dict[sympy_eq.lhs.args[0]] = sympy_eq.rhs

            # Check we always have the same independent variable.
            if indep_var is None:
                indep_var = sympy_eq.lhs.args[1]
            else:
                if indep_var != sympy_eq.lhs.args[1]:
                    raise ValueError('Must be ordinary differential equation. Two indep variables {} and {} found.'.format(indep_var, sympy_eq.lhs.args[1]))

        return cls.from_dict(deriv_dict=derivative_dict)


    def power_matrix(self):
        '''
        Determine the 'exponent' or 'power' matrix of the system, denoted by :math:`K` in the literature,
        by gluing together the power matrices of each derivative.

        In particular, it concatenates :math:`K_{\\left(\\frac{t}{x} \\cdot \\frac{dx}{dt}\\right)}` for :math:`x` in :attr:`~variables`,
        where :math:`t` is the independent variable.

        >>> eqns = '\\n'.join(['ds/dt = -k_1*e_0*s + (k_1*s + k_m1)*c',
        ... 'dc/dt = k_1*e_0*s - (k_1*s + k_m1 + k_2)*c'])
        >>> system = ODESystem.from_equations(eqns)
        >>> system.variables
        (t, c, s, e_0, k_1, k_2, k_m1)
        >>> system.power_matrix()
        Matrix([
        [1, 1, 1,  1, 1, 1,  1],
        [0, 0, 0, -1, 0, 1,  1],
        [1, 0, 0,  1, 0, 0, -1],
        [0, 0, 0,  1, 1, 0,  0],
        [1, 0, 0,  1, 1, 1,  0],
        [0, 1, 0,  0, 0, 0,  0],
        [0, 0, 1,  0, 0, 0,  1]])

        While we get a different answer to the example in the paper, this is just due to choosing our reference exponent in a different way.

        Todo:
            * Change the code to agree with the paper.

        >>> system.update_initial_conditions({'s': 's_0'})
        >>> system.power_matrix()
        Matrix([
        [1, 1, 1,  1, 1, 1,  1,  0],
        [0, 0, 0, -1, 0, 1,  1,  0],
        [1, 0, 0,  1, 0, 0, -1,  1],
        [0, 0, 0,  1, 1, 0,  0,  0],
        [1, 0, 0,  1, 1, 1,  0,  0],
        [0, 1, 0,  0, 0, 0,  0,  0],
        [0, 0, 1,  0, 0, 0,  1,  0],
        [0, 0, 0,  0, 0, 0,  0, -1]])
        '''
        exprs = [self._indep_var * expr / var for var, expr in self.derivative_dict.iteritems() if expr != 1]
        exprs.extend([var / init_cond for var, init_cond in self.initial_conditions.items()])
        matrices = [rational_expr_to_power_matrix(expr, self.variables) for expr in exprs]
        out = sympy.Matrix.hstack(*matrices)
        assert out.shape[0] == len(self.variables)
        return out

    def maximal_scaling_matrix(self):
        '''
        Determine the maximal scaling matrix leaving this system invariant.

        Returns:
            sympy.Matrix: Maximal scaling matrix.


        >>> eqns = '\\n'.join(['ds/dt = -k_1*e_0*s + (k_1*s + k_m1)*c',
        ... 'dc/dt = k_1*e_0*s - (k_1*s + k_m1 + k_2)*c'])
        >>> system = ODESystem.from_equations(eqns)
        >>> system.maximal_scaling_matrix()
        Matrix([
        [1, 0, 0, 0, -1, -1, -1],
        [0, 1, 1, 1, -1,  0,  0]])
        '''
        exprs = [self._indep_var * expr / var for var, expr in self.derivative_dict.iteritems()]
        exprs.extend([var / init_cond for var, init_cond in self.initial_conditions.items()])
        return maximal_scaling_matrix(exprs, variables=self.variables)

    def reorder_variables(self, variables):
        '''
        Reorder the equation according to the new order of variables.

        Args:
            variables (str, iter):
                Another ordering of the variables.

        >>> eqns = ['dz_1/dt = z_1*z_3', 'dz_2/dt = z_1*z_2 / (z_3 ** 2)']
        >>> system = ODESystem.from_equations('\\n'.join(eqns))
        >>> system.variables
        (t, z_1, z_2, z_3)
        >>> system.derivatives
        [1, z_1*z_3, z_1*z_2/z_3**2, 0]

        >>> system.reorder_variables(['z_2', 'z_3', 't', 'z_1'])
        >>> system.variables
        (z_2, z_3, t, z_1)
        >>> system.derivatives
        [z_1*z_2/z_3**2, 0, 1, z_1*z_3]
        '''
        if isinstance(variables, basestring):
            if ' ' in variables:
                variables = variables.split(' ')
            else:
                variables = tuple(variables)
        if not sorted(map(str, variables)) == sorted(map(str, self.variables)):
            raise ValueError('Mismatching variables:\n{} vs\n{}'.format(sorted(map(str, self.variables)), sorted(map(str, variables))))
        column_shuffle = []
        for new_var in variables:
            for i, var in enumerate(self.variables):
                if str(var) == str(new_var):
                    column_shuffle.append(i)
        self._variables = tuple(sympy.Matrix(self._variables).extract(column_shuffle, [0]))
        self._derivatives = tuple(sympy.Matrix(self._derivatives).extract(column_shuffle, [0]))

    def default_order_variables(self):
        '''
        Reorder the variables into (independent variable, dependent variables, constant variables),
        which generally gives the simplest reductions.
        Variables of the same type are sorted by their string representations.


        >>> eqns = ['dz_1/dt = z_1*z_3', 'dz_2/dt = z_1*z_2 / (z_3 ** 2)']
        >>> system = ODESystem.from_equations('\\n'.join(eqns))
        >>> system.variables
        (t, z_1, z_2, z_3)

        >>> system.reorder_variables(['z_2', 'z_3', 't', 'z_1'])
        >>> system.variables
        (z_2, z_3, t, z_1)

        >>> system.default_order_variables()
        >>> system.variables
        (t, z_1, z_2, z_3)
        '''
        all_var = self.variables
        dep_var = sorted(self.derivative_dict.keys(), key=str)
        dep_var.remove(self.indep_var)
        const_var = sorted(set(all_var).difference(dep_var).difference(set([self.indep_var])), key=str)

        # Order variables as independent, dependent, parameters
        variables = [self.indep_var] + dep_var + const_var
        assert len(variables) == len(set(variables))
        self.reorder_variables(variables=variables)

def parse_de(diff_eq, indep_var='t'):
    ''' Parse a first order ordinary differential equation and return (variable of derivative, rational function

        >>> parse_de('dn/dt = n( r(1 - n/K) - kp/(n+d) )')
        (n, n(-kp/(d + n) + r(1 - n/K)))

        >>> parse_de('dp/dt==sp(1 - hp / n)')
        (p, sp(-hp/n + 1))
    '''
    diff_eq = diff_eq.strip()
    match = re.match(r'd([a-zA-Z0-9_]*)/d([a-zA-Z0-9_]*)\s*=*\s*(.*)', diff_eq)
    if match is None:
        raise ValueError("Invalid differential equation: {}".format(diff_eq))
    if match.group(2) != indep_var:
        raise ValueError('We only work in ordinary DEs in {}'.format(indep_var))
    # Feed in _clash1 so that we can use variables S, C, etc., which are special characters in sympy.
    return sympy.var(match.group(1)), sympy.sympify(match.group(3), _clash1)

def rational_expr_to_power_matrix(expr, variables):
    '''
    Take a rational expression and determine the power matrix wrt an ordering on the variables, as on page 497 of
    Hubert-Labahn.

    >>> exprs = map(sympy.sympify, "n*( r*(1 - n/K) - k*p/(n+d) );s*p*(1 - h*p / n)".split(';'))
    >>> variables = sorted(expressions_to_variables(exprs), key=str)
    >>> variables
    [K, d, h, k, n, p, r, s]
    >>> rational_expr_to_power_matrix(exprs[0], variables)
    Matrix([
    [0, -1, -1, 0, 0,  0],
    [0,  1,  0, 1, 0,  1],
    [0,  0,  0, 0, 0,  0],
    [1,  0,  0, 0, 0,  0],
    [0,  1,  2, 0, 1, -1],
    [1,  0,  0, 0, 0,  0],
    [0,  1,  1, 1, 1,  0],
    [0,  0,  0, 0, 0,  0]])

    >>> rational_expr_to_power_matrix(exprs[1], variables)
    Matrix([
    [ 0, 0],
    [ 0, 0],
    [ 1, 0],
    [ 0, 0],
    [-1, 0],
    [ 2, 1],
    [ 0, 0],
    [ 1, 1]])
    '''
    expr = expr.cancel()
    num, denom = expr.as_numer_denom()
    num_const, num_terms = num.as_coeff_add()
    denom_const, denom_terms = denom.as_coeff_add()
    num_terms = sorted(num_terms, key=str)
    denom_terms = sorted(denom_terms, key=str)

    if denom_const != 0:
        ref_power = 1
        # If we have another constant in the numerator, add it onto the terms for processing.
        if num_const != 0:
            num_terms = list(num_terms)
            num_terms.append(num_const)
    else:
        if num_const != 0:
            ref_power = 1
        else:
            denom_terms = list(denom_terms)

            # Find the lowest power
            ref_power = min(denom_terms, key=lambda x: map(abs, monomial_to_powers(x, variables)))

            denom_terms.remove(ref_power)  # Use the last term of the denominator as our reference power

    powers = []
    for mon in itertools.chain(num_terms, denom_terms):
        powers.append(monomial_to_powers(mon / ref_power, variables))

    powers = sympy.Matrix(powers).T
    return powers

def maximal_scaling_matrix(exprs, variables=None):
    ''' Determine the maximal scaling matrix leaving this system invariant, in row Hermite normal form.

    Args:
        exprs (iter): Iterable of sympy.Expressions.
        variables: An ordering on the variables. If None, sort according to the string representation.
    Returns:
        sympy.Matrix

    >>> exprs = ['z_1*z_3', 'z_1*z_2 / (z_3 ** 2)']
    >>> exprs = map(sympy.sympify, exprs)
    >>> maximal_scaling_matrix(exprs)
    Matrix([[1, -3, -1]])

    >>> exprs = ['(z_1 + z_2**2) / z_3']
    >>> exprs = map(sympy.sympify, exprs)
    >>> maximal_scaling_matrix(exprs)
    Matrix([[2, 1, 2]])
    '''
    if variables is None:
        variables = sorted(expressions_to_variables(exprs), key=str)
    matrices = [rational_expr_to_power_matrix(expr, variables) for expr in exprs]
    power_matrix = sympy.Matrix.hstack(*matrices)
    assert power_matrix.shape[0] == len(variables)

    hermite_rform, multiplier_rform = hnf_row(power_matrix)

    # Find the non-zero rows at the bottom
    row_is_zero = [all([i == 0 for i in row]) for row in hermite_rform.tolist()]
    # Make sure they all come at the end
    num_nonzero = sum(map(int, row_is_zero))
    if num_nonzero == 0:
        return sympy.zeros(1, len(variables))
    assert hermite_rform[-num_nonzero:, :].is_zero

    # Make sure we have the right number of columns
    assert multiplier_rform.shape[1] == len(variables)
    # Return the last num_nonzero rows of the Hermite multiplier
    return hnf_row(multiplier_rform[-num_nonzero:, :])[0]


if __name__ == '__main__':
    import doctest
    doctest.testmod()