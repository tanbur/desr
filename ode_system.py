import itertools
import numpy
import re

import sympy

from hermite_helper import INT_TYPE_DEF, hnf_col, hnf_row, normal_hnf_col
from sympy_helper import expressions_to_variables, unique_array_stable, monomial_to_powers
from tex_tools import expr_to_tex, var_to_tex

class ODESystem(object):
    ''' A class which represents a system of differential equations '''

    def __init__(self, variables, derivatives, indep_var=None):
        self._variables = tuple(variables)
        self._derivatives = tuple(derivatives)

        self._indep_var = sympy.var('t') if indep_var is None else indep_var

        assert len(self._variables) == len(self._derivatives)
        assert self.derivatives[self.indep_var_index] == sympy.sympify(1)

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

    @property
    def indep_var(self):
        return self._indep_var

    @property
    def indep_var_index(self):
        return self.variables.index(self.indep_var)

    @property
    def variables(self):
        ''' Getter for an ordered tuple of variables '''
        return self._variables

    @property
    def num_constants(self):
        ''' Return the number of constant variables - specifically those which have a None derivative '''
        return sum([1 if deriv is None else 0 for deriv in self._derivatives])

    @property
    def derivatives(self):
        ''' Getter for an ordered tuple of expressions representing the derivatives of self.variables '''
        return [expr if expr is not None else sympy.sympify(0) for expr in self._derivatives]

    @property
    def derivative_dict(self):
        ''' Return a variable: expr mapping, filtering out the Nones in expr '''
        return dict(filter(lambda x: x[1] is not None, zip(self.variables, self._derivatives)))

    @classmethod
    def from_equations(cls, equations, indep_var=sympy.var('t')):
        ''' Instantiate from a text of equations '''
        if isinstance(equations, str):
            equations = equations.strip().split('\n')

        deriv_dict = dict(map(parse_de, equations))
        return cls.from_dict(deriv_dict=deriv_dict)

    @classmethod
    def from_dict(cls, deriv_dict, indep_var=sympy.var('t')):
        ''' Instantiate from a text of equations '''
        all_var = sorted(expressions_to_variables(deriv_dict.values()), key=str)
        dep_var = sorted(deriv_dict.keys(), key=str)
        const_var = sorted(set(all_var).difference(dep_var).difference(set([indep_var])), key=str)
        # Order variables as independent, dependent, parameters
        variables = [indep_var] + dep_var + const_var
        variables = unique_array_stable(variables)

        assert deriv_dict.get(indep_var) is None
        deriv_dict[indep_var] = sympy.sympify(1)

        return cls(variables, tuple([deriv_dict.get(var) for var in variables]))


    def __repr__(self):
        lines = ['d{}/d{} = {}'.format(var, self.indep_var, expr) for var, expr in zip(self.variables, self.derivatives)]
        return '\n'.join(lines)

    def to_tex(self):
        ''' Return a tex version '''
        line_template = '\\frac{{d{}}}{{d{}}} &= {}'
        lines = [line_template.format(var_to_tex(var), var_to_tex(self.indep_var), expr_to_tex(expr))
                 for var, expr in zip(self.variables, self.derivatives)]
        return ' \\\\\n'.join(lines)

    def _power_matrix(self):
        ''' Determine the 'power' matrix of the system, by gluing together the power matrices of each derivative
            expression

        '''
        exprs = [self._indep_var * expr / var for var, expr in self.derivative_dict.iteritems()]
        matrices = [rational_expr_to_power_matrix(expr, self.variables) for expr in exprs]
        out = numpy.hstack(matrices)
        assert out.shape[0] == len(self.variables)
        return out

    def maximal_scaling_matrix(self):
        ''' Determine the maximal scaling matrix leaving this system invariant '''
        exprs = [self._indep_var * expr / var for var, expr in self.derivative_dict.iteritems()]
        return maximal_scaling_matrix(exprs, variables=self.variables)

    def reorder_variables(self, variables):
        ''' Reorder the equation according to the new order of variables '''
        if isinstance(variables, basestring):
            if ' ' in variables:
                variables = variables.split(' ')
            else:
                variables = tuple(variables)
        assert sorted(map(str, variables)) == sorted(map(str, self.variables))
        column_shuffle = []
        for new_var in variables:
            for i, var in enumerate(self.variables):
                if str(var) == str(new_var):
                    column_shuffle.append(i)
        self._variables = tuple(numpy.array(self._variables)[column_shuffle])
        self._derivatives = tuple(numpy.array(self._derivatives)[column_shuffle])


def parse_de(diff_eq, indep_var='t'):
    ''' Parse a first order ordinary differential equation and return (variable of derivative, rational function

        >>> parse_de('dn/dt = n( r(1 - n/K) - kp/(n+d) )')
        (n, n(-kp/(d + n) + r(1 - n/K)))

        >>> parse_de('dp/dt==sp(1 - hp / n)')
        (p, sp(-hp/n + 1))
    '''
    diff_eq = diff_eq.strip()
    ##TODO Replace variable( with variable*(
    match = re.match(r'd([a-zA-Z0-9]*)/d([a-zA-Z0-9]*)\s*=*\s*(.*)', diff_eq)
    if match.group(2) != indep_var:
        raise ValueError('We only work in ordinary DEs in {}'.format(indep_var))
    return sympy.var(match.group(1)), sympy.sympify(match.group(3))

def rational_expr_to_power_matrix(expr, variables):
    ''' Take a rational expression and determine the power matrix wrt an ordering on the variables, as on page 497 of
        Hubert-Labahn.

        >>> exprs = map(sympy.sympify, "n*( r*(1 - n/K) - k*p/(n+d) );s*p*(1 - h*p / n)".split(';'))
        >>> variables = sorted(expressions_to_variables(exprs), key=str)
        >>> rational_expr_to_power_matrix(exprs[0], variables)
        array([[-1,  0,  0,  0, -1,  0],
               [ 0,  1,  0,  0,  1,  1],
               [ 0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  1,  0,  0],
               [ 2,  0,  1,  0,  1, -1],
               [ 0,  0,  0,  1,  0,  0],
               [ 1,  1,  1,  0,  1,  0],
               [ 0,  0,  0,  0,  0,  0]])

        >>> rational_expr_to_power_matrix(exprs[1], variables)
        array([[ 0,  0],
               [ 0,  0],
               [ 1,  0],
               [ 0,  0],
               [-1,  0],
               [ 2,  1],
               [ 0,  0],
               [ 1,  1]])
    '''
    expr = expr.cancel()
    num, denom = expr.as_numer_denom()
    num_const, num_terms = num.as_coeff_add()
    denom_const, denom_terms = denom.as_coeff_add()

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

    powers = numpy.array(powers, dtype=INT_TYPE_DEF).T
    return powers

def maximal_scaling_matrix(exprs, variables=None):
    ''' Determine the maximal scaling matrix leaving this system invariant

    >>> exprs = ['z_1*z_3', 'z_1*z_2 / (z_3 ** 2)']
    >>> exprs = map(sympy.sympify, exprs)
    >>> maximal_scaling_matrix(exprs)
    array([[-1,  3,  1]])

    >>> exprs = ['(z_1 + z_2**2) / z_3']
    >>> exprs = map(sympy.sympify, exprs)
    >>> maximal_scaling_matrix(exprs)
    array([[-2, -1, -2]])
    '''
    if variables is None:
        variables = sorted(expressions_to_variables(exprs), key=str)
    matrices = [rational_expr_to_power_matrix(expr, variables) for expr in exprs]
    power_matrix = numpy.hstack(matrices)
    power_matrix.shape[0] == len(variables)

    hermite_rform, multiplier_rform = hnf_row(power_matrix)

    # Find the non-zero rows at the bottom
    row_is_zero = [numpy.all(row == 0) for row in hermite_rform]
    # Make sure they all come at the end
    num_nonzero = sum(map(int, row_is_zero))
    if num_nonzero == 0:
        return numpy.zeros((1, len(variables)))
    assert numpy.all(hermite_rform[-num_nonzero:] == 0)

    # Make sure we have the right number of columns
    assert multiplier_rform.shape[1] == len(variables)
    # Return the last num_nonzero rows of the Hermite multiplier
    return multiplier_rform[-num_nonzero:]

if __name__ == '__main__':
    import doctest
    doctest.testmod()