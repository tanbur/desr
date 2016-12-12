
import numpy
import sympy

from hermite_helper import normal_hnf_col, INT_TYPE_DEF, hnf_col, is_hnf_col
from ode_system import ODESystem

class ODETranslation(object):
    ''' An object used for translating between systems of ODEs according to a scaling matrix '''
    def __init__(self, scaling_matrix, variables_domain=None, hermite_multiplier=None):
        scaling_matrix = numpy.copy(scaling_matrix)
        self._scaling_matrix = scaling_matrix
        self._scaling_matrix_hnf, self._herm_mult = normal_hnf_col(scaling_matrix)

        if hermite_multiplier is not None:
            assert is_hnf_col(numpy.dot(scaling_matrix, hermite_multiplier))
            self._herm_mult = hermite_multiplier

        # Make sure we have an integer inverse
        float_inv = numpy.linalg.inv(self._herm_mult)
        self._inv_herm_mult = float_inv.astype(INT_TYPE_DEF)
        assert numpy.max(numpy.abs(float_inv - self._inv_herm_mult)) < 1e-20


        self._variables_domain = variables_domain

    @property
    def scaling_matrix(self):
        return self._scaling_matrix

    @property
    def r(self):
        return self._scaling_matrix.shape[0]

    @property
    def n(self):
        return self._scaling_matrix.shape[1]

    @property
    def herm_mult(self):
        ''' Better known in Hubert Labahn as V '''
        return numpy.copy(self._herm_mult)

    @property
    def herm_mult_i(self):
        ''' The first component of the Hermite multiplier V = Vi '''
        return self.herm_mult[:, :self.r]

    @property
    def herm_mult_n(self):
        ''' The second component of the Hermite multiplier V = Vn '''
        return self.herm_mult[:, self.r:]


    @property
    def inv_herm_mult(self):
        ''' Better known in Hubert Labahn as W '''
        return numpy.copy(self._inv_herm_mult)

    @property
    def inv_herm_mult_u(self):
        ''' The first component of W = Wu '''
        return self.inv_herm_mult[:self.r]

    @property
    def inv_herm_mult_d(self):
        ''' The second component W = Wd '''
        return self.inv_herm_mult[self.r:]

    @property
    def variables_domain(self):
        return self._variables_domain

    @classmethod
    def from_ode_system(cls, ode_system):
        ''' Create a translation given an ODESystem '''
        return cls(scaling_matrix=ode_system.maximal_scaling_matrix(), variables_domain=ode_system.variables)

    def translate_dep_var(self, system):
        ''' Given a system of ODEs, translate the system into a simplified version. Assume we are only working on
            dependent variables, not time.
        '''
        # First check that our scaling action doesn't act on the independent variable
        assert numpy.all(self.scaling_matrix[:, 0] == 0)
        new_herm_mult = numpy.copy(self.herm_mult)
        col_to_delete, new_herm_mult = new_herm_mult[0], new_herm_mult[1:]
        new_herm_mult = new_herm_mult[:, ~col_to_delete.astype(bool)]
        reduced_scaling = ODETranslation(scaling_matrix=self.scaling_matrix[:, 1:],
                                         variables_domain=self.variables_domain,
                                         hermite_multiplier=new_herm_mult)

        # y = numpy.array(scale_action(system.variables, self.herm_mult_n))
        #print 'y = ', numpy.array(scale_action(system.variables, self.herm_mult_n))
        num_inv_var = reduced_scaling.herm_mult_n.shape[1]
        invariant_variables = sympy.var(' '.join(['y{}'.format(i) for i in xrange(num_inv_var)]))
        if num_inv_var == 1:
            invariant_variables = [invariant_variables]
        else:
            invariant_variables = list(invariant_variables)

        # x = numpy.array(scale_action(system.variables, self.herm_mult_i))
        #print 'x = ', numpy.array(scale_action(system.variables, self.herm_mult_i))
        num_aux_var = reduced_scaling.herm_mult_i.shape[1]
        auxiliary_variables = sympy.var(' '.join(['x{}'.format(i) for i in xrange(num_aux_var)]))
        if num_aux_var == 1:
            auxiliary_variables = [auxiliary_variables]
        else:
            auxiliary_variables = list(auxiliary_variables)

        to_sub = dict(zip(system.variables[1:], scale_action(invariant_variables, reduced_scaling.inv_herm_mult_d)))
        system_derivatives = system.derivatives
        # fywd = F(y^(W_d)) in Hubert Labahn
        fywd = numpy.array([(f / v).subs(to_sub, simultaneous=True).expand().simplify() for v, f in
                            zip(system.variables, system_derivatives)])[1:]
        dydt = invariant_variables * numpy.dot(fywd, reduced_scaling.herm_mult_n)
        dxdt = auxiliary_variables * numpy.dot(fywd, reduced_scaling.herm_mult_i)

        new_variables = [system.indep_var] + list(auxiliary_variables) + list(invariant_variables)
        new_derivatives = [sympy.sympify(1)] + list(dxdt) + list(dydt)

        return ODESystem(new_variables, new_derivatives, indep_var=system.indep_var)

    def translate(self, system):
        ''' Given a system of ODEs, translate the system into a simplified version. General version
        '''
        # y = numpy.array(scale_action(system.variables, self.herm_mult_n))
        #print 'y = ', numpy.array(scale_action(system.variables, self.herm_mult_n))
        num_inv_var = self.herm_mult_n.shape[1]
        invariant_variables = sympy.var(' '.join(['y{}'.format(i) for i in xrange(num_inv_var)]))
        if num_inv_var == 1:
            invariant_variables = [invariant_variables]
        else:
            invariant_variables = list(invariant_variables)

        # x = numpy.array(scale_action(system.variables, self.herm_mult_i))
        #print 'x = ', numpy.array(scale_action(system.variables, self.herm_mult_i))
        num_aux_var = self.herm_mult_i.shape[1]
        auxiliary_variables = sympy.var(' '.join(['x{}'.format(i) for i in xrange(num_aux_var)]))
        if num_aux_var == 1:
            auxiliary_variables = [auxiliary_variables]
        else:
            auxiliary_variables = list(auxiliary_variables)

        to_sub = dict(zip(system.variables, scale_action(invariant_variables, self.inv_herm_mult_d)))
        system_derivatives = system.derivatives
        # fywd = F(y^(W_d)) in Hubert Labahn
        fywd = numpy.array([(system.indep_var * f / v).subs(to_sub, simultaneous=True).expand().simplify() for v, f in
                            zip(system.variables, system_derivatives)])
        dydt = invariant_variables * numpy.dot(fywd, self.herm_mult_n) / system.indep_var
        dxdt = auxiliary_variables * numpy.dot(fywd, self.herm_mult_i) / system.indep_var

        new_variables = [system.indep_var] + list(auxiliary_variables) + list(invariant_variables)
        new_derivatives = [sympy.sympify(1)] + list(dxdt) + list(dydt)
        assert len(new_variables) == len(new_derivatives) == len(system.variables) + 1
        return ODESystem(new_variables, new_derivatives, indep_var=system.indep_var)


def scale_action(vect, scaling_matrix):
    ''' Given a vector of sympy expressions, determine the action defined by scaling_matrix
        I.e. Given vect, calculate vect^scaling_matrix (in notation of Hubert Labahn).

        >>> import sympy

        Example 3.3
        >>> v = sympy.var('l')
        >>> A = numpy.array([[2, 3]])
        >>> scale_action([v], A)
        array([l**2, l**3], dtype=object)

        Example 3.4
        >>> v = sympy.var('m v')
        >>> A = numpy.array([[6, 0, -4, 1, 3], [0, 3, 1, -4, 3]])
        >>> scale_action(v, A)
        array([m**6, v**3, v/m**4, m/v**4, m**3*v**3], dtype=object)
    '''
    assert len(vect) == scaling_matrix.shape[0]
    out = []
    for col in scaling_matrix.T:
        mon = 1
        for var, power in zip(vect, col):
            mon *= var ** power
        out.append(mon)
    return numpy.array(out)

# if __name__ == '__main__':
#     import doctest
#     doctest.testmod()