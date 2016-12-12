
import numpy
import sympy

from hermite_helper import normal_hnf_col, INT_TYPE_DEF, hnf_col
from ode_system import ODESystem

class ODETranslation(object):
    ''' An object used for translating between systems of ODEs according to a scaling matrix '''
    def __init__(self, scaling_matrix, variables_domain=None):
        scaling_matrix = numpy.copy(scaling_matrix)
        self._scaling_matrix = scaling_matrix
        self._scaling_matrix_hnf, self._herm_mult = normal_hnf_col(scaling_matrix)

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

    def translate_dep_var(self, system):
        ''' Given a system of ODEs, translate the system into a simplified version. Assume we are only working on
            dependent variables, not time.
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

        #TODO remove the variable for t properly rather than missing out an xi or yj
        indep_var_index = system.variables.index(system.indep_var)
        if indep_var_index < num_aux_var:
            auxiliary_variables[indep_var_index] = system.indep_var
        else:
            invariant_variables[indep_var_index - num_aux_var] = system.indep_var


        to_sub = dict(zip(system.variables, scale_action(invariant_variables, self.inv_herm_mult_d)))
        #print to_sub
        system_derivatives = system.derivatives
        ##TODO work out why on earth we want this (f / v). Something to do with eqn 13
        # fywd = F(y^(W_d)) in Hubert Labahn
        fywd = numpy.array([(f / v).subs(to_sub, simultaneous=True).expand().simplify() for v, f in
                            zip(system.variables, system_derivatives)])
        # Reset our time variable
        fywd[system.variables.index(system.indep_var)] = sympy.sympify(1)
        dydt = invariant_variables * numpy.dot(fywd, self.herm_mult_n)
        dxdt = auxiliary_variables * numpy.dot(fywd, self.herm_mult_i)

        new_variables = list(invariant_variables) + list(auxiliary_variables)
        new_derivatives = list(dydt) + list(dxdt)

        # Divide through by t again, accounting for swapping over the x and y blocks. This is only needed for dependent case and if we don't divide through by t elsewhere
        new_derivatives[num_inv_var + indep_var_index if indep_var_index < num_aux_var
                            else indep_var_index - num_aux_var] /= system.indep_var

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

if __name__ == '__main__':

    ### Example 6.4
    import sympy
    from ode_system import ODESystem

    equations = '''dz1/dt = z1*(1+z1*z2);dz2/dt = z2*(1/t - z1*z2)'''.split(';')
    #equations = '''dx/dt = x*y'''.split(';')

    system = ODESystem.from_equations(equations)

    # Match the maximal scaling matrix
    max_scal = system.maximal_scaling_matrix()
    print 'A = ', max_scal

    translation = ODETranslation(max_scal)
    translated = translation.translate_dep_var(system)
    t = translated.derivatives[0]

    #t = t.expand()
    print translated

    self = translation
    # invariant_variables = sympy.var(' '.join(['y{}'.format(i) for i in xrange(self.herm_mult_n.shape[1])]))#numpy.array(scale_action(system.variables, self.herm_mult_n))  # y
    # auxiliary_variables = sympy.var(' '.join(['x{}'.format(i) for i in xrange(self.herm_mult_i.shape[1])]))#numpy.array(scale_action(system.variables, self.herm_mult_i))  # x
    #
    # #new_variables = list(invariant_variables) + list(auxiliary_variables)
    #
    # # print 'x = ', invariant_variables
    # # print 'y = ', auxiliary_variables
    # # print self.inv_herm_mult_d
    print 'V = ', self.herm_mult
    # print 'V.T = ', self.herm_mult
    # print 'W = ', self.inv_herm_mult
    # # print scale_action(auxiliary_variables, self.inv_herm_mult_d)
    # import sympy
    #
    # to_sub = dict(zip(system.variables, scale_action(invariant_variables, self.inv_herm_mult_d)))
    # print to_sub
    # dydt = invariant_variables * numpy.dot(numpy.array([f.subs(to_sub).expand().simplify() for f in system.derivatives]), self.herm_mult_n)
    # dxdt = auxiliary_variables * numpy.dot(numpy.array([f.subs(to_sub).expand().simplify() for f in system.derivatives]), self.herm_mult_i)
    # print 'x = ', numpy.array(scale_action(system.variables, self.herm_mult_i))  # x #auxiliary_variables
    # print dxdt
    # print 'y = ', numpy.array(scale_action(system.variables, self.herm_mult_n))  # y #invariant_variables
    # print [e.expand().simplify() for e in dydt]
    # dydt = numpy.dot(numpy.expand_dims(dydt, 0), self.herm_mult_n)
    # print dydt
    #
    # print invariant_variables
    # print auxiliary_variables


# if __name__ == '__main__':
#     import doctest
#     doctest.testmod()