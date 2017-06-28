
import numpy
import sympy

from hermite_helper import normal_hnf_col, INT_TYPE_DEF, hnf_col, is_hnf_col
from ode_system import ODESystem
from tex_tools import matrix_to_tex

def _int_inv(matrix_):
    ''' Given an integer matrix, return the inverse, ensuring we do it right in the integers'''
    float_inv = numpy.linalg.inv(matrix_)
    int_inv = float_inv.astype(INT_TYPE_DEF)
    if numpy.max(numpy.abs(float_inv - int_inv)) > 1e-10:
        raise ValueError('Unable to calculate integer inverse of:\n{}\nDifferent inverses are:\n{}\n{}'.format(matrix_, int_inv, float_inv))
    return int_inv

class ODETranslation(object):
    ''' An object used for translating between systems of ODEs according to a scaling matrix '''
    def __init__(self, scaling_matrix, variables_domain=None, hermite_multiplier=None):
        scaling_matrix = numpy.copy(scaling_matrix)

        self._scaling_matrix = scaling_matrix
        self._scaling_matrix_hnf, self._herm_mult = normal_hnf_col(scaling_matrix)

        if hermite_multiplier is not None:
            if not is_hnf_col(numpy.dot(scaling_matrix, hermite_multiplier)):
                raise ValueError('{}.{}={} is not in HNF'.format(scaling_matrix,
                                 hermite_multiplier, numpy.dot(scaling_matrix, hermite_multiplier)))
            self._herm_mult = hermite_multiplier

        self._inv_herm_mult = _int_inv(self._herm_mult)
        self._variables_domain = variables_domain

    def __repr__(self):
        return 'A=\n{}\nV=\n{}\n\nW={}'.format(self.scaling_matrix, self.herm_mult, self.inv_herm_mult)

    def to_tex(self):
        ''' Tex goodness '''
        to_print = (self.scaling_matrix, self.herm_mult, self.inv_herm_mult)
        to_print = map(matrix_to_tex, to_print)
        return 'A=\n{}\nV=\n{}\n\nW={}'.format(*to_print)

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
    def herm_form(self):
        ''' Return the Hermite normal form '''
        return numpy.copy(self._scaling_matrix_hnf)

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
    def dep_var_herm_mult(self, indep_var_index=0):
        ''' Return the Hermite multiplier, ignoring the independent variable '''
        new_herm_mult = numpy.copy(self.herm_mult)
        col_to_delete, new_herm_mult = new_herm_mult[indep_var_index], numpy.delete(new_herm_mult, indep_var_index, axis=0)
        new_herm_mult = new_herm_mult[:, ~col_to_delete.astype(bool)]
        return new_herm_mult

    @property
    def dep_var_inv_herm_mult(self):
        ''' Return the inverse Hermite multiplier, ignoring the independent variable '''
        return _int_inv(self.dep_var_herm_mult)

    @property
    def variables_domain(self):
        return self._variables_domain

    @classmethod
    def from_ode_system(cls, ode_system):
        ''' Create a translation given an ODESystem '''
        return cls(scaling_matrix=ode_system.maximal_scaling_matrix(), variables_domain=ode_system.variables)

    def translate(self, system):
        ''' Translate, depending on whether the scaling matrix acts on time or not '''
        if ((len(system.variables) == self.scaling_matrix.shape[1] + 1) or
            (numpy.all(self.scaling_matrix[:, system.indep_var_index] == 0))):
            return self.translate_dep_var(system=system)
        elif (len(system.variables) == self.scaling_matrix.shape[1]):
            return self.translate_general(system=system)
        raise ValueError("System doesn't have the right number of variables for translation")


    def translate_dep_var(self, system):
        ''' Given a system of ODEs, translate the system into a simplified version. Assume we are only working on
            dependent variables, not time.
        '''
        # First check that our scaling action doesn't act on the independent variable
        if self.scaling_matrix.shape[1] == len(system.variables):
            assert numpy.all(self.scaling_matrix[:, system.indep_var_index] == 0)
            scaling_matrix = numpy.delete(self.scaling_matrix, system.indep_var_index, axis=1)
            new_herm_mult = self.dep_var_herm_mult
        else:
            assert self.scaling_matrix.shape[1] == len(system.variables) - 1
            scaling_matrix = numpy.copy(self.scaling_matrix)
            new_herm_mult = self.dep_var_herm_mult
            assert numpy.all(new_herm_mult[-1] == 0)
            new_herm_mult = numpy.delete(new_herm_mult, system.indep_var_index, axis=0)
            new_herm_mult = numpy.delete(new_herm_mult, system.indep_var_index, axis=1)

        reduced_scaling = ODETranslation(scaling_matrix=scaling_matrix,
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

        system_var_no_indep = list(system.variables)
        system_var_no_indep.pop(system.indep_var_index)
        to_sub = dict(zip(system_var_no_indep, scale_action(invariant_variables, reduced_scaling.inv_herm_mult_d)))
        system_derivatives = system.derivatives
        # fywd = F(y^(W_d)) in Hubert Labahn
        fywd = numpy.array([(f / v).subs(to_sub, simultaneous=True).expand() for v, f in
                            zip(system.variables, system_derivatives)])
        fywd = numpy.delete(fywd, system.indep_var_index)
        dydt = invariant_variables * numpy.dot(fywd, reduced_scaling.herm_mult_n)
        dxdt = auxiliary_variables * numpy.dot(fywd, reduced_scaling.herm_mult_i)

        new_variables = [system.indep_var] + list(auxiliary_variables) + list(invariant_variables)
        new_derivatives = [sympy.sympify(1)] + list(dxdt) + list(dydt)

        return ODESystem(new_variables, new_derivatives, indep_var=system.indep_var)

    def translate_general(self, system):
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
        fywd = numpy.array([(system.indep_var * f / v).subs(to_sub, simultaneous=True).expand() for v, f in
                            zip(system.variables, system_derivatives)])
        dydt = invariant_variables * numpy.dot(fywd, self.herm_mult_n) / system.indep_var
        dxdt = auxiliary_variables * numpy.dot(fywd, self.herm_mult_i) / system.indep_var

        new_variables = [system.indep_var] + list(auxiliary_variables) + list(invariant_variables)
        new_derivatives = [sympy.sympify(1)] + list(dxdt) + list(dydt)
        assert len(new_variables) == len(new_derivatives) == len(system.variables) + 1
        return ODESystem(new_variables, new_derivatives, indep_var=system.indep_var)

    def translate_general_2(self, system):
        ''' Translation with the cleanest conceptual framework '''
        raise NotImplemented('Not yet finished for the general case')
        num_inv_var = self.herm_mult_n.shape[1]
        invariant_variables = sympy.var(' '.join(['y{}'.format(i) for i in xrange(num_inv_var)]))
        if num_inv_var == 1:
            invariant_variables = [invariant_variables]
        else:
            invariant_variables = list(invariant_variables)

        num_aux_var = self.herm_mult_i.shape[1]
        auxiliary_variables = sympy.var(' '.join(['x{}'.format(i) for i in xrange(num_aux_var)]))
        if num_aux_var == 1:
            auxiliary_variables = [auxiliary_variables]
        else:
            auxiliary_variables = list(auxiliary_variables)

        to_sub = dict(zip(system.variables, scale_action(invariant_variables, self.inv_herm_mult_d)))
        system_derivatives = system.derivatives
        # fywd = F(y^(W_d)) in Hubert Labahn
        fywd = numpy.array([(system.indep_var * f / v).subs(to_sub, simultaneous=True).expand() for v, f in
                            zip(system.variables, system_derivatives)])
        dydt = invariant_variables * numpy.dot(fywd, self.herm_mult_n) / system.indep_var
        dxdt = auxiliary_variables * numpy.dot(fywd, self.herm_mult_i) / system.indep_var

        new_variables = list(auxiliary_variables) + list(invariant_variables)
        new_derivatives = list(dxdt) + list(dydt)
        indep_deriv = new_derivatives[system.indep_var_index]
        new_derivatives = [deriv / indep_deriv for deriv in new_derivatives]
        assert len(new_variables) == len(new_derivatives) == len(system.variables)
        return ODESystem(new_variables, new_derivatives, indep_var=new_variables[system.indep_var_index])

    def _is_translate_parameter_compatible(self, system):
        ''' Check weather a system can be used to use the parameter scheme of translation '''
        # First check our system is in order
        if system.indep_var_index != 0:
            return False
        for i in xrange(system.num_constants):
            if system.derivatives[-i - 1] != sympy.sympify(0):
                return False

        # Now check our transformation is valid
        m = len(system.variables) - system.num_constants
        if not numpy.all(self.herm_mult_i[:m] == 0):
            return False
        if not numpy.all(self.herm_mult_n[:m, :m] == numpy.eye(m)):
            return False
        if not numpy.all(self.herm_mult_n[:m, m:] == 0):
            return False

        if not numpy.all(self.inv_herm_mult[self.r:self.r + m] ==
                                 numpy.hstack((numpy.eye(m), numpy.zeros((m, system.num_constants))))):
            return False

        return True

    def translate_parameter_substitutions(self, system):
        ''' Given a system, determine the substitutions made in translation '''
        num_variables = len(system.variables) - system.num_constants - 1  # Excluding indep
        m = num_variables + 1  # Include indep

        if not self._is_translate_parameter_compatible(system):
            err_str = ['System is not compatible for parameter translation.']
            err_str.append('System may not be ordered properly. Must be independent, dependent, constants. Order is: {}'.format(system.variables))
            err_str.append('Transformation may not be appropriate. Should be 0, I, 0. Transformation is:')
            err_str.append(repr(self.herm_mult_i[:m]))
            err_str.append(repr(self.herm_mult_n[:m, :m]))
            err_str.append(repr(self.herm_mult_n[:m, m:]))
            raise ValueError('\n'.join(err_str))

        # Extract the right bits of W
        inv_herm_mult_d = self.inv_herm_mult_d[m:]
        W_t = inv_herm_mult_d[:, :1]
        W_v = inv_herm_mult_d[:, 1:m]
        W_c = inv_herm_mult_d[:, m:]

        # Form new constants
        new_const = ['c{}'.format(i) for i in xrange(system.num_constants - self.r)]
        new_const = map(sympy.sympify, new_const)
        to_sub = {}

        # Scale t
        to_sub[system.indep_var] = scale_action(new_const, W_t)[0] * system.indep_var

        # Scale dependents
        const_scale = scale_action(new_const, W_v)
        assert len(system.variables[1:num_variables + 1]) == len(const_scale.T)
        for dep_var, _const_scale in zip(system.variables[1:num_variables + 1], const_scale.T):
            to_sub[dep_var] = _const_scale * dep_var

        # Scale constants
        const_scale = scale_action(new_const, W_c)
        assert len(system.variables[- system.num_constants:]) == len(const_scale.T)
        for const, _const_scale in zip(system.variables[- system.num_constants:], const_scale.T):
            to_sub[const] = _const_scale

        return to_sub

    def reverse_translate_parameter_substitutions(self, system):
        ''' Given a system, determine the substitutions made in translation '''
        num_variables = len(system.variables) - system.num_constants - 1  # Excluding indep
        m = num_variables + 1  # Include indep

        if not self._is_translate_parameter_compatible(system):
            err_str = ['System is not compatible for parameter translation.']
            err_str.append('System may not be ordered properly. Must be independent, dependent, constants. Order is: {}'.format(system.variables))
            err_str.append('Transformation may not be appropriate. Should be 0, I, 0. Transformation is:')
            err_str.append(repr(self.herm_mult_i[:m]))
            err_str.append(repr(self.herm_mult_n[:m, :m]))
            err_str.append(repr(self.herm_mult_n[:m, m:]))
            raise ValueError('\n'.join(err_str))

        # Extract the right bits of W
        herm_mult_n = self.herm_mult_n[m:]
        V_t = herm_mult_n[:, :1]
        V_v = herm_mult_n[:, 1:m]
        V_c = herm_mult_n[:, m:]

        # Form new constants
        new_const = ['c{}'.format(i) for i in xrange(system.num_constants - self.r)]
        new_const = map(sympy.sympify, new_const)

        old_const = system.constant_variables
        to_sub = {}

        # Scale t
        to_sub[system.indep_var] = scale_action(old_const, V_t)[0] * system.indep_var

        # Scale dependents
        const_scale = scale_action(old_const, V_v)
        assert len(system.variables[1:num_variables + 1]) == len(const_scale.T)
        for dep_var, _const_scale in zip(system.variables[1:num_variables + 1], const_scale.T):
            to_sub[dep_var] = _const_scale * dep_var

        # Scale constants
        const_scale = scale_action(old_const, V_c)
        assert len(new_const) == len(const_scale.T)
        for _new_const, _const_scale in zip(new_const, const_scale.T):
            to_sub[_new_const] = _const_scale

        return to_sub

    def translate_parameter(self, system):
        ''' Translate according to the parameter scheme '''
        to_sub = self.translate_parameter_substitutions(system=system)

        new_deriv_dict = {}
        for key, val in system.derivative_dict.iteritems():
            if key in system.constant_variables:
                continue
            new_deriv_dict[key] = val.subs(to_sub)

        return ODESystem.from_dict(new_deriv_dict)

    def reverse_translate(self, variables):
        ''' Given an iterable of variables, attempt to reverse translate '''
        if len(variables) == self.scaling_matrix.shape[1]:
            return self.reverse_translate_dep_var(variables=variables)
        elif len(variables) == self.scaling_matrix.shape[1] - 1:
            return self.reverse_translate_dep_var(variables=variables)
        elif len(variables) == self.scaling_matrix.shape[1] + 1:
            return self.reverse_translate_general(variables=variables)
        else:
            raise ValueError('Incorrect number of variables for reverse translation')

    def reverse_translate_dep_var(self, variables):
        ''' Given an iterable of variables, or exprs, reverse translate into the original variables.
        '''
        if len(variables) == self.scaling_matrix.shape[1]:
            return type(variables)(scale_action(variables, self.inv_herm_mult))
        elif len(variables) == self.scaling_matrix.shape[1] - 1:
            return type(variables)(scale_action(variables, self.dep_var_inv_herm_mult))
        else:
            raise ValueError('Incorrect number of variables for reverse translation')


    def reverse_translate_general(self, variables, system_indep_var_index=0):
        ''' Given an iterable of variables, or exprs, reverse translate into the original variables.
            Here we expect t as the first variable, since we need to divide by it and substitute
        '''
        if len(variables) != self.scaling_matrix.shape[1] + 1:
            raise ValueError('Incorrect number of variables for reverse translation')

        indep_var, variables = variables[0], variables[1:]  # Indep var is always first in reverse translation only
        raw_solutions = scale_action(variables, self.inv_herm_mult)
        indep_const = raw_solutions[system_indep_var_index] / indep_var  # Shift up by 1 as our first variable is the independent variable
        raw_solutions = numpy.hstack((raw_solutions[:system_indep_var_index], raw_solutions[system_indep_var_index + 1:]))

        # Check indep_const is independent of t
        indep_const_atoms = indep_const.atoms(sympy.Symbol)
        if indep_var in indep_const_atoms:
            raise ValueError('{} is not independent of the independent variable'.format(indep_const))
        # And if we can, the original variables
        if (self.variables_domain is not None) and len(set(self.variables_domain).intersection(indep_const_atoms)):
            raise ValueError('{} is not independent of {}'.format(indep_const,
                                                                  set(self.variables_domain).intersection(indep_const_atoms)))


        to_sub = {indep_var: indep_var / indep_const}
        solns = [soln.subs(to_sub) for soln in raw_solutions]
        return type(variables)(solns)

    def _validate_variables(self, variables, num_var, stem, use_domain_var):
        ''' Make sure we have the right number of variables if given, else make
            them up
        '''
        if variables is None:
            if use_domain_var and (self.variables_domain is not None):
                variables = self.variables_domain
            else:
                variables = sympy.var(', '.join('{}{}'.format(stem, i) for i in xrange(num_var)))

        if len(variables) != num_var:
            raise ValueError('Expecting {} variables not {}'.format(num_var, variables))
        return variables

    def invariants(self, variables=None):
        ''' Give the invariants of the system'''
        variables = self._validate_variables(variables, self.n, 'y', True)
        return scale_action(variables, self.herm_mult_n)

    def auxiliaries(self, variables=None):
        ''' Return the auxiliary variables '''
        variables = self._validate_variables(variables, self.n, 'x', True)
        return scale_action(variables, self.herm_mult_i)

    def rewrite_rules(self, variables=None):
        ''' Given a set of variables, print the rewrite rules '''
        variables = self._validate_variables(variables, self.n, 'z', True)

        rules = scale_action(variables, numpy.dot(self.herm_mult_n, self.inv_herm_mult_d)).T
        assert len(rules) == len(variables)

        # Create a dict of rules
        rules = dict(zip(variables, rules))

        # Now print
        for var in variables:
            print '{}\t|-->\t{}'.format(var, rules[var])


    def moving_frame(self, variables=None):
        ''' Given a set of variables, print the rewrite rules '''
        variables = self._validate_variables(variables, self.n, 'z', True)

        rules = scale_action(variables, - self.herm_mult_i)
        assert len(rules) == self.r
        print '{}\t->\t{}'.format(tuple(variables), tuple(rules))

    def rational_section(self, variables):
        ''' Give the polynomials defining the moving frame '''
        variables = self._validate_variables(variables, self.n, 'z', True)

        vip = self.herm_mult_i
        vip[vip < 0] = 0

        vim = self.herm_mult_i
        vim[vim > 0] = 0

        rational_section = scale_action(variables, vip) - scale_action(variables, vim)
        print rational_section

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
    import doctest
    doctest.testmod()