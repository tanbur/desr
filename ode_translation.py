
import sympy

from hermite_helper import normal_hnf_col, hnf_col, is_hnf_col, smf
from ode_system import ODESystem
from tex_tools import matrix_to_tex

def _int_inv(matrix_):
    ''' Given an integer matrix, return the inverse, ensuring we do it right in the integers

    >>> matrix_ = sympy.Matrix([[5, 2, -2],
    ...                         [-7, -3, 3],
    ...                         [17, 8, -7]])
    >>> _int_inv(matrix_)
    Matrix([
    [ 3, 2, 0],
    [-2, 1, 1],
    [ 5, 6, 1]])
    '''
    if abs(matrix_.det()) != 1:
        raise ValueError('Nonunimodular matrix with det {} cannot be inverted over the integers.'.format(matrix_.det()))
    sympy_inverse = sympy.Matrix(matrix_).inv()
    return sympy_inverse

class ODETranslation(object):
    '''
    An object used for translating between systems of ODEs according to a scaling matrix.
    The key data are :attr:`~scaling_matrix` and :attr:`~herm_mult`, which together contain all the information needed
    (along with a variable order) to reduce an :class:`ode_system.ODESystem`.

    Args:
        scaling_matrix (sympy.Matrix): Matrix that defines the torus action on the system.
        variables_domain (iter of sympy.Symbol, optional): An ordering of the variables we expect to act upon.
            If this is not given, we will act on a system according to the position of the variables in
            :attr:`ode_system.ODESystem.variables`, as long as there is the correct number of variables.
        hermite_multiplier (sympy.Matrix, optional): User-defined Hermite multiplier, that puts :attr:`~scaling_matrix` into
            column Hermite normal form.
            If not given, the normal Hermite multiplier will be calculated.
    '''
    def __init__(self, scaling_matrix, variables_domain=None, hermite_multiplier=None):
        scaling_matrix = scaling_matrix.copy()

        self._scaling_matrix = scaling_matrix
        self._variables_domain = variables_domain
        if (variables_domain is not None) and (self.n != len(self.variables_domain)):
            raise ValueError('{} variables given but we have {} actions'.format(len(self.variables_domain), self.n))

        self._scaling_matrix_hnf, self._herm_mult = normal_hnf_col(scaling_matrix)

        if hermite_multiplier is not None:
            if not is_hnf_col(scaling_matrix * hermite_multiplier):
                raise ValueError('{}.{}={} is not in HNF'.format(scaling_matrix,
                                 hermite_multiplier, scaling_matrix * hermite_multiplier))
            self._herm_mult = hermite_multiplier

        self._inv_herm_mult = _int_inv(self._herm_mult)

    def __repr__(self):
        return 'A=\n{}\nV=\n{}\nW=\n{}'.format(self.scaling_matrix.__repr__(),
                                               self.herm_mult.__repr__(),
                                               self.inv_herm_mult.__repr__())

    def to_tex(self):
        '''
        Returns:
            str: The scaling matrix :math:`A`, the Hermite multiplier :math:`V` and :math:`W = V^{-1}`, in beautiful LaTeX.

        >>> print ODETranslation(sympy.Matrix(range(12)).reshape(3, 4)).to_tex()
        A=
        0 & 1 & 2 & 3 \\\\
        4 & 5 & 6 & 7 \\\\
        8 & 9 & 10 & 11 \\\\
        V=
        0 & 0 & 1 & 0 \\\\
        0 & 0 & 0 & 1 \\\\
        -1 & 3 & -3 & -2 \\\\
        1 & -2 & 2 & 1 \\\\
        W=
        0 & 1 & 2 & 3 \\\\
        1 & 1 & 1 & 1 \\\\
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\

        '''
        to_print = (self.scaling_matrix, self.herm_mult, self.inv_herm_mult)
        to_print = map(matrix_to_tex, to_print)
        return 'A=\n{}\nV=\n{}\nW=\n{}'.format(*to_print)

    @property
    def scaling_matrix(self):
        """
        Returns:
            sympy.Matrix: The scaling matrix that this translation corresponds to, often denoted :math:`A`.
        """
        return self._scaling_matrix

    @property
    def r(self):
        '''
        Returns:
            int: The dimension of the scaling action: :math:`r`.
                In particular it is the number of rows of :attr:`~scaling_matrix`.
        '''
        return self._scaling_matrix.shape[0]

    @property
    def n(self):
        '''
        Returns:
            int: The number of original variables that the scaling action is acting on: :math:`n`.
                In particular, it is the number of columns of :attr:`~scaling_matrix`.
        '''
        return self._scaling_matrix.shape[1]

    @property
    def herm_mult(self):
        '''
        Returns:
            sympy.Matrix:
                A column Hermite multiplier :math:`V` that puts the :attr:`~scaling_matrix` in column Hermite normal form.
                That is: :math:`AV = H` is in column Hermite normal form.
        '''
        return self._herm_mult.copy()

    @property
    def herm_form(self):
        '''
        Returns:
            sympy.Matrix: The Hermite normal form of the scaling matrix: :math:`H = AV`.
        '''
        return self._scaling_matrix_hnf.copy()

    @property
    def herm_mult_i(self):
        '''
        Returns:
            sympy.Matrix: :math:`V_{\\mathfrak{i}}`: the first :math:`r` columns of :math:`V`.

                The columns represent the auxiliary variables of the reduction.
        '''
        return self.herm_mult[:, :self.r]

    @property
    def herm_mult_n(self):
        '''
        Returns:
            sympy.Matrix:
                :math:`V_{\\mathfrak{n}}`: the last :math:`n-r` columns of the Hermite multiplier :math:`V``.
                The columns represent the invariants of the scaling action.
        '''
        return self.herm_mult[:, self.r:]


    @property
    def inv_herm_mult(self):
        '''
        Returns:
            sympy.Matrix:
                The inverse of the Hermite multiplier :math:`W=V^{-1}`.
        '''
        return self._inv_herm_mult.copy()

    @property
    def inv_herm_mult_u(self):
        '''
        Returns:
            sympy.Matrix:
                :math:`W_{\\mathfrak{u}}`: the first :math:`r` rows of :math:`W`.
        '''
        return self.inv_herm_mult[:self.r, :]

    @property
    def inv_herm_mult_d(self):
        """
        Returns:
            sympy.Matrix:
                :math:`W_{\\mathfrak{d}}`: the last :math:`n-r` rows of :math:`W`.
        """
        return self.inv_herm_mult[self.r:, :]


    def dep_var_herm_mult(self, indep_var_index=0):
        '''
        Args:
            indep_var_index (int): The index of the independent variable.

        Returns:
            sympy.Matrix:
                The Hermite multiplier :math:`V`, ignoring the independent variable.


        >>> translation = ODETranslation(sympy.Matrix(range(12)).reshape(3, 4))
        >>> translation.herm_mult
        Matrix([
        [ 0,  0,  1,  0],
        [ 0,  0,  0,  1],
        [-1,  3, -3, -2],
        [ 1, -2,  2,  1]])
        >>> translation.dep_var_herm_mult(1)
        Matrix([
        [ 0,  0,  1],
        [-1,  3, -3],
        [ 1, -2,  2]])
        '''
        new_herm_mult = self.herm_mult.copy()
        col_to_delete = new_herm_mult[indep_var_index, :]
        new_herm_mult.row_del(indep_var_index)
        cols_to_keep = [ind for ind, val in enumerate(col_to_delete) if val == 0]
        new_herm_mult = new_herm_mult.extract(range(new_herm_mult.rows), cols_to_keep)
        # new_herm_mult = new_herm_mult[:, ~col_to_delete.astype(bool)]
        return new_herm_mult

    def dep_var_inv_herm_mult(self, indep_var_index=0):
        '''
        Args:
            indep_var_index (int): The index of the independent variable.

        Returns:
            sympy.Matrix:
                The inverse Hermite multiplier :math:`W`, ignoring the independent variable.

        >>> translation = ODETranslation(sympy.Matrix(range(12)).reshape(3, 4))
        >>> translation.inv_herm_mult
        Matrix([
        [0, 1, 2, 3],
        [1, 1, 1, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0]])
        >>> translation.dep_var_inv_herm_mult(1)
        Matrix([
        [0, 2, 3],
        [1, 1, 1],
        [1, 0, 0]])
        '''
        return _int_inv(self.dep_var_herm_mult(indep_var_index=indep_var_index))

    @property
    def variables_domain(self):
        '''
        Returns:
            tuple, None:
                The variables that the scaling action acts on.
        '''
        return self._variables_domain

    @classmethod
    def from_ode_system(cls, ode_system):
        '''
        Create a :class:`ODETranslation` given an :class:`ode_system.ODESystem` instance, by taking the maximal scaling matrix.

        Args:
            ode_system (ODESystem)

        :rtype: ODETranslation
        '''
        return cls(scaling_matrix=ode_system.maximal_scaling_matrix(), variables_domain=ode_system.variables)

    def translate(self, system):
        '''
        Translate an :class:`ode_system.ODESystem` into a reduced system.

        First, try the simplest parameter reduction method, then the dependent variable translation (where the scaling action ignores the independent variable) and finally the general reduction scheme.

        Args:
            system (ODESystem): System to reduce.

        :rtype: ODESystem
        '''
        if self._is_translate_parameter_compatible(system=system):
            return self.translate_parameter(system=system)
        elif ((len(system.variables) == self.scaling_matrix.shape[1] + 1) or
            (self.scaling_matrix[:, system.indep_var_index].is_zero)):
            return self.translate_dep_var(system=system)
        elif (len(system.variables) == self.scaling_matrix.shape[1]):
            return self.translate_general(system=system)
        raise ValueError("System doesn't have the right number of variables for translation")


    def translate_dep_var(self, system):
        '''
        Given a system of ODEs, translate the system into a simplified version. Assume we are only working on
        dependent variables, not the independent variable.

        Args:
            system (ODESystem): System to reduce.

        :rtype: ODESystem

        >>> equations = 'dz1/dt = z1*(1+z1*z2);dz2/dt = z2*(1/t - z1*z2)'.split(';')
        >>> system = ODESystem.from_equations(equations)
        >>> translation = ODETranslation.from_ode_system(system)
        >>> translation.translate_dep_var(system=system)
        dt/dt = 1
        dx0/dt = x0*(y0 - 1/t)
        dy0/dt = y0*(1 + 1/t)
        '''
        # First check that our scaling action doesn't act on the independent variable
        if self.n == len(system.variables):
            if not self.scaling_matrix[:, system.indep_var_index].is_zero:
                raise ValueError('The independent variable of\n{}\nis acted on by\n{}.'.format(system, self))
            scaling_matrix = self.scaling_matrix.copy()
            scaling_matrix.col_del(system.indep_var_index)
            new_herm_mult = self.dep_var_herm_mult(indep_var_index=system.indep_var_index)
        else:
            assert self.scaling_matrix.shape[1] == len(system.variables) - 1
            scaling_matrix = self.scaling_matrix.copy()
            new_herm_mult = self.dep_var_herm_mult(indep_var_index=system.indep_var_index)
            assert new_herm_mult[-1, :].is_zero
            new_herm_mult.col_del(system.indep_var_index)
            new_herm_mult.row_del(system.indep_var_index)

        if self.variables_domain is not None:
            variables_domain = list(self.variables_domain)
            variables_domain.pop(system.indep_var_index)
        else:
            variables_domain = None

        reduced_scaling = ODETranslation(scaling_matrix=scaling_matrix,
                                         variables_domain=variables_domain,
                                         hermite_multiplier=new_herm_mult)

        # y = sympy.Matrix(scale_action(system.variables, self.herm_mult_n))
        #print 'y = ', sympy.Matrix(scale_action(system.variables, self.herm_mult_n))
        num_inv_var = reduced_scaling.herm_mult_n.shape[1]
        invariant_variables = sympy.var(' '.join(['y{}'.format(i) for i in xrange(num_inv_var)]))
        if num_inv_var == 1:
            invariant_variables = [invariant_variables]
        else:
            invariant_variables = list(invariant_variables)

        # x = sympy.Matrix(scale_action(system.variables, self.herm_mult_i))
        #print 'x = ', sympy.Matrix(scale_action(system.variables, self.herm_mult_i))
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
        fywd = sympy.Matrix([(f / v).subs(to_sub, simultaneous=True).expand() for v, f in
                            zip(system.variables, system_derivatives)]).T
        fywd.col_del(system.indep_var_index)
        dydt = sympy.Matrix(invariant_variables).T.multiply_elementwise(fywd * reduced_scaling.herm_mult_n)
        dxdt = sympy.Matrix(auxiliary_variables).T.multiply_elementwise(fywd * reduced_scaling.herm_mult_i)

        new_variables = [system.indep_var] + list(auxiliary_variables) + list(invariant_variables)
        new_derivatives = [sympy.sympify(1)] + list(dxdt) + list(dydt)

        return ODESystem(new_variables, new_derivatives, indep_var=system.indep_var)

    def translate_general(self, system):
        '''
        The most general reduction scheme.
        If there are :math:`n` variables (including the independent variable) then there will be a system of
        :math:`n-r+1` invariants and :math:`r` auxiliary variables.

        Args:
            system (ODESystem): System to reduce.

        :rtype: ODESystem

        >>> equations = 'dn/dt = n*( r*(1 - n/K) - k*p/(n+d) );dp/dt = s*p*(1 - h*p / n)'.split(';')
        >>> system = ODESystem.from_equations(equations)
        >>> translation = ODETranslation.from_ode_system(system)
        >>> translation.translate_general(system=system)
        dt/dt = 1
        dx0/dt = 0
        dx1/dt = 0
        dx2/dt = 0
        dy0/dt = y0/t
        dy1/dt = y1*(-y0*y1*y5/y3 - y0*y2/(y1 + 1) + y0*y5)/t
        dy2/dt = y2*(y0 - y0*y2*y4/y1)/t
        dy3/dt = 0
        dy4/dt = 0
        dy5/dt = 0
        '''
        # y = sympy.Matrix(scale_action(system.variables, self.herm_mult_n))
        #print 'y = ', sympy.Matrix(scale_action(system.variables, self.herm_mult_n))
        num_inv_var = self.herm_mult_n.shape[1]
        invariant_variables = sympy.var(' '.join(['y{}'.format(i) for i in xrange(num_inv_var)]))
        if num_inv_var == 1:
            invariant_variables = [invariant_variables]
        else:
            invariant_variables = list(invariant_variables)

        # x = sympy.Matrix(scale_action(system.variables, self.herm_mult_i))
        #print 'x = ', sympy.Matrix(scale_action(system.variables, self.herm_mult_i))
        num_aux_var = self.herm_mult_i.shape[1]
        auxiliary_variables = sympy.var(' '.join(['x{}'.format(i) for i in xrange(num_aux_var)]))
        if num_aux_var == 1:
            auxiliary_variables = [auxiliary_variables]
        else:
            auxiliary_variables = list(auxiliary_variables)

        to_sub = dict(zip(system.variables, scale_action(invariant_variables, self.inv_herm_mult_d)))
        system_derivatives = system.derivatives
        # fywd = F(y^(W_d)) in Hubert Labahn
        fywd = sympy.Matrix([(system.indep_var * f / v).subs(to_sub, simultaneous=True).expand() for v, f in
                            zip(system.variables, system_derivatives)]).T
        dydt = sympy.Matrix(invariant_variables).T.multiply_elementwise(fywd * self.herm_mult_n) / system.indep_var
        dxdt = sympy.Matrix(auxiliary_variables).T.multiply_elementwise(fywd * self.herm_mult_i) / system.indep_var

        new_variables = [system.indep_var] + list(auxiliary_variables) + list(invariant_variables)
        new_derivatives = [sympy.sympify(1)] + list(dxdt) + list(dydt)
        assert len(new_variables) == len(new_derivatives) == len(system.variables) + 1
        return ODESystem(new_variables, new_derivatives, indep_var=system.indep_var)

    def _is_translate_parameter_compatible(self, system):
        ''' Check whether a system satisfies the conditions of the parameter scheme of translation '''
        # First check our system's variables are in the required order
        # Independent at the beginning
        if system.indep_var_index != 0:
            return False
        # Constant variables at the end
        for i in xrange(system.num_constants):
            if system.derivatives[-i - 1] != sympy.sympify(0):
                return False

        # Now check our transformation is valid: that V and W have the required forms
        # m is number of non-constant variables (including independent variable)
        m = len(system.variables) - system.num_constants
        if not self.herm_mult_i[:m, :].is_zero:
            return False
        if not self.herm_mult_n[:m, :m] == sympy.eye(m):
            return False
        if not self.herm_mult_n[:m, m:].is_zero:
            return False

        if not (self.inv_herm_mult[self.r:self.r + m, :] ==
                sympy.Matrix.hstack(sympy.eye(m), sympy.zeros(m, system.num_constants))):
            return False

        return True

    def translate_parameter_substitutions(self, system):
        '''
        Given a system, determine the substitutions made in the parameter reduction.

        Args:
            system (ODESystem):
                The system in question.

        :rtype: dict

        >>> equations = 'dn/dt = n*( r*(1 - n/K) - k*p/(n+d) );dp/dt = s*p*(1 - h*p / n)'.split(';')
        >>> system = ODESystem.from_equations(equations)
        >>> translation = ODETranslation.from_ode_system(system)
        >>> translation.translate_parameter_substitutions(system=system)
        {k: 1, n: n, r: c2, d: 1, K: c0, h: c1, s: 1, p: p, t: t}
        '''
        num_variables = len(system.variables) - system.num_constants - 1  # Excluding indep
        m = num_variables + 1  # Include indep

        if not self._is_translate_parameter_compatible(system):
            err_str = ['System is not compatible for parameter translation.']
            err_str.append('System may not be ordered properly. Must be independent, dependent, constants. Order is: {}'.format(system.variables))
            err_str.append('Transformation may not be appropriate. Should be 0, I, 0. Transformation is:')
            err_str.append(repr(self.herm_mult_i[:m, :]))
            err_str.append(repr(self.herm_mult_n[:m, :m]))
            err_str.append(repr(self.herm_mult_n[:m, m:]))
            raise ValueError('\n'.join(err_str))

        # Extract the right bits of W
        inv_herm_mult_d = self.inv_herm_mult_d[m:, :]
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
        '''
        Args:
            system (ODESystem): The *reduced* system.

        Returns:
            dict:
                The substitutions needed to reverse translate.

        >>> equations = 'dn/dt = n*( r*(1 - n/K) - k*p/(n+d) );dp/dt = s*p*(1 - h*p / n)'.split(';')
        >>> system = ODESystem.from_equations(equations)
        >>> translation = ODETranslation.from_ode_system(system)
        >>> reduced = translation.translate_parameter(system)
        >>> translation.reverse_translate_parameter_substitutions(system=reduced)
        {k: 1, n: n, r: c2, d: 1, K: c0, h: c1, s: 1, p: p, t: t}
        '''
        num_variables = len(system.variables) - system.num_constants - 1  # Excluding indep
        m = num_variables + 1  # Include indep

        if not self._is_translate_parameter_compatible(system):
            err_str = ['System is not compatible for parameter translation.']
            err_str.append('System may not be ordered properly. Must be independent, dependent, constants. Order is: {}'.format(system.variables))
            err_str.append('Transformation may not be appropriate. Should be 0, I, 0. Transformation is:')
            err_str.append(repr(self.herm_mult_i[:m, :]))
            err_str.append(repr(self.herm_mult_n[:m, :m]))
            err_str.append(repr(self.herm_mult_n[:m, m:]))
            raise ValueError('\n'.join(err_str))

        # Extract the right bits of V
        herm_mult_n = self.herm_mult_n[m:, :]
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
        ''' Translate according to parameter scheme '''
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

    def reverse_translate_dep_var(self, variables, indep_var_index):
        ''' Given an iterable of variables, or exprs, reverse translate into the original variables.
        '''
        if len(variables) == self.scaling_matrix.shape[1]:
            return type(variables)(scale_action(variables, self.inv_herm_mult(indep_var_index=indep_var_index)))
        elif len(variables) == self.scaling_matrix.shape[1] - 1:
            return type(variables)(scale_action(variables, self.dep_var_inv_herm_mult(indep_var_index=indep_var_index)))
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
        raw_solutions.col_del(system_indep_var_index)

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

        rules = scale_action(variables, self.herm_mult_n * self.inv_herm_mult_d).T
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

    ## Choosing invariants
    def extend_from_invariants(self, invariant_choice):
        '''
        Extend a given set of invariants, expressed as a matrix of exponents, to find a Hermite multiplier that will
        rewrite the system in terms of invariant_choice.


        Parameters
        ----------
        invariant_choice : sympy.Matrix
            The :math:`n \\times k` matrix representing the invariants, where :math:`n` is the number of variables and :math:`k` is the
            number of given invariants.


        Returns
        -------
        ODETranslation
            An ODETranslation representing the rewrite rules in terms of the given invariants.


        >>> variables = sympy.symbols(' '.join(['y{}'.format(i) for i in xrange(6)]))
        >>> ode_translation = ODETranslation(sympy.Matrix([[1, 0, 3, 0, 2, 2],
        ...                                                [0, 2, 0, 1, 0, 1],
        ...                                                [2, 0, 0, 3, 0, 0]]))
        >>> ode_translation.invariants(variables=variables)
        Matrix([[y0**3*y2*y5**2/(y3**2*y4**5), y1*y4**2/y5**2, y2**2/y4**3]])

        Now we can express two new invariants, which we think are more interesting, as a matrix.
        We pick the product of the first two invariants, and the product of the last two invariants:
        :code:`y0**3 * y1 * y2/(y3**2 * y4**3)` and :code:`y1 *  y2**2/(y4 * y5**2)`

        >>> new_inv = sympy.Matrix([[3, 1, 1, -2, -3, 0],
        ...                         [0, 1, 2, 0, -1, -2]]).T

        >>> new_translation = ode_translation.extend_from_invariants(new_inv)
        >>> new_translation.invariants(variables=variables)
        Matrix([[y0**3*y1*y2/(y3**2*y4**3), y1*y2**2/(y4*y5**2), y1*y4**2/y5**2]])
        '''
        ## Step 1: Check we have invariants
        choice_actions = self.scaling_matrix * invariant_choice
        if not choice_actions.is_zero:
            raise ValueError('The desired combinations {} are not invariants of the scaling action.'.format(invariant_choice))

        ## Step 2: Try and extend the choices by a basis of invariants
        ## Step 2a: Extend (W_d . invariant_choice) to a unimodular matrix
        column_operations = extend_rectangular_matrix(self.inv_herm_mult_d * invariant_choice)

        # Step 2b: Apply these column operations to the current Hermite multiplier
        hermite_multiplier = self.herm_mult.copy()
        hermite_multiplier[:, self.r:] = self.herm_mult_n * column_operations
        max_scal = ODETranslation(scaling_matrix=self.scaling_matrix, variables_domain=self.variables_domain,
                                  hermite_multiplier=hermite_multiplier)

        return max_scal

    ## Determine whether an expression is an invariant or not
    def is_invariant_expr(self, expr, variable_order=None):
        '''
        Determine whether an expression is an invariant or not

        Parameters
        ----------
        expr : sympy.Expr
            A symbolic expression which may or may not be an invariant under the scaling action.
        variable_order : list, tuple
            An ordered list that determines how the matrix acts on the variables.

        Returns
        -------
        bool
            True if the expression is an invariant.
        '''
        if variable_order is not None:
            variable_order = tuple(variable_order)
            if len(variable_order) != self.n:
                raise ValueError('{} variables given but we have {} actions'.format(len(variable_order), self.n))
        else:
            if self.variables_domain is None:
                raise ValueError('No variable order found to determine invariance.')
            variable_order = self.variables_domain

        missing_var = set(variable_order).difference(expr.atoms(sympy.Symbol))
        if len(missing_var):
            raise ValueError('Unknown action on variables: {}'.format())

        raise NotImplemented()



def scale_action(vect, scaling_matrix):
    '''
    Given a vector of sympy expressions, determine the action defined by scaling_matrix
    I.e. Given vect, calculate vect^scaling_matrix (in notation of Hubert Labahn).

    Example 3.3

    >>> x = sympy.var('x')
    >>> A = sympy.Matrix([[2, 3]])
    >>> scale_action([x], A)
    Matrix([[x**2, x**3]])

    Example 3.4

    >>> v = sympy.var('m v')
    >>> A = sympy.Matrix([[6, 0, -4, 1, 3], [0, 3, 1, -4, 3]])
    >>> scale_action(v, A)
    Matrix([[m**6, v**3, v/m**4, m/v**4, m**3*v**3]])
    '''
    assert len(vect) == scaling_matrix.shape[0]
    out = []
    for col in scaling_matrix.T.tolist():
        mon = 1
        for var, power in zip(vect, col):
            mon *= var ** power
        out.append(mon)
    return sympy.Matrix(out).T

def extend_rectangular_matrix(matrix_, check_unimodular=True):
    """
    Given a rectangular :math:`n \\times m` integer matrix, extend it to a unimodular one by appending columns.


    Parameters
    ----------
    matrix_
        The rectangular matrix to be extended.


    Returns
    -------
    sympy.Matrix
        Square matrix of determinant 1.


    >>> matrix_ = sympy.Matrix([[1, 0],
    ...                         [0, 1],
    ...                         [0, 0],])
    >>> extend_rectangular_matrix(matrix_)
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])

    By default, we will throw if we cannot extend a matrix to a unimodular matrix.

    >>> matrix_[0, 0] = 2
    >>> extend_rectangular_matrix(matrix_=matrix_)
    Traceback (most recent call last):
        ...
    ValueError: Unable to extend the matrix
    Matrix([[2, 0], [0, 1], [0, 0]])
    to a unimodular matrix.

    We can extend some more interesting matrices.

    >>> matrix_ = sympy.Matrix([[3, 2],
    ...                         [-2, 1],
    ...                         [5, 6],])
    >>> extend_rectangular_matrix(matrix_)
    Matrix([
    [ 3, 2, 0],
    [-2, 1, 1],
    [ 5, 6, 1]])
    """
    matrix_ = matrix_.copy()
    if len(matrix_.shape) != 2:
        raise ValueError('Can only extend arrays of dimension 2, not {}'.format(len(matrix_.shape)))

    n, m = matrix_.shape

    # Assume we have more rows than columns
    if n < m:
        return extend_rectangular_matrix(matrix_=matrix_.T, check_unimodular=check_unimodular).T
    elif n == m:
        return matrix_

    # First find a Smith normal form decomposition of matrix_
    smith_normal_form, row_actions, col_actions = smf(matrix_=matrix_)

    if check_unimodular:
        # We require our extension to have determinant 1, which is only possible if the final entry on the leading diagonal
        # is 1.
        if smith_normal_form[min(n, m) - 1, min(n, m) - 1] != 1:
            raise ValueError('Unable to extend the matrix\n{}\nto a unimodular matrix.'.format(matrix_))

    # To extend to a unimodular matrix, we can extend matrix_ using the last n-m columns of the row_actions matrix
    # Since we can extend modulo column operations, put these last few columns in column Hermite normal form
    extension = hnf_col(_int_inv(row_actions)[:, m:])[0]

    extended = sympy.Matrix.hstack(matrix_, extension)

    assert extended.shape[0] == extended.shape[1] == n
    if check_unimodular:
        if abs(extended.det()) != 1:
            raise RuntimeError('Extended matrix\n{}\nhas determinant {}, not +-1'.format(extended, extended.det()))

    return extended

if __name__ == '__main__':
    import doctest
    doctest.testmod()