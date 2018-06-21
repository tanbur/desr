"""Michael-Mentis Equations

This example script walks through a possible analysis of the Michael Mentis equations.


"""
import sympy
from desr.matrix_normal_forms import smf
from desr.ode_system import ODESystem
from desr.ode_system import maximal_scaling_matrix, rational_expr_to_power_matrix, hnf_col, hnf_row, normal_hnf_col
from desr.matrix_normal_forms import normal_hnf_row
from desr.ode_translation import ODETranslation, scale_action
from desr.tex_tools import expr_to_tex, matrix_to_tex

def example_michael_mentis():
    """ Example """
    # Enable the best printing
    sympy.init_printing(pretty_print=True, use_latex=True)


    system_tex = '''\frac{dE}{dt} &= - k_1 E S + k_{-1} C + k_2 C \\\\
             \frac{dS}{dt} &= - k_1 E S + k_{-1} C \\\\
             \frac{dC}{dt} &= k_1 E S - k_{-1} C - k_2 C \\\\
             \frac{dP}{dt} &= k_2 C'''

    original_system = ODESystem.from_tex(system_tex)

    max_scal1 = ODETranslation.from_ode_system(original_system)
    # Print variable order
    print 'Variable order: ', max_scal1.variables_domain

    # Print scaling matrices
    print 'Scaling matrix:'
    print max_scal1.scaling_matrix.__repr__()

    # Print invariants
    print 'Invariants: ', max_scal1.invariants()

    # Print translated system
    print 'Reduced system:'
    print max_scal1.translate(original_system)
    print

    print 'Changing the order of the variables'
    print '-----------------------------------'
    # Show what happens when we put k_2 at the end instead
    original_system_reorder = original_system.copy()
    variable_order = list(original_system.variables)
    variable_order[-1], variable_order[-2] = variable_order[-2], variable_order[-1]
    print 'Variable order:'
    print variable_order
    original_system_reorder.reorder_variables(variable_order)
    max_scal1_reorder = ODETranslation.from_ode_system(original_system_reorder)
    print 'Invariants:'
    print ', '.join(map(str, max_scal1_reorder.invariants()))
    # Now do the reduction
    reduced_system = max_scal1_reorder.translate(original_system_reorder)
    print 'Reduced system:'
    print reduced_system
    print

    print 'Extending a choice of invariants'
    print '--------------------------------'
    # Extend a choice of invariants   t  C  E  P  S k_1 k_2 k_{-1}
    invariant_choice = sympy.Matrix([[0, 1, 0, 0, 0, 1, -1, 0],
                                     [0, 0, 0, 1, 0, 1, 0, -1]]).T
    print 'P ='
    print invariant_choice.__repr__()
    print 'Chosen invariants:'
    print scale_action(max_scal1.variables_domain, invariant_choice)

    ## Method that does the extension automatically.
    max_scal2 = max_scal1.extend_from_invariants(invariant_choice=invariant_choice)

    ## Stepping through the above function for the sake of the paper.
    ## Step 1: Check we have invariants
    choice_actions = max_scal1.scaling_matrix * invariant_choice
    assert choice_actions.is_zero  # Else we have to stop.

    ## Step 2: Try and extend the choices by a basis of invariants
    ## Step 2a: Extend (W_d . invariant_choice) to a unimodular matrix
    WdP = max_scal1.inv_herm_mult_d * invariant_choice
    smith_normal_form, row_ops, col_ops = smf(WdP)

    print 'Smith normal form decomposition:'
    print '{}\n{}\n{}\n{}\n=\n{}'.format(*map(lambda x: x.__repr__(), (row_ops,
                                                                       max_scal1.inv_herm_mult_d,
                                                                       invariant_choice,
                                                                       col_ops,
                                                                       smith_normal_form)))
    print 'U^{-1}:'
    print row_ops.inv()
    print 'Wd P:'
    print WdP

    C = sympy.Matrix.hstack(WdP, row_ops.inv()[:, 2:])  # C is the column operations we're going to apply to Vn
    print 'C:'
    print C

    print 'New Vn:'
    print max_scal1.herm_mult_n * C

    print 'New invariants:'
    print ', '.join(map(str, scale_action(max_scal1.variables_domain, max_scal1.herm_mult_n * C)))
    max_scal3 = max_scal1.herm_mult_n * C
    # The permutation is (0 1 3 4 2 5)
    max_scal3.col_swap(0, 1)
    max_scal3.col_swap(0, 3)
    max_scal3.col_swap(0, 4)
    max_scal3.col_swap(0, 2)
    max_scal3.col_swap(0, 5)
    print 'Permuted Vn:'
    print max_scal3

    print 'Reduced system:'
    # We need to add on Vi - the original will do.
    max_scal3 = sympy.Matrix.hstack(max_scal1.herm_mult_i, max_scal3)
    max_scal3 = ODETranslation(max_scal1.scaling_matrix, hermite_multiplier=max_scal3)
    print max_scal3.translate(original_system)

def example_michael_mentis_simplified(verbose=True):
    """ Example """
    # Enable the best printing
    sympy.init_printing(pretty_print=True, use_latex=True)


    system_tex = '''\frac{ds}{dt} &= - k_1 e_0 s + k_1 c s + k_{-1} c \\\\
             \frac{dc}{dt} &= k_1 e_0 s - k_1 c s - k_{-1} c - k_2 c'''

    system_tex_reduced_km1 = system_tex.replace('k_{-1}', '(K - k_2)')

    reduced_system_km1 = ODESystem.from_tex(system_tex_reduced_km1)

    reduced_system_km1.reorder_variables(['t', 's', 'c', 'K', 'k_2', 'k_1', 'e_0'])

    # Print variable order
    print 'Variable order: ', reduced_system_km1.variables

    print 'Power Matrix:', reduced_system_km1.power_matrix().__repr__()

    # Print scaling matrices
    max_scal1 = ODETranslation.from_ode_system(reduced_system_km1)
    print 'Scaling matrix:'
    print max_scal1.scaling_matrix.__repr__()

    # Print invariants
    print 'Invariants: ', max_scal1.invariants()
    # print ',\quad '.join(map(expr_to_tex, max_scal1.invariants()))

    # Print translated system
    print 'Reduced system:'
    # print max_scal1.translate(original_system)#.to_tex()

    print 'Adding in the initial condition for s'
    print '-------------------------------------'
    reduced_system_km1.update_initial_conditions({'s': 's_0'})
    max_scal2 = ODETranslation.from_ode_system(reduced_system_km1)

    print 'Invariants: ', max_scal2.invariants()

    # print max_scal2.herm_mult.__repr__()
    print 'Michaelis-Menten Reparametrisation 1'
    print 'Changing invariants by column operations on the Hermite multiplier'
    max_scal2.multiplier_add_columns(2, -1, 1)
    max_scal2.multiplier_add_columns(4, -1, -1)

    print 'Invariants: ', max_scal2.invariants()
    # print max_scal2.herm_mult.__repr__()

    # Print translated system
    print 'Reduced system:'
    print max_scal2.translate(reduced_system_km1)#.to_tex()

    print 'Michaelis-Menten Reparametrisation 2'
    print 'Changing invariants by column operations on the Hermite multiplier'
    # Divide time through by epsilon
    max_scal2.multiplier_add_columns(2, -1, -1)


    print 'Invariants: ', max_scal2.invariants()
    # print max_scal2.herm_mult.__repr__()

    # Print translated system
    print 'Reduced system:'
    print max_scal2.translate(reduced_system_km1)#.to_tex()






    print 'Michaelis-Menten Reparametrisation 2'
    print 'What if epsilon = e_0 / s_0 is not small?'
    print 'In order '
    # Substitute K_m into the equations
    system_tex_reduced_km = system_tex.replace('k_{-1}', 'K').replace('K', 'K_m k_1')
    # Now set L = K_m + s_0
    system_tex_reduced_l = system_tex_reduced_km.replace('K_m', '(L - s_0)')
    # print system_tex_reduced_km
    reduced_system_l = ODESystem.from_tex(system_tex_reduced_l)
    reduced_system_l.update_initial_conditions({'s': 's_0'})
    reduced_system_l.reorder_variables(['t', 's', 'c', 'L', 'k_2', 'k_1', 'e_0', 's_0'])

    max_scal3 = ODETranslation.from_ode_system(reduced_system_l)

    print max_scal3.invariants()

    # Scale t correctly to t/t_C = k_1 L t
    max_scal3.multiplier_add_columns(2, 5, 1)
    # Scale c correctly to c / (e_0 s_0 / L)
    max_scal3.multiplier_add_columns(4, 5, 1)
    max_scal3.multiplier_add_columns(4, -1, -1)
    # Find the epsilon constant
    max_scal3.multiplier_negate_column(5)
    max_scal3.multiplier_add_columns(5, -1, 1)

    print max_scal3.invariants()

    return


if __name__ == '__main__':
    # example_michael_mentis()
    example_michael_mentis_simplified()
