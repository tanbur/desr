"""Michael-Mentis Equations

This example script walks through a possible analysis of the Michael Mentis equations.


"""
import sympy
from desr.matrix_normal_forms import smf
from desr.ode_system import ODESystem
from desr.ode_translation import ODETranslation, scale_action
from desr.tex_tools import expr_to_tex

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



if __name__ == '__main__':
    example_michael_mentis()