"""Michael-Mentis Equations

This example script walks through a possible analysis of the Michael Mentis equations.


"""
import sympy
from desr.ode_system import ODESystem
from desr.ode_translation import ODETranslation, scale_action
from desr.tex_tools import expr_to_tex

def example_michael_mentis():
    """ Example """
    # Enable the best printing
    sympy.init_printing()

    system_tex = '''\frac{dE}{dt} &= - k_1 E S + k_{-1} C + k_2 C \\\\
             \frac{dS}{dt} &= - k_1 E S + k_{-1} C \\\\
             \frac{dC}{dt} &= k_1 E S - k_{-1} C - k_2 C \\\\
             \frac{dP}{dt} &= k_2 C'''
    # print system_tex
    original_system = ODESystem.from_tex(system_tex)
    print original_system

    max_scal1 = ODETranslation.from_ode_system(original_system)
    print ',\quad '.join(map(expr_to_tex, max_scal1.variables_domain))
    print sympy.latex(max_scal1.scaling_matrix)
    print sympy.latex(max_scal1.herm_mult)
    print max_scal1.invariants()
    print ',\n\\qquad\n'.join(map(expr_to_tex, max_scal1.invariants()))
    print max_scal1.translate(original_system)
    print

    # Show what happens when we put k_2 at the end instead
    variable_order = list(original_system.variables)
    variable_order[-1], variable_order[-2] = variable_order[-2], variable_order[-1]
    original_system.reorder_variables(variable_order)
    max_scal1 = ODETranslation.from_ode_system(original_system)
    print ',\n\\qquad\n'.join(map(expr_to_tex, max_scal1.invariants()))

    # Now do the reduction
    reduced_system = max_scal1.translate(original_system)
    print max_scal1._is_translate_parameter_compatible(original_system)
    print reduced_system

if __name__ == '__main__':
    example_michael_mentis()