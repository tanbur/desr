
import sympy
from ode_system import ODESystem
from ode_translation import ODETranslation, scale_action


def example_two_step_phosphorelay():
    ''' Two-step Phosphorelay example.
        Let S be a signal that activates X 1 . The activated form of X 1 gives its phosphate group to X 2
        to activate it. Both X 1 and X 2 have phosphatases, F 1 and F 2 , respectively. Y 1 , Y 2 and Y 3 are
        intermediate species.
    '''
    variables = 'Xs1 X1 Xs2 X2 S0 F1 F2 Y1 Y2 Y3'
    variables = sympy.var(variables)
    Xs1, X1, Xs2, X2, S0, F1, F2, Y1, Y2, Y3 = variables

    equations = [
        'k2*Y1 + km3*Y2 - (k3*F1 + k5*X2)*Xs1',  # Xs1
        'km1*Y1 + k4*Y2 + k5*Xs1*X2 - k1*X1*S0',  # X1
        'km5*Y3 + k5*Xs1*X2 - k6*Xs2*F2',  # Xs2
        'k7*Y3 - k5*Xs1*X2',  # X2
        '(km1 + k2)*Y1 - k1*X1*S0',  # S
        '(km3 + k4)*Y2 - k3*Xs1*F1',  # F1
        '(km6 + k7)*Y3 - k6*Xs2*F2',  # F2
    ]

    equations = map(sympy.sympify, equations)

    deriv_dict = dict(zip(variables, equations))

    deriv_dict[Y1] = - deriv_dict[S0]
    deriv_dict[Y2] = - deriv_dict[F1]
    deriv_dict[Y3] = - deriv_dict[F2]

    eq_const = sum([deriv_dict[v] for v in [Xs1, X1, Xs2, X2, Y1, Y2, Y3]])
    km5, km6 = sympy.var('km5 km6')
    assert eq_const.expand() == Y3*km5 - Y3 * km6
    import warnings
    warnings.warn('{}  =>  km5 = km6'.format(eq_const.expand()))
    for key in deriv_dict.keys():
        deriv_dict[key] = deriv_dict[key].subs({km6: km5})

    system = ODESystem.from_dict(deriv_dict)
    return system

def example_one_site_modification():
    ''' One-site modification example.
        Consider the substrate S0 , with kinase E and phosphatase F , and active form S1. X and Y are
        the intermediate species. As before, all the species need to be highly diffusible in order to verify
        the conservation laws.
    '''
    variables = 'S0 S1 Ev F X Y'
    variables = sympy.var(variables)
    S0, S1, Ev, F, X, Y = variables

    equations = [
        'k4*Y + km1*X - k1*S0*Ev',  # S0
        'k2*X + km3*Y - k3*S1*F',  # S1
        '(km1 + k2)*X - k1*S0*Ev',  # Ev
        '(km3 + k4)*Y - k3*S1*F',  # F
    ]

    equations = map(sympy.sympify, equations)

    deriv_dict = dict(zip([S0, S1, Ev, F], equations))
    deriv_dict[X] = - deriv_dict[Ev]
    deriv_dict[Y] = - deriv_dict[F]

    eq_const = sum([deriv_dict[v] for v in [S0, S1, X, Y]])
    assert eq_const.expand() == 0

    system = ODESystem.from_dict(deriv_dict)
    return system

def example_two_site_modification():
    ''' Two-site modification example.
        Consider species S0, S1 and S2, with kinases E1 and E2, and phosphatases F1 and F2. X1, X2, Y1
        and Y2 are intermediate species.
    '''
    variables = 'S0, S1, S2, Ev1, Ev2, F1, F2, X1, X2, Y1, Y2'
    variables = sympy.var(variables)
    S0, S1, S2, Ev1, Ev2, F1, F2, X1, X2, Y1, Y2 = variables

    equations = [
        'k4*Y1 + km1*X1 - k1*S0*Ev1',  # S0
        'k2*X1 + km3*Y1 + km5*X2 + k8*Y2 - (k3*F1 + k5*Ev2)*S1',  # S1
        'k6*X2 + km7*Y2 - k7*S2*F2',  # S2
        '(km1 + k2)*X1 - k1*S0*Ev1',  # Ev1
        '(km5 + k6)*X2 - k5*S1*Ev2',  # Ev2
        '(km3 + k4)*Y1 - k3*S1*F1',  # F1
        '(km7 + k8)*Y2 - k7*S2*F2',  # F2
    ]

    equations = map(sympy.sympify, equations)

    deriv_dict = dict(zip([S0, S1, S2, Ev1, Ev2, F1, F2], equations))
    deriv_dict[X1] = - deriv_dict[Ev1]
    deriv_dict[X2] = - deriv_dict[Ev2]
    deriv_dict[Y1] = - deriv_dict[F1]
    deriv_dict[Y2] = - deriv_dict[F2]

    eq_const = sum([deriv_dict[v] for v in [S0, S1, S2, X1, X2, Y1, Y2]])
    assert eq_const.expand() == 0

    system = ODESystem.from_dict(deriv_dict)
    return system


def example_two_substrate_modification():
    ''' Modification of two substrates example.
        Consider the kinase E acting on the substrates S0 and P0 , with phosphatases F1 and F2 ,
        respectively.
        X1 , X2 , Y1 and Y2 are intermediate species. Once again, all the species are highly
        diffusible.
    '''
    variables = 'S0, S1, F1, Ev, P0, P1, F2, X1, X2, Y1, Y2'
    variables = sympy.var(variables)
    S0, S1, F1, Ev, P0, P1, F2, X1, X2, Y1, Y2 = variables

    equations = [
        'km1*X1 + k4*Y1 - k1*S0*Ev',  # S0
        'km3*Y1 + k2*X1 - k3*S1*F1',  # S1
        '(km3 + k4)*Y1 - k3*S1*F1',  # F1
        '(km1 + k2)*X1 + (km5 + k6)*X2 - (k1*S0 + k5*P0)*Ev',  # Ev
        'km5*X2 + k8*Y2 - k5*P0*Ev',  # P0
        'km7*Y2 + k6*X2 - k7*P1*F2',  # P1
        '(km7 + k8)*Y2 - k7*P1*F2',  # F2
    ]

    equations = map(sympy.sympify, equations)

    deriv_dict = dict(zip([S0, S1, F1, Ev, P0, P1, F2], equations))
    deriv_dict[Y1] = - deriv_dict[F1]
    deriv_dict[Y2] = - deriv_dict[F2]
    deriv_dict[X1] = - deriv_dict[Y1] - deriv_dict[S0] - deriv_dict[S1]
    deriv_dict[X2] = - deriv_dict[Y2] - deriv_dict[P0] - deriv_dict[P1]

    eq_const = sum([deriv_dict[v] for v in [Ev, X1, X2]])
    assert eq_const.expand() == 0

    system = ODESystem.from_dict(deriv_dict)
    return system

def example_two_layer_cascade():
    ''' Two layer cascade example.
        Consider a substrate S0 with kinase E and phosphatase F1 , with active form S1 that acts as a
        kinase on the substrate P0 , which has phosphatase F2 . All the species are highly diffusible.
    '''
    variables = 'S0, S1, P0, P1, Ev, F1, F2, X1, X2, Y1, Y2'
    variables = sympy.var(variables)
    S0, S1, P0, P1, Ev, F1, F2, X1, X2, Y1, Y2 = variables

    equations = [
        'km1*X1 + k4*Y1 - k1*S0*Ev',  # S0
        'k2*X1 + (km5 + k6)*X2 + km3*Y1 - (k3*F1 + k5*P0)*S1',  # S1
        'km5*X2 + k8*Y2 - k5*P0*S1',  # P0
        'km7*Y2 + k6*X2 - k7*P1*F2',  # P1
        '(km1 + k2)*X1 - k1*S0*Ev',  # Ev
        '(km3 + k4)*Y1 - k3*S1*F1',  # F1
        '(km7 + k8)*Y2 - k7*P1*F2',  # F2
    ]

    equations = map(sympy.sympify, equations)

    deriv_dict = dict(zip([S0, S1, P0, P1, Ev, F1, F2], equations))
    deriv_dict[X1] = - deriv_dict[Ev]
    deriv_dict[Y1] = - deriv_dict[F1]
    deriv_dict[Y2] = - deriv_dict[F2]
    deriv_dict[X2] = - deriv_dict[Y2] - deriv_dict[P0] - deriv_dict[P1]

    eq_const = sum([deriv_dict[v] for v in [S0, S1, X1, Y1, X2]])
    assert eq_const.expand() == 0

    system = ODESystem.from_dict(deriv_dict)
    return system


def main():
    examples = (('Two Step Phosphorelay', example_two_step_phosphorelay),
                ('One Site Modification', example_one_site_modification),
                ('Two Site Modification', example_two_site_modification),
                ('Two Substrate Modification', example_two_substrate_modification),
                ('Two Layer Cascade', example_two_layer_cascade),
                )

    for name, example in examples:
        print name
        system = example()
        max_scal = system.maximal_scaling_matrix()
        print 'Original system:\n{}\n'.format(system)
        print 'Scaling action:\n{}\n{}\n'.format(system.variables, max_scal)
        translation = ODETranslation(max_scal)
        print 'Hermite multiplier:\n{}\n'.format(translation.herm_mult)

        print 'Invariants of the system:\n{}\n'.format(translation.invariants(system.variables))
        reduced = translation.translate_general(system)
        print 'Reduced system:\n{}\n'.format(reduced)
        print '\n\n' + '*' * 10 + '\n\n'


if __name__ == '__main__':
    main()