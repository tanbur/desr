import itertools
import sympy
from unittest import TestCase, main

from matrix_normal_forms import is_hnf_row, hnf_row_lll, is_hnf_col, is_normal_hermite_multiplier, normal_hnf_col, hnf_row
from ode_system import ODESystem
from ode_translation import ODETranslation
from chemical_reaction_network import ChemicalReactionNetwork, ChemicalSpecies, Complex, Reaction


class TestHermiteMethods(TestCase):

    def test_example_sage(self):
        ''' Example from the Sage website
            See: http://doc.sagemath.org/html/en/reference/matrices/sage/matrix/matrix_integer_dense.html
        '''
        A = sympy.Matrix([[1,2],[3,4]])
        H, V = hnf_row_lll(A)
        self.assertEqual(H, sympy.Matrix([[1,0],[0,2]]))
        self.assertEqual(H, V * A)
        self.assertTrue(is_hnf_row(H))

        A = sympy.Matrix(range(25)).reshape(5,5)
        H, V = hnf_row_lll(A)
        H_A = sympy.Matrix.vstack(sympy.Matrix([[5,   0,  -5, -10, -15],
                                                [0,   1,   2,   3,   4]]),
                                  sympy.zeros(3, 5))
        self.assertEqual(H, H_A)
        self.assertEqual(H, V * A)
        self.assertTrue(is_hnf_row(H))

        ##TODO This test case is broken, the answer should be [[0, 1]]
        # A = sympy.Matrix([[0, -1]])
        # H, V = hnf_row_lll(A)
        # self.assertEqual(H, sympy.Matrix([[0, -1]]))
        # self.assertEqual(H, V * A)
        # self.assertTrue(is_hnf_row(H))

    def test_example1(self):
        ''' Example from Extended gcd and Hermite nomal form via lattice basis reduction '''

        A = sympy.Matrix(10, 10, lambda i, j: (i + 1) ** 3 * (j + 1) ** 2 + i + j + 2)

        H_answer_paper = sympy.Matrix([[1, 0, 7, 22, 45, 76, 115, 162, 217, 280],
                                       [0, 1, 4, 9, 16, 25, 36, 49, 64, 81],
                                       [0, 0, 12, 36, 72, 120, 180, 252, 336, 432]])
        H_answer_webcalc = sympy.Matrix([[1, 0, 7, 22, 45, 76, 115, 162, 217, 280],
                                          [0, 1, 4, 9, 16, 25, 36, 49, 64, 81],
                                          [0, 0, 12, 36, 72, 120, 180, 252, 336, 432]])

        self.assertTrue(is_hnf_row(H_answer_webcalc))
        self.assertTrue(is_hnf_row(sympy.Matrix.vstack(H_answer_webcalc, sympy.zeros(7, 10))))

        V_answer_paper = sympy.Matrix([[-10, -8, -5, 1, 2, 3, 5, 3, 0, -4],
                                       [-2, -1, 0, 1, -1, 0, 1, 0, 1, -1],
                                       [-15, -11, -4, 0, 4, 5, 4, 3, 1, -5],
                                       [1, -1, -1, 0, 2, -1, 0, 0, 0, 0],
                                       [0, 1, -1, -1, 1, -1, 2, -1, 0, 0],
                                       [1, 0, -1, -1, -1, 2, 0, 1, -1, 0],
                                       [1, 0, -1, 1, -1, 1, -1, 1, 1, -1],
                                       [-1, 0, 1, 0, 1, 1, -1, -2, 0, 1],
                                       [1, -1, 0, -1, 1, 0, 0, -1, 2, -1],
                                       [1, -2, 1, 1, -2, 0, 2, -1, 0, 0]])

        V_answer_webcalc = sympy.Matrix([[-10, -8, -5, 1, 2, 3, 5, 3, 0, -4],
                                         [-2, -1, 0, 1, -1, 0, 1, 0, 1, -1],
                                         [-15, -11, -4, 0, 4, 5, 4, 3, 1, -5],
                                         [-1, 2, -1, -1, 2, 0, -2, 1, 0, 0],
                                         [-1, 1, 0, 1, -1, 0, 0, 1, -2, 1],
                                         [1, 0, -1, 0, -1, -1, 1, 2, 0, -1],
                                         [-1, 0, 2, -1, 1, -1, 1, -1, -1, 1],
                                         [-1, 0, 1, 1, 1, -2, 0, -1, 1, 0],
                                         [0, -1, 1, 1, -1, 1, -2, 1, 0, 0],
                                         [-1, 1, 1, 0, -2, 1, 0, 0, 0, 0]])

        H, V = hnf_row_lll(A)
        H_nz, H_z = H[:3, :], H[3:, :]
        self.assertTrue(H_z.is_zero)
        self.assertEqual(H_nz, H_answer_webcalc)
        self.assertEqual(V, V_answer_webcalc)
        self.assertEqual(H, V * A)

    def test_wiki_example(self):
        ''' Test the examples from wikipedia
            https://en.wikipedia.org/wiki/Hermite_normal_form
        '''

        A1 = sympy.Matrix([[3,3,1,4],
                           [0,1,0,0],
                           [0,0,19,16],
                           [0,0,0,3]])
        H1 = sympy.Matrix([[3,0,1,1],
                           [0,1,0,0],
                           [0,0,19,1],
                           [0,0,0,3]])

        A2 = sympy.Matrix([[5, 0, 1, 4],
                           [0, -1, -4, 99],
                           [0, 20, 19, 16],
                           [0, 0, 2, 1],
                           [0, 0, 0, 3]])
        A2 = sympy.Matrix.hstack(sympy.zeros(5, 2), A2)
        A2 = sympy.Matrix.vstack(A2, sympy.zeros(1, 6))
        H2 = sympy.Matrix([[5, 0, 0, 2],
                           [0, 1, 0, 1],
                           [0, 0, 1, 2],
                           [0, 0, 0, 3]])
        H2 = sympy.Matrix.hstack(sympy.zeros(4, 2), H2)
        H2 = sympy.Matrix.vstack(H2, sympy.zeros(2, 6))

        A3 = sympy.Matrix([[2, 3, 6, 2],
                          [5, 6, 1, 6],
                          [8, 3, 1, 1]])
        H3 = sympy.Matrix([[1, 0, 50, -11],
                           [0, 3, 28,-2],
                           [0, 0, 61, -13]])

        for A, H in ((A1, H1), (A2, H2), (A3, H3)):
            H_calc, V = hnf_row_lll(A)
            self.assertTrue(is_hnf_row(H))
            self.assertEqual(H, H_calc)
            self.assertEqual(H, V * A)

    def test_normal_hermite_multiplier_example(self):
        ''' Example from Hubert Labahn '''
        # Broken due to different definitions of HNF
        A = sympy.Matrix([[8, 2, 15, 9, 11],
                          [6, 0, 6, 2, 3]])

        H_answer = sympy.Matrix.hstack(sympy.eye(2), sympy.zeros(2, 3))

        self.assertTrue(is_hnf_col(H_answer))

        H, V = normal_hnf_col(A)
        self.assertEqual(H, H_answer)
        self.assertTrue(is_hnf_col(H))

        # Check V_n is in column hnf.
        self.assertTrue(is_hnf_col(V[:, 2:]))
        self.assertTrue(is_normal_hermite_multiplier(V, A))

        # We can't compare Hermite mmultipliers since we are using different definitions
        V_answer = sympy.Matrix([[-1, -2, -2, -2, -1],
                                 [-3, -14, -7, -13, -7],
                                 [1, 1, 2, 1, 0],
                                 [0, 2, 0, 3, 0],
                                 [0, 1, 0, 0, 2]])

class TestODESystemScaling(TestCase):
    ''' Test ode_system.py scaling methods '''

    def test_example_pred_prey_hub_lab(self):
        ''' Predator prey model from Hubert Labahn Scaling symmetries paper
            dn/dt = n( r(1 - n/K) - kp/(n+d) )
            dp/dt = sp(1 - hp / n)
        '''

        equations = '''dn/dt = n*( r*(1 - n/K) - k*p/(n+d) );dp/dt = s*p*(1 - h*p / n)'''.split(';')

        system = ODESystem.from_equations(equations)

        # Swap around some columns so we agree on the order (quick and dirty)
        answer_var = 'rhKskdtnp'
        system.reorder_variables(answer_var)

        # Take the maximal scaling matrix
        max_scal = system.maximal_scaling_matrix()
        self.assertEqual(max_scal.shape, (3, 9))

        # # Now do a couple of row ops so we get exactly the same matrix. This amount to changing the scalar
        # # operations lambda1 -> lambda1^-1, swap lambda1<->lambda2 etc
        # max_scal = max_scal.extract([1, 0, 2], range(max_scal.shape[1]))
        # max_scal[1, :] -= max_scal[2, :]

        max_scal_ans = sympy.Matrix([[-1, 0, 0, -1, -1, 0, 1, 0, 0],
                                     [0, 1, 1, 0, 1, 1, 0, 1, 0],
                                     [0, -1, 0, 0, -1, 0, 0, 0, 1]])

        # Compare row HNFs
        self.assertEqual(max_scal, hnf_row(max_scal_ans)[0])


    def test_example_6_4_hub_lab(self):
        ''' Predator prey model from Hubert Labahn Scaling symmetries paper
            dz1/dt = z1*(1+z1*z2)
            dz2/dt = z2*(1/t - z1*z2)
        '''
        equations = '''dz1/dt = z1*(1+z1*z2);dz2/dt = z2*(1/t - z1*z2)'''.split(';')

        system = ODESystem.from_equations(equations)
        var_order = ['t', 'z1', 'z2']
        system.reorder_variables(var_order)

        ## Check against the answer in the paper
        # Match the maximal scaling matrix
        max_scal = system.maximal_scaling_matrix()
        self.assertTrue(max_scal == sympy.Matrix([[0, 1, -1]]))

        # Give Hermite multiplier from the paper. Padd it with a row and column for t to work with current infrastructure
        hermite_multiplier_example = sympy.Matrix([[0, 1, 0],
                                                   [1, 0, 1],
                                                   [0, 0, 1]])

        translation = ODETranslation(max_scal, hermite_multiplier=hermite_multiplier_example)

        translated = translation.translate_dep_var(system)
        answer = ODESystem.from_equations('dx0/dt = x0*(y0 + 1)\ndy0/dt = y0*(1 + 1/t)')
        self.assertEqual(translated, answer)
        self.assertEqual(translation.translate_dep_var(system), translation.translate(system))

        ## Check our answer using the general translation
        translated = translation.translate_general(system)
        answer = ODESystem.from_equations('dx0/dt = x0*(y0*y1 + y0)/t\ndy0/dt = y0/t\ndy1/dt = y1*(y0 + 1)/t')
        self.assertEqual(translated, answer)

        ## Test reverse translating
        t_var, c1_var, c2_var = sympy.var('t c1 c2')
        reduced_soln = (c2_var*sympy.exp(t_var+c1_var*(1-t_var)*sympy.exp(t_var)), c1_var * t_var * sympy.exp(t_var))
        orig_soln = translation.reverse_translate_dep_var(reduced_soln, system.indep_var_index)
        self.assertTupleEqual(orig_soln, (reduced_soln[0], reduced_soln[1] / reduced_soln[0]))

        ## Check our answer hasn't changed, using our own Hermite multiplier
        translation = ODETranslation.from_ode_system(system)
        translated = translation.translate_dep_var(system)
        answer = ODESystem.from_equations('dx0/dt = x0*(y0 - 1/t)\ndy0/dt = y0*(1 + 1/t)')
        self.assertEqual(translated, answer)

    def test_example_6_6_hub_lab(self):
        ''' Example 6.6 from Hubert Labahn Scaling symmetries paper, where we act on time
            dz1/dt = z1*(z1**5*z2 - 2)/(3*t)
            dz2/dt = z2*(10 - 2*z1**5*z2 + 3*z1**2*z2/t )/(3*t)
        '''
        equations = '''dz1/dt = z1*(z1**5*z2 - 2)/(3*t);dz2/dt = z2*(10 - 2*z1**5*z2 + 3*z1**2*z2/t )/(3*t)'''.split(';')

        system = ODESystem.from_equations(equations)
        system.reorder_variables(['t', 'z1', 'z2'])

        ## Check against the answer in the paper
        # Match the maximal scaling matrix
        max_scal = system.maximal_scaling_matrix()
        max_scal_ans = sympy.Matrix([[3, -1, 5]])
        self.assertEqual(max_scal, max_scal_ans)

        # Give Hermite multiplier from the paper. Padd it with a row and column for t to work with current infrastructure
        hermite_multiplier_ans = sympy.Matrix([[1, 1, -1],
                                               [2, 3, 2],
                                               [0, 0, 1]])

        translation = ODETranslation(max_scal_ans, hermite_multiplier=hermite_multiplier_ans)

        translated = translation.translate(system)
        answer = ODESystem.from_equations('dx0/dt = x0*(2*y0*y1/3 - 1/3)/t\ndy0/dt = y0*(y0*y1 - 1)/t\ndy1/dt = y1*(y1 + 1)/t')
        self.assertEqual(translated, answer)

        ## Check reverse translation
        reduced_soln = (sympy.var('t'),
                        sympy.sympify('c3/(t**(1/3)*(ln(t-c1)-ln(t)+c2)**(2/3))'),  # x
                        sympy.sympify('c1/(t*(ln(t-c1)-ln(t)+c2))'),  # y1
                        sympy.sympify('t/(c1 - t)'))  # y2

        orig_soln = translation.reverse_translate_general(reduced_soln)

        # Solution from the paper
        orig_soln_paper = [reduced_soln[2] / reduced_soln[1],
                                          reduced_soln[1] ** 5 * reduced_soln[3] / reduced_soln[2] ** 4]
        orig_soln_paper = [soln.subs({sympy.var('t'): sympy.sympify('t / (c3**3 / c1**2)')}) for soln in orig_soln_paper]

        self.assertEqual(len(orig_soln), len(orig_soln_paper))
        for sol1, sol2 in zip(orig_soln, orig_soln_paper):
            self.assertEqual(sol1, sol2)

        ## Check our answer hasn't changed, using our own Hermite multiplier
        translation = ODETranslation.from_ode_system(system)
        translated = translation.translate(system)
        answer = ODESystem.from_equations('dx0/dt = x0*(2*y1/3 + 2/3 + y1/y0)/t\ndy0/dt = y0*(y1 - 1)/t\ndy1/dt = y1*(5*y1/3 - 10/3 + (2*y0*(-y1 + 5)/3 + y1)/y0)/t')
        self.assertEqual(translated, answer)


    def test_verhulst_log_growth(self):
        ''' Verhult logistic growth model from Hubert Labahn Scaling symmetries paper
            dn/dt = r*n*(1-n/k)
        '''
        equations = '''dn/dt = r*n*(1-n/k)'''

        system = ODESystem.from_equations(equations)
        system.reorder_variables('rktn')

        ## Check against the answer in the paper
        # Match the maximal scaling matrix
        max_scal = system.maximal_scaling_matrix()
        # Compare to the paper by performing one row operation
        max_scal[0, :] *= -1
        self.assertEqual(max_scal, sympy.Matrix([[-1, 0, 1, 0], [0, 1, 0, 1]]))

        herm_mult_paper = sympy.Matrix([[-1, 0, 1, 0],
                                       [0, 1, 0, -1],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]])
        translator = ODETranslation(max_scal,
                                    variables_domain=system.variables,
                                    hermite_multiplier=herm_mult_paper)
        translated = translator.translate(system)
        answer = ODESystem.from_equations('dx0/dt = 0\ndx1/dt = 0\ndy0/dt = y0/t\ndy1/dt = y1*(-y0*y1 + y0)/t')
        self.assertEqual(translated, answer)

        ## Check reverse translation
        reduced_solutions = (sympy.var('t'),  # indep var t
                     sympy.var('c1'),  # x0
                     sympy.var('c2'),  # x1
                     sympy.sympify('t'),  # y0
                     sympy.sympify('1/(1+c*exp(-t))'))  # y1

        general_soln = translator.reverse_translate_general(reduced_solutions, system_indep_var_index=2)
        paper_soln = tuple(map(sympy.sympify, ['1/c1', 'c2', 'c2/(c*exp(-t/c1) + 1)']))
        self.assertTupleEqual(general_soln, paper_soln)

        ## Now do it with our own Hermite multiplier, except we need the variables in a different order
        system.reorder_variables('tnrk')
        translator = ODETranslation.from_ode_system(system)
        translated = translator.translate_general(system)
        self.assertEqual(translated, answer)
        general_soln = translator.reverse_translate_general(reduced_solutions, system_indep_var_index=0)
        saved_soln = tuple(map(sympy.sympify, ['c2/(c*exp(-t/c1) + 1)', '1/c1', 'c2']))  # Just a cached value
        self.assertTupleEqual(general_soln, saved_soln)

    def test_example_pred_prey_choosing_invariants(self):
        ''' Predator prey model from Hubert Labahn Scaling symmetries paper
            dn/dt = n( r(1 - n/K) - kp/(n+d) )
            dp/dt = sp(1 - hp / n)

            Rather than using the invariants suggested by the algorithm, we pick our own and extend it.
        '''
        equations = '''dn/dt = n*( r*(1 - n/K) - k*p/(n+d) );dp/dt = s*p*(1 - h*p / n)'''.split(';')
        system = ODESystem.from_equations(equations)

        # Take the maximal scaling matrix
        max_scal = ODETranslation.from_ode_system(system)

        t, n, p, K, d, h, k, r, s = max_scal.variables_domain

        self.assertTupleEqual(tuple(max_scal.invariants()), (s*t, n / d, k*p/(d*s), K/d, h*s/k, r/s))

        # Choose a new invariant
        invariant_choice = sympy.Matrix([[1, 0, 0, 0, 0, 0, 0, 1, 0],  # t * r
                                        [0, 0, 1, 0, -1, 1, 0, 0, 0],  # p * h /d
                                        ]).T  # Note the transpose! Each column expresses an invariant

        max_scal2 = max_scal.extend_from_invariants(invariant_choice=invariant_choice)
        self.assertTupleEqual(tuple(max_scal2.invariants()), (t*r, h*p/d, n / d, K/d, h*s/k, r/s))

        # This should work even if we move the time about
        invariant_choice = sympy.Matrix([[0, 0, 1, 0, -1, 1, 0, 0, 0],  # p * h /d
                                        ]).T  # Note the transpose! Each column expresses an invariant
        max_scal3 = max_scal.extend_from_invariants(invariant_choice=invariant_choice)
        self.assertTupleEqual(tuple(max_scal3.invariants()), (h*p/d, t*s, n / d, K/d, h*s/k, r/s))


        reduced_system = max_scal.translate(system=system)
        reduced_system2 = max_scal2.translate(system=system)

        self.assertNotEqual(reduced_system, reduced_system2)


class TestChemicalReactionNetwork(TestCase):
    ''' Test cases for the ChemicalReactionNetwork class '''

    def test_crn_harrington(self):
        ''' Example 2.8 from Harrington - Joining and decomposing '''
        species = sympy.var('x1 x2')
        species = map(ChemicalSpecies, species)
        x1, x2 = species

        complex0 = Complex()
        complex1 = Complex({x1: 1})
        complex2 = Complex({x2: 1})
        complexes = (complex0, complex1, complex2)

        r01 = Reaction(complex0, complex1)
        r12 = Reaction(complex1, complex2)
        r21 = Reaction(complex2, complex1)
        r20 = Reaction(complex2, complex0)
        reactions = [r01, r12, r21, r20]

        reaction_network = ChemicalReactionNetwork(species, complexes, reactions)

        system = reaction_network.to_ode_system()

        answer = ODESystem.from_equations('dx1/dt = k_0_1 - k_1_2*x1 + k_2_1*x2\ndx2/dt = k_1_2*x1 - k_2_0*x2 - k_2_1*x2')

        self.assertEqual(system, answer)

    def test_crn_harrington2(self):
        ''' Example 1 from Harrington board notes - Joining and decomposing '''
        species = sympy.var('x1 x2')
        species = map(ChemicalSpecies, species)
        x1, x2 = species

        complex0 = Complex({x1: 1, x2: 1})
        complex1 = Complex({x2: 2})
        complex2 = Complex({x1: 1})
        complex3 = Complex({x2: 1})
        complexes = (complex0, complex1, complex2, complex3)

        r1 = Reaction(complex0, complex1)
        r2 = Reaction(complex3, complex2)
        reactions = [r1, r2]

        reaction_network = ChemicalReactionNetwork(species, complexes, reactions)

        system = reaction_network.to_ode_system()

        answer = ODESystem.from_equations('dx1/dt = -k_0_1*x1*x2 + k_3_2*x2\ndx2/dt = k_0_1*x1*x2 - k_3_2*x2')

        self.assertEqual(system, answer)

if __name__ == '__main__':
    main()