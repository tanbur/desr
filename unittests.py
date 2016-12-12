import itertools
import numpy
from numpy.testing import assert_array_equal
from unittest import TestCase, main

from hermite_helper import is_hnf_row, INT_TYPE_DEF, hnf_row_lll, is_hnf_col, is_normal_hermite_multiplier, normal_hnf_col
from ode_system import ODESystem
from ode_translation import ODETranslation


class TestHermiteMethods(TestCase):

    def test_example_sage(self):
        ''' Example from the Sage website
            See: http://doc.sagemath.org/html/en/reference/matrices/sage/matrix/matrix_integer_dense.html
        '''
        A = numpy.array([[1,2],[3,4]], dtype=INT_TYPE_DEF)
        H, V = hnf_row_lll(A)
        assert_array_equal(H, numpy.array([[1,0],[0,2]], dtype=INT_TYPE_DEF))
        assert_array_equal(H, numpy.dot(V, A))
        self.assertTrue(is_hnf_row(H))

        A = numpy.arange(25, dtype=INT_TYPE_DEF).reshape((5,5))
        H, V = hnf_row_lll(A)
        H_A = numpy.vstack([numpy.array([[5,   0,  -5, -10, -15],
                           [0,   1,   2,   3,   4]]),numpy.zeros((3, 5))]).astype(INT_TYPE_DEF)
        assert_array_equal(H, H_A)
        assert_array_equal(H, numpy.dot(V, A))
        self.assertTrue(is_hnf_row(H))

        ##TODO This test case is broken, the answer should be [[0, 1]]
        # A = numpy.array([[0, -1]], dtype=INT_TYPE_DEF)
        # H, V = hnf_row_lll(A)
        # assert_array_equal(H, numpy.array([[0, -1]], dtype=INT_TYPE_DEF))
        # assert_array_equal(H, numpy.dot(V, A))
        # #self.assertTrue(is_hnf(H))

    def test_example1(self):
        ''' Example from Extended gcd and Hermite nomal form via lattice basis reduction '''

        A = numpy.empty((10, 10), dtype=INT_TYPE_DEF)
        for i, j in itertools.product(range(10), repeat=2):
            A[i, j] = (i + 1) ** 3 * (j + 1) ** 2 + i + j + 2

        H_answer_paper = numpy.array([[1, 0, 7, 22, 45, 76, 115, 162, 217, 280],
                                     [0, 1, 4, 9, 16, 25, 36, 49, 64, 81],
                                     [0, 0, 12, 36, 72, 120, 180, 252, 336, 432]], dtype=INT_TYPE_DEF)
        H_answer_webcalc = numpy.array([[1, 0, 7, 22, 45, 76, 115, 162, 217, 280],
                                        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81],
                                        [0, 0, 12, 36, 72, 120, 180, 252, 336, 432]], dtype=INT_TYPE_DEF)

        self.assertTrue(is_hnf_row(H_answer_webcalc))
        self.assertTrue(is_hnf_row(numpy.vstack([H_answer_webcalc, numpy.zeros((7, 10))]).astype(INT_TYPE_DEF)))

        V_answer_paper = numpy.array([[-10, -8, -5, 1, 2, 3, 5, 3, 0, -4],
                                [-2, -1, 0, 1, -1, 0, 1, 0, 1, -1],
                                [-15, -11, -4, 0, 4, 5, 4, 3, 1, -5],
                                [1, -1, -1, 0, 2, -1, 0, 0, 0, 0],
                                [0, 1, -1, -1, 1, -1, 2, -1, 0, 0],
                                [1, 0, -1, -1, -1, 2, 0, 1, -1, 0],
                                [1, 0, -1, 1, -1, 1, -1, 1, 1, -1],
                                [-1, 0, 1, 0, 1, 1, -1, -2, 0, 1],
                                [1, -1, 0, -1, 1, 0, 0, -1, 2, -1],
                                [1, -2, 1, 1, -2, 0, 2, -1, 0, 0]], dtype=INT_TYPE_DEF)

        V_answer_webcalc = numpy.array([[-10, -8, -5, 1, 2, 3, 5, 3, 0, -4],
                                        [-2, -1, 0, 1, -1, 0, 1, 0, 1, -1],
                                        [-15, -11, -4, 0, 4, 5, 4, 3, 1, -5],
                                        [-1, 2, -1, -1, 2, 0, -2, 1, 0, 0],
                                        [-1, 1, 0, 1, -1, 0, 0, 1, -2, 1],
                                        [1, 0, -1, 0, -1, -1, 1, 2, 0, -1],
                                        [-1, 0, 2, -1, 1, -1, 1, -1, -1, 1],
                                        [-1, 0, 1, 1, 1, -2, 0, -1, 1, 0],
                                        [0, -1, 1, 1, -1, 1, -2, 1, 0, 0],
                                        [-1, 1, 1, 0, -2, 1, 0, 0, 0, 0]], dtype=INT_TYPE_DEF)

        H, V = hnf_row_lll(A)
        H_nz, H_z = H[:3], H[3:]
        self.assertTrue(numpy.all(H_z == 0))
        assert_array_equal(H_nz, H_answer_webcalc)
        assert_array_equal(V, V_answer_webcalc)
        assert_array_equal(H, numpy.dot(V, A))

    def test_wiki_example(self):
        ''' Test the examples from wikipedia
            https://en.wikipedia.org/wiki/Hermite_normal_form
        '''

        A1 = numpy.array([[3,3,1,4],
                         [0,1,0,0],
                         [0,0,19,16],
                         [0,0,0,3]])
        H1 = numpy.array([[3,0,1,1],
                         [0,1,0,0],
                         [0,0,19,1],
                         [0,0,0,3]])

        A2 = numpy.array([[5, 0, 1, 4],
                          [0, -1, -4, 99],
                          [0, 20, 19, 16],
                          [0, 0, 2, 1],
                          [0, 0, 0, 3]])
        A2 = numpy.hstack((numpy.zeros((5, 2)), A2))
        A2 = numpy.vstack((A2, numpy.zeros((1, 6))))
        H2 = numpy.array([[5, 0, 0, 2],
                          [0, 1, 0, 1],
                          [0, 0, 1, 2],
                          [0, 0, 0, 3]])
        H2 = numpy.hstack((numpy.zeros((4, 2)), H2))
        H2 = numpy.vstack((H2, numpy.zeros((2, 6))))

        A3 = numpy.array([[2, 3, 6, 2],
                         [5, 6, 1, 6],
                         [8, 3, 1, 1]])
        H3 = numpy.array([[1, 0, 50, -11],
                          [0, 3, 28,-2],
                          [0, 0, 61, -13]])

        for A, H in ((A1, H1), (A2, H2), (A3, H3)):
            H_calc, V = hnf_row_lll(A)
            self.assertTrue(is_hnf_row(H))
            assert_array_equal(H, H_calc)
            assert_array_equal(H, numpy.dot(V, A))

    def test_normal_hermite_multiplier_example(self):
        ''' Example from Hubert Labahn '''
        # Broken due to different definitions of HNF
        A = numpy.array([[8, 2, 15, 9, 11],
                         [6, 0, 6, 2, 3]])

        H_answer = numpy.hstack([numpy.eye(2), numpy.zeros((2, 3))]).astype(INT_TYPE_DEF)

        self.assertTrue(is_hnf_col(H_answer))

        H, V = normal_hnf_col(A)
        assert_array_equal(H, H_answer)
        self.assertTrue(is_hnf_col(H))

        # Check V_n is in column hnf.
        self.assertTrue(is_hnf_col(V[:, 2:]))
        self.assertTrue(is_normal_hermite_multiplier(V, A))

        # We can't compare Hermite mmultipliers since we are using different definitions
        V_answer = numpy.array([[-1, -2, -2, -2, -1],
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

        # Now do a couple of row ops so we get exactly the same matrix. This amount to changing the scalar
        # operations lambda1 -> lambda1^-1, swap lambda1<->lambda2 etc
        max_scal = max_scal[[1, 0, 2]]
        max_scal[1] -= max_scal[2]

        max_scal_ans = numpy.array([[-1, 0, 0, -1, -1, 0, 1, 0, 0],
                                    [0, 1, 1, 0, 1, 1, 0, 1, 0],
                                    [0, -1, 0, 0, -1, 0, 0, 0, 1]])

        self.assertTrue(numpy.all(max_scal == max_scal_ans))


    def test_example_6_4_hub_lab(self):
        ''' Predator prey model from Hubert Labahn Scaling symmetries paper
            dz1/dt = z1*(1+z1*z2)
            dz2/dt = z2*(1/t - z1*z2)
        '''
        equations = '''dz1/dt = z1*(1+z1*z2);dz2/dt = z2*(1/t - z1*z2)'''.split(';')

        system = ODESystem.from_equations(equations)
        # Match the maximal scaling matrix
        max_scal = system.maximal_scaling_matrix()
        # Multiply by -1 (a trivial row operation) so that answers match.
        self.assertTrue(numpy.all(- max_scal == numpy.array([[1, -1, 0]])))

        translation = ODETranslation(max_scal)

        translated = translation.translate_dep_var(system)

        self.assertEqual(str(translated), 'dy0/dt = y0*(1 + 1/t)\ndt/dt = 1\ndx0/dt = x0*(-y0 + 1/t)')

if __name__ == '__main__':
    main()