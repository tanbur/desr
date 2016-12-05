import itertools
import numpy
from numpy.testing import assert_array_equal
from unittest import TestCase, main

from hermite_helper import is_hnf_row, INT_TYPE_DEF, hnf_row_lll, is_hnf_col, hnf_col_lll


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

    def test_example2(self):
        ''' Example from Hubert Labahn '''
        A = numpy.array([[8, 2, 15, 9, 11],
                         [6, 0, 0, 2, 3]])

        H_answer = numpy.hstack([numpy.eye(2), numpy.zeros((2, 3))]).astype(INT_TYPE_DEF)

        self.assertTrue(is_hnf_col(H_answer))

        V_answer = numpy.array([[-1, -2, -2, -2, -1],
                             [-3, -14, -7, -13, -7],
                             [1, 1, 2, 1, 0],
                             [0, 2, 0, 3, 0],
                             [0, 1, 0, 0, 2]])

        H, V = hnf_col_lll(A)
        assert_array_equal(H, H_answer)
        self.assertTrue(is_hnf_col(H))


if __name__ == '__main__':
    main()