
##### Richard Tanburn
import numpy
import itertools

import diophantine

INT_TYPES = (numpy.int64, int,)
INT_TYPE_DEF = INT_TYPES[0]

# ## Code in case we don't want to use the above
# def hnf_lll(G):
#     ''' Algorithm taken from
#     Extended gcd and Hermite nomal form via lattice basis reduction
#
#     '''
#     m, n = G.shape
#     B = numpy.eye(m, dtype=INT_TYPE_DEF)
#     A = numpy.array(G, dtype=INT_TYPE_DEF)
#     L = numpy.zeros((m, m), dtype=INT_TYPE_DEF)
#
#     D_array = numpy.ones(m + 1, dtype=INT_TYPE_DEF)
#     m1, n1 = 3, 4
#     k = 2
#     while k <= m:
#         col1, col2 = _hnf_lll_reduce(k, k-1, A, B)
# #        if (())
#
# def _hnf_lll_reduce(k, i, A, B):
#     ''' Reduce2 function from the paper '''
#     pivot_ind = get_pivot_row_indices(A)
#     col1 = pivot_ind[i]
#     if A[i, col1] < 0:
#         _hnf_minus(i)
#         B[i] *= -1
#     else:
#         col1 = A.shape[1] + 1

def get_pivot_row_indices(matrix_):
    ''' Return the pivot indices of the matrix '''
    indices = []
    for row_ind, row in enumerate(matrix_):
        for ind, col in enumerate(row):
            if col != 0:
                indices.append(ind)
                break
        # If we haven't found an index. throw an error, as we're assuming the rows are non-zero
        if len(indices) != row_ind + 1:
            raise ValueError('Zero row in {}'.format(matrix_))
    return indices

def is_hnf_col(matrix_):
    ''' Decide whether a matrix is in Hermite normal form, when acting on rows

    '''
    return is_hnf_row(matrix_.T)

def is_hnf_row(matrix_):
    ''' Decide whether a matrix is in Hermite normal form, when acting on rows.
        This is according to the Havas Majewski Matthews definition.

    '''
    m, n = matrix_.shape

    # The first r rows are non-zero. Determine r and check rows r through m are 0.
    row_is_zero = [numpy.all(row == 0) for row in matrix_]
    r = m - sum(map(int, row_is_zero))
    for i in xrange(r, m):
        if not row_is_zero[i]:
            return False

    # If b_{i,j_i} are the indices of the first non-zero entry of row i then j_i are strictly ascending
    indices = get_pivot_row_indices(matrix_[:r])
    for i in xrange(len(indices) - 1):
        if indices[i] >= indices[i + 1]:
            return False

    # Check each of the pivots are positive
    for i in xrange(r):
        if matrix_[i, indices[i]] <= 0:
            return False

    # Check we are 0 to the right of the pivot
    for i in xrange(r):
        for k in xrange(i):
            if not (0 <= matrix_[k, indices[i]] < matrix_[i, indices[i]]):
                return False
    return True

def hnf_row_lll(matrix_):
    ''' Compute the Hermite normal form, ACTS ON THE ROWS OF A MATRIX
        Input: integer mxn matrix A, nonzero, at least two rows
        Output: small unimodular matrix B and HNF(A), such that BA=HNF(A)+
        The Havas, Majewski, Matthews LLL method is used
        We usually take alpha=m1/n1, with (m1,n1)=(1,1) to get best results
    '''
    while len(matrix_.shape) < 2:
        matrix_ = numpy.expand_dims(matrix_, axis=0)
    hnf, unimodular_matrix, rank = diophantine.lllhermite(matrix_, m1=1, n1=1)
    assert numpy.abs(numpy.abs(numpy.linalg.det(unimodular_matrix)) - 1) < 1e-10
    assert is_hnf_row(hnf)
    return hnf, unimodular_matrix

def hnf_col_lll(matrix_):
    ''' Compute the Hermite normal form, ACTS ON THE COLUMNS OF A MATRIX
        Input: integer mxn matrix A, nonzero, at least two rows
        Output: small unimodular matrix B and HNF(A), such that BA=HNF(A)+
        The Havas, Majewski, Matthews LLL method is used
        We usually take alpha=m1/n1, with (m1,n1)=(1,1) to get best results

        >>> A = numpy.array([[8, 2, 15, 9, 11],
        ...                  [6, 0, 6, 2, 3]])
        >>> h, v = hnf_col(A)
        >>> print h
        [[1 0 0 0 0]
         [0 1 0 0 0]]
        >>> print v
        [[-1  0  0 -1 -1]
         [-3 -1  6 -1  0]
         [ 1  0  1  1  2]
         [ 0 -1 -3 -3  0]
         [ 0  1  0  2 -2]]
         >>> numpy.all(numpy.dot(A, v) == h)
         True
    '''
    hnf, unimod = hnf_row_lll(matrix_.T)
    return hnf.T, unimod.T


## Normal Hermite multiplier from Hubert-Labahn
def is_normal_hermite_multiplier(hermite_multiplier, matrix_):
    ''' Determine whether hermite_multiplier is the normal Hermite multiplier of matrix.
        This is the COLUMN version.
    '''
    if numpy.linalg.matrix_rank(matrix_) != matrix_.shape[0]:
        raise ValueError('Matrix must have full row rank')

    r, n = matrix_.shape
    if hermite_multiplier.shape != (n, n):
        raise ValueError("Matrix dimensions don't match")

    # Check matrix . hermite_multiplier = [H 0]
    prod = numpy.dot(matrix_, hermite_multiplier)
    hermite_form, residue = prod[:, :r], prod[:, r:]
    # Condition a)
    if not numpy.all(residue == 0):
        return False
    if not is_hnf_col(hermite_form):
        return False
    # Condition b)
    herm_mul_i, herm_mul_n = hermite_multiplier[:, :r], hermite_multiplier[:, r:]
    if not is_hnf_col(herm_mul_n):
        return False
    # Condition c)
    for i in xrange(r):
        pivot_val = numpy.max(herm_mul_n[i])
        if numpy.any(numpy.abs(herm_mul_i[i]) >= pivot_val):
            return False

    return True

def normal_hnf_col(matrix_):
    ''' Return the hnf and the unique normal Hermite multiplier

        >>> A = numpy.array([[8, 2, 15, 9, 11],
        ...                  [6, 0, 6, 2, 3]])
        >>> h, v = normal_hnf_col(A)
        >>> print h
        [[1 0 0 0 0]
         [0 1 0 0 0]]
        >>> print v
        [[  0   0   1   0   0]
         [  0   0   0   1   0]
         [  2   1   3   1   5]
         [  9   2  21   3  21]
         [-10  -3 -22  -4 -24]]
         >>> numpy.all(numpy.dot(A, v) == h)
         True
    '''
    r, n = matrix_.shape
    out, _ = hnf_col_lll(numpy.vstack((matrix_, numpy.eye(n))))
    h, v = out[:r], out[r:]
    from numpy.testing import assert_array_equal
    assert_array_equal(numpy.dot(matrix_, v), h)
    return h, v

def normal_hnf_row(matrix_):
    ''' Row version of the normal hnf multiplier '''
    h, v = normal_hnf_col(matrix_=matrix_.T)
    return h.T, v.T

## Set defaults
hnf_row = hnf_row_lll
hnf_col = hnf_col_lll


if __name__ == '__main__':
    import doctest
    doctest.testmod()