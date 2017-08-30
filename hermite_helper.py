
##### Richard Tanburn
import numpy
import sympy
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

    >>> matrix_ = numpy.array([[2, 0],
    ...                        [3, 3],
    ...                        [0, 0]])
    >>> result = hnf_row_lll(matrix_)
    >>> result[0]
    array([[1, 3],
           [0, 6],
           [0, 0]])
    >>> result[1]
    array([[-1,  1,  0],
           [-3,  2,  0],
           [ 0,  0,  1]])
    >>> numpy.all(numpy.dot(result[1], matrix_) == result[0])
    True
    '''
    while len(matrix_.shape) < 2:
        matrix_ = numpy.expand_dims(matrix_, axis=0)
    hnf, unimodular_matrix, rank = diophantine.lllhermite(matrix_, m1=1, n1=1)
    unimodular_matrix = unimodular_matrix.astype(INT_TYPE_DEF)
    if not numpy.abs(numpy.abs(numpy.linalg.det(unimodular_matrix)) - 1) < 1e-5:
        raise RuntimeError('Row operation matrix {} has determinant {}, not +-1'.format(unimodular_matrix, numpy.linalg.det(unimodular_matrix)))

    # Rectify any negative entries in the HNF:
    for row_ind, row in enumerate(hnf):
        nonzero_ind = numpy.nonzero(row)[0]
        if len(nonzero_ind):
            if row[nonzero_ind[0]] < 0:
                hnf[row_ind] *= -1
                unimodular_matrix[row_ind] *= -1

    if not is_hnf_row(hnf):
        raise ValueError('{} not able to be put into row HNF. Output is:\n{}'.format(matrix_, hnf))
    assert is_hnf_row(hnf)
    assert numpy.all(numpy.dot(unimodular_matrix, matrix_) == hnf)
    return hnf, unimodular_matrix.astype(INT_TYPE_DEF)

def hnf_col_lll(matrix_):
    ''' Compute the Hermite normal form, ACTS ON THE COLUMNS OF A MATRIX
        Input: integer mxn matrix A, nonzero, at least two rows
        Output: small unimodular matrix B and HNF(A), such that AB=HNF(A)+
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

## Smith normal form

def expand_matrix(matrix_):
    """
        Given a rectangular $n \times m$ integer matrix, return an $n+1 \times m+1$ matrix where the extra row and
        column are 0 except on the first entry which is 1.

        Parameters
        ----------
        matrix_ : sympy.Matrix
            The rectangular matrix to be expanded

        Returns
        -------
        sympy.Matrix
            An $n+1 \times m+1$ matrix

    >>> matrix_ = sympy.diag(*[1, 1, 2])
    >>> expand_matrix(matrix_)
    Matrix([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 2]])

    >>> matrix_ = sympy.Matrix([1])
    >>> expand_matrix(matrix_)
    Matrix([
    [1, 0],
    [0, 1]])

    >>> matrix_ = sympy.Matrix([[0]])
    >>> expand_matrix(matrix_)
    Matrix([
    [1, 0],
    [0, 0]])

    >>> matrix_ = sympy.Matrix([[]])
    >>> expand_matrix(matrix_)
    Matrix([[1]])

    >>> matrix_ = sympy.Matrix([[1, 2, 3]])
    >>> expand_matrix(matrix_)
    Matrix([
    [1, 0, 0, 0],
    [0, 1, 2, 3]])
    >>> expand_matrix(matrix_.T)
    Matrix([
    [1, 0],
    [0, 1],
    [0, 2],
    [0, 3]])
    """
    if len(matrix_.shape) != 2:
        raise ValueError('expand_matrix called for non-square matrix of dimension {}.'.format(len(matrix_.shape)))

    if len(matrix_) == 0:
        return sympy.Matrix([[1]])

    # Add top row
    matrix_ = sympy.Matrix.vstack(sympy.zeros(1, matrix_.shape[1]), matrix_.copy())
    # Add left column
    matrix_ = sympy.Matrix.hstack(sympy.zeros(matrix_.shape[0], 1), matrix_)
    matrix_[0, 0] = 1
    return matrix_

def _swap_ij_rows(matrices, i, j):
    """
        Given an iterable of matrices, swap all of the ith and jth rows.

        Parameters
        ----------
        matrices : iter
            The matrices to be acted on.
        i : int
            Index of first row
        j : int
            Index of second row

        Returns
        -------
        matrices : tuple
            The resulting matrices.

    >>> matrices = [sympy.eye(3) for _ in xrange(2)]
    >>> matrices = _swap_ij_rows(matrices, 0, 1)
    >>> matrices[0]
    Matrix([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1]])
    >>> matrices[1]
    Matrix([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1]])
    """
    matrices = [matrix_.copy() for matrix_ in matrices]
    for matrix_ in matrices:
        matrix_[i, :], matrix_[j, :] = matrix_[j, :].copy(), matrix_[i, :].copy()
    return tuple(matrices)

def _swap_ij_cols(matrices, i, j):
    """
        Given an iterable of matrices, swap all of the ith and jth columns. Acts in place.

        Parameters
        ----------
        matrices : iter
            The matrices to be acted on.
        i : int
            Index of first column
        j : int
            Index of second column

        Returns
        -------
        matrices : tuple
            The resulting matrices.

    >>> matrices = [sympy.eye(3) for _ in xrange(2)]
    >>> matrices = _swap_ij_cols(matrices, 0, 1)
    >>> matrices[0]
    Matrix([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1]])
    >>> matrices[1]
    Matrix([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1]])
    """
    return tuple(matrix_.T for matrix_ in _swap_ij_rows([_matrix.T for _matrix in matrices], i, j))


def is_smf(matrix_):
    """
        Given a rectangular $n \times m$ integer matrix, determine whether it is in Smith normal form or not.

        Parameters
        ----------
        matrix_ : numpy.ndarray
            The rectangular matrix to be decomposed

        Returns
        -------
        bool
            True if in Smith normal form, False otherwise.

    >>> matrix_ = numpy.diag([1, 1, 2])
    >>> is_smf(matrix_)
    True
    >>> matrix_ = numpy.diag([-1, 1, 2])
    >>> is_smf(matrix_)
    False
    >>> matrix_ = numpy.diag([2, 1, 1])
    >>> is_smf(matrix_)
    False
    >>> matrix_ = numpy.diag([1, 2, 0])
    >>> is_smf(matrix_)
    True
    >>> matrix_ = numpy.diag([2, 6, 0])
    >>> is_smf(matrix_)
    True
    >>> matrix_ = numpy.diag([2, 5, 0])
    >>> is_smf(matrix_)
    False
    >>> matrix_ = numpy.diag([0, 1, 1])
    >>> is_smf(matrix_)
    False
    >>> matrix_ = numpy.diag([0])
    >>> is_smf(numpy.diag([0])), is_smf(numpy.diag([1])), is_smf(numpy.diag([])),
    (True, True, True)

    Check a real example
    >>> matrix_ = numpy.array([[2, 4, 4],
    ...                        [-6, 6, 12],
    ...                        [10, -4, -16]])
    >>> is_smf(matrix_)
    False

    >>> matrix_ = numpy.diag([2, 6, 12])
    >>> is_smf(matrix_)
    True

    Check it works for non-square matrices
    >>> matrix_ = numpy.arange(20).reshape((4,5))
    >>> is_smf(matrix_)
    False

    >>> matrix_ = numpy.array([[1, 0], [1, 2]])
    >>> is_smf(matrix_)
    False
    """
    diag = numpy.diagonal(matrix_)
    if diag.size == 0:
        return True

    # Check its a diagonal matrix
    # Make a copy as fill_diagonal acts in place
    _copy = matrix_.copy()
    numpy.fill_diagonal(_copy, 0)
    if numpy.count_nonzero(_copy) > 0:
        return False

    # Check all entries are non-negative
    if numpy.any(diag < 0):
        return False

    current_int = diag[0]
    for i in xrange(1, len(diag)):
        next_int = diag[i]
        # All zeros go at the end
        if (current_int == 0):
            if (next_int != 0):
                return False

        if (current_int > 0):
            if next_int % current_int:
                return False
        current_int = next_int
    return True

def smf(matrix_):
    """
        Given a rectangular $n \times m$ integer matrix, calculate the Smith Normal Form $S$ and multipliers $U$, $V$ such
        that $U matrix_ V = S$.

        Parameters
        ----------
        matrix_ : numpy.ndarray
            The rectangular matrix to be decomposed

        Returns
        -------
        :rtype: (numpy.ndarray, numpy.ndarray, numpy.ndarray)
            The Smith normal form of matrix_, $U$ (the matrix representing the row operations of the decomposition),
            $V$ (the matrix representing the column operations of the decomposition).

    >>> matrix_ = numpy.array([[2, 4, 4],
    ...                        [-6, 6, 12],
    ...                        [10, -4, -16]])
    >>> smf(matrix_)[0]
    array([[ 2,  0,  0],
           [ 0,  6,  0],
           [ 0,  0, 12]])

    >>> matrix_ = numpy.diag([2, 1, 0])
    >>> smf(matrix_)[0]
    array([[1, 0, 0],
           [0, 2, 0],
           [0, 0, 0]])

    >>> matrix_ = numpy.diag([5, 2, 0])
    >>> smf(matrix_)[0]
    array([[ 1,  0,  0],
           [ 0, 10,  0],
           [ 0,  0,  0]])
    """
    matrix_ = sympy.Matrix(matrix_)
    transformed = matrix_.copy()

    if is_smf(sympy.matrix2numpy(transformed)):
        row_actions = sympy.eye(matrix_.shape[0])
        col_actions = sympy.eye(matrix_.shape[1])
        return transformed, row_actions, col_actions

    # First make a pivot in the top left corner
    transformed, row_actions = hnf_row(matrix_=sympy.matrix2numpy(transformed))
    transformed, col_actions = hnf_col(matrix_=transformed)
    transformed, row_actions, col_actions = map(sympy.Matrix, [transformed, row_actions, col_actions])

    # Now put the lower block into Smith normal form
    block, _row_actions, _col_actions = smf(transformed[1:, 1:])
    block, _row_actions, _col_actions = map(sympy.Matrix, [block, _row_actions, _col_actions])

    # Compose the row and column actions found
    row_actions = row_actions * expand_matrix(_row_actions)
    col_actions = expand_matrix(_col_actions) * col_actions
    transformed[1:, 1:] = block
    assert row_actions * matrix_ * col_actions == transformed

    # Now make sure the pivot is in the right place
    for i in xrange(min(*transformed.shape) - 1):
        if transformed[i+1,i+1] == 0:
            break
        # Move the pivot entry down if it's 0
        if transformed[i,i] == 0:
            transformed, row_actions = _swap_ij_rows([transformed, row_actions], i, i+1)
            transformed, col_actions = _swap_ij_cols([transformed, col_actions], i, i+1)

    assert row_actions * matrix_ *  col_actions == transformed
    assert transformed[1:, 0].is_zero
    assert transformed[0, 1:].is_zero

    # Enforce diagonal entries to divide the next diagonal entry
    for i in xrange(min(*transformed.shape) - 1):
        assert row_actions * matrix_ * col_actions == transformed
        if transformed[i+1,i+1] == 0:
            break

        # If we don't have division, ensure we do
        if transformed[i+1,i+1] % transformed[i,i]:
            # If we have division in the wrong order, just swap them around
            if not transformed[i,i] % transformed[i+1,i+1]:
                transformed, row_actions = _swap_ij_rows([transformed, row_actions], i, i + 1)
                transformed, col_actions = _swap_ij_cols([transformed, col_actions], i, i + 1)
                # Reset the counter
                i = 0
                continue

            # Add one of column i+1 to column i, and perform a row reduction to get the hcf
            transformed[:, i] += transformed[:, i+1]
            col_actions[:, i] += col_actions[:, i+1]
            assert row_actions * matrix_ * col_actions == transformed

            transformed, _row_actions = hnf_row(sympy.matrix2numpy(transformed))
            transformed, _row_actions = map(sympy.Matrix, [transformed, _row_actions])
            row_actions = _row_actions * row_actions
            assert row_actions * matrix_ * col_actions == transformed

            # Now subtract the right multiple of column i from column i+1
            assert (transformed[i, i+1] % transformed[i, i]) == 0
            multiple = int(transformed[i, i+1] / transformed[i, i])
            transformed[:, i+1] -= multiple * transformed[:, i]
            col_actions[:, i+1] -= multiple * col_actions[:, i]
            assert row_actions * matrix_ * col_actions == transformed

            # Reset the loop, so we check from the beginning of the diagonal.
            i = 0

    # _prod = row_actions * matrix_ *  col_actions
    assert row_actions * matrix_ *  col_actions == transformed

    assert is_smf(sympy.matrix2numpy(transformed))
    return tuple(map(lambda x: sympy.matrix2numpy(x, INT_TYPE_DEF), [transformed, row_actions, col_actions]))

if __name__ == '__main__':
    import doctest
    doctest.testmod()