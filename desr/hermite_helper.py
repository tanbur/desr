
##### Richard Tanburn
import sympy
import itertools

import diophantine

def get_pivot_row_indices(matrix_):
    ''' Return the pivot indices of the matrix

        Args:
            matrix_ (sympy.Matrix): The matrix in question.

        Returns:
            indices (list): A list of indices, where the pivot entry of row i is [i, indices[i]]

        >>> matrix_ = sympy.eye(3)
        >>> get_pivot_row_indices(matrix_)
        [0, 1, 2]

        >>> matrix_ = sympy.Matrix([[1, 0, 0],[0, 0, 1]])
        >>> get_pivot_row_indices(matrix_)
        [0, 2]
    '''
    indices = []
    for row_ind in xrange(matrix_.shape[0]):
        row = matrix_[row_ind, :]
        for ind, col in enumerate(row):
            if col != 0:
                indices.append(ind)
                break
        # If we haven't found an index. throw an error, as we're assuming the rows are non-zero
        if len(indices) != row_ind + 1:
            raise ValueError('Zero row in {}'.format(matrix_))
    return indices

def is_hnf_col(matrix_):
    '''
    Decide whether a matrix is in row Hermite normal form, when acting on rows.

    Args:
        matrix_ (sympy.Matrix): The matrix in question.

    Returns:
        bool
    '''
    return is_hnf_row(matrix_.T)

def is_hnf_row(matrix_):
    ''' Decide whether a matrix is in Hermite normal form, when acting on rows.
        This is according to the Havas Majewski Matthews definition.

        Args:
            matrix_ (sympy.Matrix): The matrix in question.

        Returns:
            bool

        >>> is_hnf_row(sympy.eye(4))
        True
        >>> is_hnf_row(sympy.ones(2))
        False
        >>> is_hnf_row(sympy.Matrix([[1, 2, 0], [0, 1, 0]]))
        False
        >>> is_hnf_row(sympy.Matrix([[1, 0, 0], [0, 2, 1]]))
        True
        >>> is_hnf_row(sympy.Matrix([[1, 0, 0], [0, -2, 1]]))
        False
    '''
    m, n = matrix_.shape

    # The first r rows are non-zero. Determine r and check rows r through m are 0.
    row_is_zero = [all([i == 0 for i in row]) for row in matrix_.tolist()]
    r = m - sum(map(int, row_is_zero))
    for i in xrange(r, m):
        if not row_is_zero[i]:
            return False

    # If b_{i,j_i} are the indices of the first non-zero entry of row i then j_i are strictly ascending
    indices = get_pivot_row_indices(matrix_[:r, :])
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
    '''
    Compute the Hermite normal form, acts on the ROWS of a matrix.
    The Havas, Majewski, Matthews LLL method is used.
    We usually take :math:`\\alpha = \\frac{m_1}{n_1}`, with :math:`(m_1,n_1)=(1,1)` to get best results.

    Args:
        matrix_ (sympy.Matrix): Integer mxn matrix, nonzero, at least two rows

    Returns:
        tuple:
            hnf (sympy.Matrix): The row Hermite normal form of matrix\\_.

            unimod (sympy.Matrix): Unimodular matrix representing the row actions.

    :rtype: (sympy.Matrix, sympy.Matrix)

    >>> matrix_ = sympy.Matrix([[2, 0],
    ...                         [3, 3],
    ...                         [0, 0]])
    >>> result = hnf_row_lll(matrix_)
    >>> result[0]
    Matrix([
    [1, 3],
    [0, 6],
    [0, 0]])
    >>> result[1]
    Matrix([
    [-1, 1, 0],
    [-3, 2, 0],
    [ 0, 0, 1]])
    >>> result[1] * matrix_ == result[0]
    True

    >>> hnf_row_lll(sympy.Matrix([[0, -2, 0]]))
    (Matrix([[0, 2, 0]]), Matrix([[-1]]))

    >>> matrix_ = sympy.Matrix([[0, 1, 0, 1], [-1, 0, 1, 0]])
    >>> hnf_row_lll(matrix_)[0]
    Matrix([
    [1, 0, -1, 0],
    [0, 1,  0, 1]])
    '''
    assert len(matrix_.shape) == 2

    # For some reason diophantine barfs if we only have one row. Work around this.
    if matrix_.shape[0] == 1:
        hnf = matrix_.copy()
        unimodular_matrix = sympy.Matrix([[1]])
        rank = 1 - int(hnf.is_zero)
    else:
        hnf, unimodular_matrix, rank = diophantine.lllhermite(matrix_, m1=1, n1=1)

    if not abs(unimodular_matrix.det()) == 1:
        raise RuntimeError('Row operation matrix {} has determinant {}, not +-1'.format(unimodular_matrix, unimodular_matrix.det()))

    # Rectify any negative entries in the HNF:
    for row_ind, row in enumerate(hnf.tolist()):
        nonzero_ind = [_ind for _ind, _val in enumerate(row) if _val != 0]
        if len(nonzero_ind):
            if row[nonzero_ind[0]] < 0:
                hnf[row_ind, :] *= -1
                unimodular_matrix[row_ind, :] *= -1

    if not is_hnf_row(hnf):
        raise ValueError('{} not able to be put into row HNF. Output is:\n{}'.format(matrix_, hnf))
    assert is_hnf_row(hnf)
    assert (unimodular_matrix * matrix_) == hnf
    return hnf, unimodular_matrix

def hnf_col_lll(matrix_):
    '''
    Compute the Hermite normal form, acts on the COLUMNS of a matrix.
    The Havas, Majewski, Matthews LLL method is used.
    We usually take :math:`\\alpha = \\frac{m_1}{n_1}`, with :math:`(m_1,n_1)=(1,1)` to get best results.

    Args:
        matrix_ (sympy.Matrix): Integer mxn matrix, nonzero, at least two rows

    Returns:
        tuple:
            hnf (sympy.Matrix): The column Hermite normal form of matrix\\_.
            unimod (sympy.Matrix): Unimodular matrix representing the column actions.

    :rtype: (sympy.Matrix, sympy.Matrix)

    >>> A = sympy.Matrix([[8, 2, 15, 9, 11],
    ...                   [6, 0, 6, 2, 3]])
    >>> h, v = hnf_col(A)
    >>> h
    Matrix([
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0]])
    >>> v
    Matrix([
    [-1,  0,  0, -1, -1],
    [-3, -1,  6, -1,  0],
    [ 1,  0,  1,  1,  2],
    [ 0, -1, -3, -3,  0],
    [ 0,  1,  0,  2, -2]])
     >>> A * v == h
     True
    '''
    hnf, unimod = hnf_row_lll(matrix_.T)
    return hnf.T, unimod.T


## Normal Hermite multiplier from Hubert-Labahn
def is_normal_hermite_multiplier(hermite_multiplier, matrix_):
    '''
    Determine whether hermite_multiplier is the normal Hermite multiplier of matrix.
    This is the COLUMN version.

    Args:
        hermite_multiplier (sympy.Matrix): Candidate column normal Hermite multiplier
        matrix_ (sympy.Matrix): Matrix

    Returns:
        bool: matrix\\_ * hermite_multiplier is in Hermite normal form and hermite_multiplier is in normal form.
    '''
    if matrix_.rank() != matrix_.shape[0]:
        raise ValueError('Matrix must have full row rank')

    r, n = matrix_.shape
    if hermite_multiplier.shape != (n, n):
        raise ValueError("Matrix dimensions don't match")

    # Check matrix . hermite_multiplier = [H 0]
    prod = matrix_ * hermite_multiplier
    hermite_form, residue = prod[:, :r], prod[:, r:]
    # Condition a)
    if not residue.is_zero:
        return False
    if not is_hnf_col(hermite_form):
        return False
    # Condition b)
    herm_mul_i, herm_mul_n = hermite_multiplier[:, :r], hermite_multiplier[:, r:]
    if not is_hnf_col(herm_mul_n):
        return False
    # Condition c)
    for i in xrange(r):
        pivot_val = max(herm_mul_n[i, :])
        if any(herm_mul_i.row(i).applyfunc(lambda x: abs(x) >= pivot_val)):
            return False

    return True

def normal_hnf_col(matrix_):
    '''
    Return the column HNF and the unique normal Hermite multiplier.

    Args:
        matrix_ (sympy.Matrix): Input matrix.

    Returns:
        tuple: Tuple containing:
            hermite_normal_form (sympy.Matrix): The column Hermite normal form of matrix\\_.
            normal_multiplier (sympy.Matrix): The normal Hermite multiplier.

    >>> A = sympy.Matrix([[8, 2, 15, 9, 11],
    ...                   [6, 0, 6, 2, 3]])
    >>> h, v = normal_hnf_col(A)
    >>> h
    Matrix([
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0]])
    >>> v
    Matrix([
    [  0,  0,   1,  0,   0],
    [  0,  0,   0,  1,   0],
    [  2,  1,   3,  1,   5],
    [  9,  2,  21,  3,  21],
    [-10, -3, -22, -4, -24]])
    >>> (A * v) == h
    True
    '''
    r, n = matrix_.shape
    out, _ = hnf_col_lll(sympy.Matrix.vstack(matrix_, sympy.eye(n)))
    hermite_normal_form, normal_multiplier = out[:r, :], out[r:, :]
    assert (matrix_ * normal_multiplier) == hermite_normal_form
    return hermite_normal_form, normal_multiplier

def normal_hnf_row(matrix_):
    '''
    Return the row HNF and the unique normal Hermite multiplier.

    Args:
        matrix_ (sympy.Matrix): Input matrix.

    Returns:
        tuple: Tuple containing:

            hermite_normal_form (sympy.Matrix): The row Hermite normal form of matrix\\_.
            normal_multiplier (sympy.Matrix): The normal Hermite multiplier.
    '''
    hermite_normal_form, normal_multiplier = normal_hnf_col(matrix_=matrix_.T)
    return hermite_normal_form.T, normal_multiplier.T

## Set defaults
hnf_row = hnf_row_lll
"""
Default function for calculating row Hermite normal forms.

Args:
    matrix_ (sympy.Matrix): Input matrix.

Returns:
    tuple:
        hermite_normal_form (sympy.Matrix): The column Hermite normal form of matrix\\_.

        normal_multiplier (sympy.Matrix): The normal Hermite multiplier.

:rtype: (sympy.Matrix, sympy.Matrix)
"""

hnf_col = hnf_col_lll
"""
Default function for calculating column Hermite normal forms.

Args:
    matrix_ (sympy.Matrix): Input matrix.

Returns:
    tuple: Tuple containing:
        hermite_normal_form (sympy.Matrix): The column Hermite normal form of matrix\\_.

        normal_multiplier (sympy.Matrix): The normal Hermite multiplier.
:rtype: (sympy.Matrix, sympy.Matrix)
"""

## Smith normal form

def expand_matrix(matrix_):
    """
    Given a rectangular :math:`n \\times m` integer matrix, return an :math:`(n+1) \\times (m+1)` matrix where the extra row and
    column are 0 except on the first entry which is 1.

    Parameters
    ----------
    matrix_ : sympy.Matrix
        The rectangular matrix to be expanded

    Returns
    -------
    sympy.Matrix
        An :math:`(n+1) \\times (m+1)` matrix


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
        matrices: tuple
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
        matrix_.row_swap(i, j)
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

def element_wise_lt(matrix_, other):
    """
    Given a rectangular :math:`n \\times m` integer matrix, return a matrix of bools with (i, j)th entry equal to
    matrix_[i,j] < other or other[i,j], depending on the type of other.

    Parameters
    ----------
    matrix_ : sympy.Matrix
        The input matrix
    other : sympy.Matrix, int
        Value(s) to compare it to.

    Returns
    -------
    sympy.Matrix
        :math:`n \\times m` Boolean matrix


    >>> x = sympy.eye(2) * 3
    >>> element_wise_lt(x, 2)
    Matrix([
    [False,  True],
    [ True, False]])
    >>> element_wise_lt(x, sympy.Matrix([[4, -1], [1, 1]]))
    Matrix([
    [True, False],
    [True, False]])
    """
    if isinstance(other, sympy.Matrix):
        if other.shape != matrix_.shape:
            raise ValueError('Incompatible shapes: {} != {}'.format(matrix_.shape, other.shape))
        return sympy.Matrix(matrix_.shape[0], matrix_.shape[1], lambda i, j: matrix_[i, j] < other[i, j])
    else:
        return sympy.Matrix(matrix_.shape[0], matrix_.shape[1], lambda i, j: matrix_[i, j] < other)

def is_smf(matrix_):
    """
    Given a rectangular :math:`n \\times m` integer matrix, determine whether it is in Smith normal form or not.

    Parameters
    ----------
    matrix_ : sympy.Matrix
        The rectangular matrix to be decomposed

    Returns
    -------
    bool
        True if in Smith normal form, False otherwise.


    >>> matrix_ = sympy.diag(1, 1, 2)
    >>> is_smf(matrix_)
    True
    >>> matrix_ = sympy.diag(-1, 1, 2)
    >>> is_smf(matrix_)
    False
    >>> matrix_ = sympy.diag(2, 1, 1)
    >>> is_smf(matrix_)
    False
    >>> matrix_ = sympy.diag(1, 2, 0)
    >>> is_smf(matrix_)
    True
    >>> matrix_ = sympy.diag(2, 6, 0)
    >>> is_smf(matrix_)
    True
    >>> matrix_ = sympy.diag(2, 5, 0)
    >>> is_smf(matrix_)
    False
    >>> matrix_ = sympy.diag(0, 1, 1)
    >>> is_smf(matrix_)
    False
    >>> matrix_ = sympy.diag(0)
    >>> is_smf(sympy.diag(0)), is_smf(sympy.diag(1)), is_smf(sympy.Matrix()),
    (True, True, True)

    Check a real example:

    >>> matrix_ = sympy.Matrix([[2, 4, 4],
    ...                         [-6, 6, 12],
    ...                         [10, -4, -16]])
    >>> is_smf(matrix_)
    False

    >>> matrix_ = sympy.diag(2, 6, 12)
    >>> is_smf(matrix_)
    True

    Check it works for non-square matrices:

    >>> matrix_ = sympy.Matrix(4, 5, range(20))
    >>> is_smf(matrix_)
    False

    >>> matrix_ = sympy.Matrix([[1, 0], [1, 2]])
    >>> is_smf(matrix_)
    False
    """
    if len(matrix_) == 0:
        return True

    # Check its a diagonal matrix
    if not matrix_.is_diagonal():
        return False

    # Check all entries are non-negative
    if any(element_wise_lt(matrix_, 0)):
        return False

    current_int = matrix_[0, 0]
    for i in xrange(1, min(matrix_.shape)):
        next_int = matrix_[i, i]
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
    Given a rectangular :math:`n \\times m` integer matrix, calculate the Smith Normal Form :math:`S` and multipliers
    :math:`U`, :math:`V` such that :math:`U \\textrm{matrix_} V = S`.

    Parameters
    ----------
    matrix_ : sympy.Matrix
        The rectangular matrix to be decomposed.

    Returns
    -------
    S : sympy.Matrix
        The Smith normal form of matrix\\_.
    U : sympy.Matrix
        :math:`U` (the matrix representing the row operations of the decomposition).
    V : sympy.Matrix
        :math:`V` (the matrix representing the column operations of the decomposition).


    :rtype: (sympy.Matrix, sympy.Matrix, sympy.Matrix)


    >>> matrix_ = sympy.Matrix([[2, 4, 4],
    ...                         [-6, 6, 12],
    ...                         [10, -4, -16]])
    >>> smf(matrix_)[0]
    Matrix([
    [2, 0,  0],
    [0, 6,  0],
    [0, 0, 12]])

    >>> matrix_ = sympy.diag(2, 1, 0)
    >>> smf(matrix_)[0]
    Matrix([
    [1, 0, 0],
    [0, 2, 0],
    [0, 0, 0]])

    >>> matrix_ = sympy.diag(5, 2, 0)
    >>> smf(matrix_)[0]
    Matrix([
    [1,  0, 0],
    [0, 10, 0],
    [0,  0, 0]])
    """
    matrix_ = matrix_.copy()
    transformed = matrix_.copy()

    if is_smf(transformed):
        row_actions = sympy.eye(matrix_.shape[0])
        col_actions = sympy.eye(matrix_.shape[1])
        return transformed, row_actions, col_actions

    # First make a pivot in the top left corner
    transformed, row_actions = hnf_row(matrix_=transformed)
    transformed, col_actions = hnf_col(matrix_=transformed)

    # Now put the lower block into Smith normal form
    block, _row_actions, _col_actions = smf(transformed[1:, 1:])

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

            transformed, _row_actions = hnf_row(transformed)
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

    assert is_smf(transformed)
    return transformed, row_actions, col_actions



if __name__ == '__main__':
    import doctest
    doctest.testmod()