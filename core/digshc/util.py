import numpy as np

def mirror_upper(arr):
    """
    Create a symmetric matrix by
    using values above the diagonal.

    e.g::

        a = array([[1, 2],
                   [3, 4]])
        mirror_upper(a)
        > array([[1, 2],
                 [2, 4]])
    """
    arr = np.triu(arr)
    return arr + arr.T - np.diag(arr.diagonal())


def triu_index(i, j):
    """
    Returns an index for the (i,j) or (j,i) position in a matrix,
    ensuring that it remains it the upper triangle. The (i,j) and (j,i)
    values are equivalent in a symmetric matrix.
    """
    row, col = (i, j) if i < j else (j, i)
    return row, col
