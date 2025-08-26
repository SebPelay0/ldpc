import numpy as np
from scipy.sparse import csr_matrix
###FROM PYLDPC LIBRARY### 

def gaussjordan(X, change=0):
    """Compute the binary row reduced echelon form of X.

    Parameters
    ----------
    X: array (m, n)
    change : boolean (default, False). If True returns the inverse transform

    Returns
    -------
    if `change` == 'True':
        A: array (m, n). row reduced form of X.
        P: tranformations applied to the identity
    else:
        A: array (m, n). row reduced form of X.

    """
    A = np.copy(X)
    m, n = A.shape

    if change:
        P = np.identity(m).astype(int)

    pivot_old = -1
    for j in range(n):
        filtre_down = A[pivot_old+1:m, j]
        pivot = np.argmax(filtre_down)+pivot_old+1

        if A[pivot, j]:
            pivot_old += 1
            if pivot_old != pivot:
                aux = np.copy(A[pivot, :])
                A[pivot, :] = A[pivot_old, :]
                A[pivot_old, :] = aux
                if change:
                    aux = np.copy(P[pivot, :])
                    P[pivot, :] = P[pivot_old, :]
                    P[pivot_old, :] = aux

            for i in range(m):
                if i != pivot_old and A[i, j]:
                    if change:
                        P[i, :] = abs(P[i, :]-P[pivot_old, :])
                    A[i, :] = abs(A[i, :]-A[pivot_old, :])

        if pivot_old == m-1:
            break

    if change:
        return A, P
    return A

def binaryproduct(X, Y):
    """Compute a matrix-matrix / vector product in Z/2Z."""
    A = X.dot(Y)
    try:
        A = A.toarray()
    except AttributeError:
        pass
    return A % 2


def coding_matrix(H, sparse=True):
    """Return the generating coding matrix G given the LDPC matrix H.

    Parameters
    ----------
    H: array (n_equations, n_code). Parity check matrix of an LDPC code with
        code length `n_code` and `n_equations` number of equations.
    sparse: (boolean, default True): if `True`, scipy.sparse format is used
        to speed up computation.

    Returns
    -------
    G.T: array (n_bits, n_code). Transposed coding matrix.

    """
    # if type(H) == csr_matrix:
    #     H = H.toarray()
    n_equations, n_code = H.shape

    # DOUBLE GAUSS-JORDAN:

    Href_colonnes, tQ = gaussjordan(H.T, 1)

    Href_diag = gaussjordan(np.transpose(Href_colonnes))

    Q = tQ.T

    n_bits = n_code - Href_diag.sum()

    Y = np.zeros(shape=(n_code, n_bits)).astype(int)
    Y[n_code - n_bits:, :] = np.identity(n_bits)

    if sparse:
        Q = csr_matrix(Q)
        Y = csr_matrix(Y)

    tG = binaryproduct(Q, Y)

    return tG