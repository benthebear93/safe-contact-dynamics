import numpy as np
import scipy.linalg
from scipy.linalg import get_lapack_funcs
import copy
from .linearized_solver import (
    RZLin,
)
from scipy import linalg


class LinearSolver:
    """Abstract base class for linear solvers."""

    pass


class EmptySolver(LinearSolver):
    def __init__(self, F):
        self.F = F


def empty_solver(A):
    return EmptySolver(A)


class LUSolver(LinearSolver):
    """
    LU-based direct linear solver.

    Doctrine:
    This solver is designed to efficiently solve linear systems of the form Ax = b,
    particularly in iterative optimization or simulation routines where A may remain
    constant or change slowly across iterations. By caching the LU decomposition
    of A, this class allows repeated solves with minimal overhead.

    Methods:
    - factorize(A): Computes the LU factorization of A and stores it.
    - solve_vector(x, A, b, reg=0.0, fact=True): Solves Ax = b for vector b.
    - solve_matrix(x, A, b, reg=0.0, fact=True): Solves AX = B for matrix B, column-wise.

    Parameters:
    - reg (float): Regularization value (currently unused, reserved for extension).
    - fact (bool): If True, recomputes LU factorization of A; otherwise reuses cached.
    """

    def __init__(self, A):
        self.A = copy.deepcopy(A)  # .copy()
        if isinstance(self.A, RZLin):
            self.lu = None
            self.piv = None
        else:
            n = A.shape[0]
            self.ipiv = np.zeros(n, dtype=np.int32)
        self.info = np.zeros(1, dtype=np.int32)
        self.lda = 0

    def factorize(self, A):
        self.A = copy.deepcopy(A)
        self.lu, self.piv = scipy.linalg.lu_factor(self.A)

    def new_factorize(self, A):
        # if scipy.sparse.issparse(A):
        #     A = A.toarray()
        getrf = get_lapack_funcs("getrf", dtype=np.float64)
        self.lu, self.piv, info = getrf(A, overwrite_a=False)

    def solve_vector(self, x, A, b, reg=0.0, fact=True):
        if fact:
            self.new_factorize(A)
        x[:] = b.flatten()
        x[:] = scipy.linalg.lu_solve((self.lu, self.piv), x).ravel()
        return x[:]

    def solve_matrix(self, x, A, b, reg=0.0, fact=True):
        if fact:
            self.factorize(A)
        x = scipy.linalg.lu_solve((self.lu, self.piv), b)
        return x


def getrf(A):
    """
    Perform LU factorization on matrix A using LAPACK's dgetrf equivalent.
    Returns the factored matrix A and pivot indices ipiv.
    """
    # Check for 1-based indexing (Python is 0-based, so no adjustment needed)
    if not np.all(A.flags["C_CONTIGUOUS"]):
        A = np.ascontiguousarray(A)
    m, n = A.shape
    # Ensure stride is compatible
    lda = max(1, A.strides[1] // A.itemsize)
    # Perform LU factorization using SciPy's lu_factor (wrapper for dgetrf)
    lu, piv = linalg.lu_factor(A)
    # piv is 0-based in SciPy, convert to 1-based to match Julia/LAPACK
    ipiv = piv + 1
    return lu, ipiv


def factorize_temp(s, A):
    """
    Prepare the LUSolver by factorizing matrix A.
    """
    # Compute condition number (optional, mimicking Julia's cond)
    cond_A = np.linalg.cond(A)
    # print("Condition number of A:", cond_A)

    # Reset solver attributes
    s.A.fill(0.0)
    s.ipiv.fill(0)
    s.lda = 0
    # Copy A into solver's matrix
    s.A[:] = A
    # Perform LU factorization
    s.A, s.ipiv = getrf(s.A)
    s.lda = max(1, s.A.strides[1] // s.A.itemsize)


def linear_solve_vector(s, x, A, b, reg=0.0, fact=True):
    """
    Solve the linear system Ax = b using LU factorization.
    """
    if fact:
        factorize_temp(s, A)
    x[:] = b

    # Solve using SciPy's lu_solve (wrapper for dgetrs)
    x[:] = linalg.lu_solve(
        (s.A, s.ipiv - 1), x, trans=0
    )  # ipiv - 1 to convert back to 0-based # FIXME : problem caused by rewriting the julia code into python


def linear_solve_matrix(s, x, A, b, reg=0.0, fact=True):
    """
    Solves linear system Ax = b for matrix x (multiple right-hand sides).
    """
    x.fill(0.0)
    n, m = x.shape
    r_idx = slice(0, n)
    if fact:
        factorize_temp(s, A)
    x[:] = b
    dgetrs = get_lapack_funcs("getrs", [s.A])[0]
    for j in range(m):
        xv = x[r_idx, j]
        _, info = dgetrs(s.A, s.ipiv, xv, trans="N", overwrite_b=True)
        if info != 0:
            raise RuntimeError(f"GETRS failed with info={info}")
