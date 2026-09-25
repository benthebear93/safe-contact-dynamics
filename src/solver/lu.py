import numpy as np
from scipy.linalg.lapack import dgetrf, dgetrs


class LinearSolver:
    pass


class EmptySolver(LinearSolver):
    def __init__(self, F):
        self.F = F


def empty_solver(A):
    return EmptySolver(A)


class LUSolver(LinearSolver):
    def __init__(self, A=None):
        self.A = None
        self.ipiv = None
        self.info = 0
        self.shape = None

        if A is not None:
            self.factorize(A)

    def factorize(self, A, reg: float = 0.0):
        # Ensure Fortran-contiguous 2D array
        A_f = np.asfortranarray(A, dtype=float)
        if A_f.ndim != 2 or A_f.shape[0] != A_f.shape[1]:
            raise ValueError("A must be a square 2D matrix")

        n = A_f.shape[0]

        if self.A is None or self.A.shape != (n, n):
            self.A = np.zeros((n, n), order="F", dtype=float)
        else:
            self.A.fill(0.0)

        np.copyto(self.A, A_f)
        lu, ipiv, info = dgetrf(self.A, overwrite_a=True)

        self.A = lu
        self.ipiv = ipiv
        self.info = int(info)
        self.shape = self.A.shape

        if self.info < 0:
            raise RuntimeError(f"dgetrf: illegal value in argument {-self.info}")

    def solve(self, A, b, x=None, reg: float = 0.0, fact: bool = True):
        """
        Solve A x = b using existing LU or refactorizing A.
        reg is ignored (to match Julia).
        - If b is 1D, we solve for a vector.
        - If b is 2D, we solve for multiple RHS.
        """
        if fact or self.A is None:
            self.factorize(A, reg=0.0)

        b_arr = np.array(b, dtype=float, copy=True, order="F")

        sol, info = dgetrs(self.A, self.ipiv, b_arr, trans=0, overwrite_b=True)
        if info < 0:
            raise RuntimeError(f"dgetrs: illegal value in argument {-info}")

        if x is None:
            return sol

        # Copy result into x (x can be vector or matrix)
        x[...] = sol
        return x


def lu_solver(A):
    return LUSolver(A)


def linear_solve(solver: LUSolver, x, A, b, reg: float = 0.0, fact: bool = True):
    # Support either plain numpy arrays or objects exposing an `M` matrix (e.g., RZLin).
    A_mat = A.M if hasattr(A, "M") else A
    return solver.solve(A_mat, b, x=x, reg=reg, fact=fact)
