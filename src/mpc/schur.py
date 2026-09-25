import numpy as np
from .qr import SDMGSSolver

class Schur:
    def __init__(self, A, B, C, D, Ai, CAi, CAiB, gs_data, u, v, x, y):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.Ai = Ai
        self.CAi = CAi
        self.CAiB = CAiB
        self.gs_data = gs_data
        self.u = u
        self.v = v
        self.x = x
        self.y = y


def schur(M, n=0, m=None):
    if m is None:
        m = M.shape[0] - n
    # assert M.shape == (n + m, n + m)
    A = M[:n, :n]
    B = M[:n, n : n + m]
    C = M[n : n + m, :n]
    D = M[n : n + m, n : n + m]

    Ai = np.linalg.inv(A)
    CAi = C @ Ai
    CAiB = C @ Ai @ B
    # print("m ", m, "n", n)
    gs_data = SDMGSSolver(m)
    # print("D-CAiB \n", D - CAiB)
    gs_data.factorize(D - CAiB)
    u = np.zeros(n)
    v = np.zeros(m)
    x = np.zeros(n)
    y = np.zeros(m)
    return Schur(A, B, C, D, Ai, CAi, CAiB, gs_data, u, v, x, y)


def schur_factorize(S, D):
    CAiB = S.CAiB
    gs_data = S.gs_data
    gs_data.factorize(D - CAiB)


def schur_solve(S, u, v):
    # Check the right-hand side
    rhs = S.CAi @ u - v

    # Check if RHS contains NaN or inf
    if np.any(np.isnan(rhs)) or np.any(np.isinf(rhs)):
        print("ERROR: RHS contains NaN or inf!")
        return

    S.u = u
    S.v = v
    us = S.u
    vs = S.v

    # Check CAi @ us - vs before QR solve
    rhs_check = S.CAi @ us - vs

    # Try to solve and catch the error
    try:
        S.gs_data.qr_solve(rhs_check)
    except Exception as e:
        print(f"QR solve failed: {e}")
        return

    temp = S.gs_data.xs

    if np.any(np.isnan(temp)):
        print("ERROR: temp contains NaN!")
        return

    S.x = S.Ai @ (us + S.B @ temp)
    S.y = -temp
