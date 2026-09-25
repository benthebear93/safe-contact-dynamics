import numpy as np


def triu_perm(k, j):
    ind = int((j - 1) * j / 2) + k
    return ind - 1  # Adjust for 0-based indexing


class SDMGSSolver:
    """
    Static Dense Modified Gram-Schmidt
    """

    def __init__(self, n, T=float):
        self.n = n
        self.as_ = [np.zeros(n, dtype=T) for _ in range(n)]
        self.qs = [np.zeros(n, dtype=T) for _ in range(n)]
        self.rs = np.zeros(int((n + 1) * n / 2), dtype=T)
        self.xv = np.zeros(n, dtype=T)
        self.xs = np.zeros(n, dtype=T)


    def factorize(self, A):
        """
        Gram-Schmidt algorithm applied to A.
        """
        for j in range(self.n):
            self.as_[j] = A[:, j].copy()
        self._factorize()
        return None


    def _factorize(self):
        """
        Gram-Schmidt algorithm applied to gs_solver.a.
        """
        # Unpack
        as_ = self.as_
        qs = self.qs
        rs = self.rs
        off = 0
        for j in range(self.n):
            # qi
            qs[j] = as_[j].copy()
            for k in range(j):
                # rk
                rs[off] = np.dot(qs[j], qs[k])
                # qu
                qs[j] -= qs[k] * rs[off]
                off += 1
            # re
            rs[off] = np.linalg.norm(qs[j])
            qs[j] /= rs[off]
            off += 1
        return None

    def qr_solve(self, b):
        """
        QR Back substitution after Gram-Schmidt.
        """
        qs = self.qs
        rs = self.rs
        xv = self.xv
        for j in range(self.n):
            xv[j] = qs[j].T @ b
        for j in range(self.n - 1, -1, -1):
            for k in range(j + 1, self.n):
                xv[j] -= rs[triu_perm(j + 1, k + 1)] * xv[k]
            xv[j] /= rs[triu_perm(j + 1, j + 1)]
        self.xs = xv
        return None


"""
    QR solver
"""
