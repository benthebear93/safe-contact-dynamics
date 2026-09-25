from autograd import numpy as anp
from autograd import jacobian

from src.helper import *
from src.solver.indices import IndicesOptimization, IndicesTheta, IndicesZ
from src.simulator.dynamics import *


class Particle:
    def __init__(self, nq, nc, nu, nw, nb, mb, mu, gravity):
        self.nq = nq
        self.nc = nc
        self.nu = nu
        self.nw = nw
        self.lc_fric = 4
        self.nb = self.lc_fric * self.nc

        self.mb = mb
        self.mu = mu
        self.friction_coeff = anp.array([self.mu])
        self.nf = len(self.friction_coeff)
        self.gravity = gravity
        self.r = 0.1
        self.bottom = anp.array([0, 0, -self.r])

        self.idx_q1 = anp.arange(0, nq)
        self.idx_q2 = anp.arange(nq, 2 * nq)
        self.idx_u = anp.arange(2 * nq, 2 * nq + nu)
        self.idx_w = anp.arange(2 * nq + nu, 2 * nq + nu + nw)
        self.idx_f = anp.arange(2 * nq + nu + nw, 2 * nq + nu + nw + self.nf)
        self.idx_h = anp.arange(
            2 * nq + nu + nw + self.nf, 2 * nq + nu + nw + self.nf + 1
        )

    def M_func(self, q):
        M_array = anp.diag(
            [
                self.mb,
                self.mb,
                self.mb,
            ]
        )
        return M_array

    def C_func(self, q, qdot):
        return anp.array([0.0, 0.0, self.gravity * self.mb])

    def kinematics_body(self, q):
        return anp.array([q[0], q[1], q[2]])

    def signed_distance(self, q):
        p = self.kinematics_body(q)
        phi = p[2] - self.r
        return anp.array([0, 0, phi])

    def input_jacobian(self, q):
        return anp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def contact_jacobian(self, q):
        # Flat-ground point contact: tangential and normal forces directly map to the
        # translational coordinates, so keep the full identity instead of the signed
        # distance gradient (which would zero-out tangential friction).
        return anp.eye(3)

    def nominal_configuration(self):
        return anp.array([0.0, 0.0, 2 * self.r + 1e-8])

    def friction_coefficients(self):
        return self.friction_coeff

    def p_function(self, q):
        map_mat = anp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])

        def p(x):
            pos = x[:3]
            bottom_world = pos + self.bottom

            xy = bottom_world[:2]
            return map_mat @ xy

        p_jacobian = jacobian(p)(q)
        return p_jacobian

    def num_var(self):
        nq = self.nq
        gamma = self.nc
        sgamma = self.nc
        psi = self.nc
        beta = self.nc * self.lc_fric
        spsi = self.nc
        sbeta = self.nc * self.lc_fric
        nvar = nq + gamma + sgamma + psi + beta + spsi + sbeta
        return nvar

    def indices_z(self):
        idx = 0
        q = anp.arange(idx, idx + self.nq)
        idx += self.nq

        gamma = anp.arange(idx, idx + self.nc)
        idx += self.nc

        b = anp.arange(idx, idx + self.nb)
        idx += self.nb

        psi = anp.arange(idx, idx + self.nc)
        idx += self.nc

        s_gamma = anp.arange(idx, idx + self.nc)
        idx += self.nc

        sb = anp.arange(idx, idx + self.nb)
        idx += self.nb

        spsi = anp.arange(idx, idx + self.nc)
        idx += self.nc

        self.q = q
        self.gamma = gamma
        self.s_gamma = s_gamma
        self.psi = psi
        self.b = b
        self.spsi = spsi
        self.sb = sb

        return IndicesZ(q, gamma, s_gamma, psi, b, spsi, sb)

    def initialize_z(self, z, q, idx=None):
        z[self.q] = q
        z[self.gamma] = 1
        z[self.s_gamma] = 1
        z[self.psi] = 1
        z[self.b] = 0.1
        z[self.spsi] = 1
        z[self.sb] = 0.1
        return z

    def indices_theta(self):
        return IndicesTheta(self)

    def indices_optimization(self):
        nq = self.nq
        nz = self.num_var()
        nc = self.nc
        nb = self.nb
        n_delta = nz
        off = nq
        ortz = [
            anp.arange(off, off + nc + nb + nc),
            anp.arange(off + 2 * nc + nb, off + 2 * nc + nb + nc + nb + nc),
        ]
        ort_delta = ortz
        socz = []
        soc_delta = []
        iimp = nc
        imdp = nb
        ifri = nc
        equr = anp.arange(0, nq + iimp + imdp + ifri)
        ibimp = nc
        ibmdp = nb
        ibfri = nc
        # Only the normal complementarity terms live in the orthant block.
        ortr = anp.arange(
            nq + iimp + imdp + ifri, nq + iimp + imdp + ifri + ibimp + ibmdp + ibfri
        )
        socr = []
        socri = socr  # residual index grouping per cone
        # bilinear block tracks all complementarity residuals (orthant + SOC)
        bil = ortr  # anp.concatenate([ortr, anp.concatenate(socr)])
        return IndicesOptimization(
            nz=nz,
            n_delta=n_delta,
            ortz=ortz,
            ort_delta=ort_delta,
            socz=socz,
            soc_delta=soc_delta,
            equr=equr,
            ortr=ortr,
            socr=socr,
            socri=socri,
            bil=bil,
        )

    def residual(self, z, theta, kappa):
        if not hasattr(self, "q"):
            self.indices_z()

        z = anp.reshape(z, (-1,))
        theta = anp.reshape(theta, (-1,))

        q0 = theta[self.idx_q1]
        q1 = theta[self.idx_q2]
        u1 = theta[self.idx_u]
        w1 = theta[self.idx_w]
        mu = theta[self.idx_f]
        h = theta[self.idx_h][0]

        q2 = z[self.q]
        gamma1 = z[self.gamma]
        sgamma1 = z[self.s_gamma]
        psi1 = z[self.psi]
        b1 = z[self.b]
        spsi1 = z[self.spsi]
        sb1 = z[self.sb]

        map_mat = anp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])

        v1 = (q2 - q1) / h
        vT_stack = self.p_function(q2) @ v1

        J = self.contact_jacobian(q2)

        cf1 = anp.concatenate([map_mat.T @ b1, gamma1])

        Lambda1 = J.T @ cf1
        dyn = dynamics_auto(self, h, q0, q1, u1, w1, Lambda1, q2)
        phi = self.signed_distance(q2)
        # res_fric = self.mu * gamma1[0] - psi1[0]

        E = np.zeros((self.nc, self.nb))
        for j in range(self.nc):
            start = j * 4
            end = start + 4
            E[j, start:end] = 1.0
        self.E = anp.array(E)

        psi_stack = E.T @ psi1
        res_sd = sgamma1 - phi[2]
        res_vT = sb1 - vT_stack - psi_stack

        res_fric_ineq = spsi1 - (mu * gamma1 - E @ b1)

        res_normal_comp = gamma1 * sgamma1 - kappa
        res_b_comp = b1 * sb1 - kappa
        res_psi_comp = psi1 * spsi1 - kappa

        # print("gamma1 ", gamma1)
        # print("sgamma1 ", sgamma1)
        # print("kappa ", kappa)

        # print("dyn", dyn.shape)
        # print("res_sd", res_sd.shape)
        # print("res_vT", res_vT.shape)
        # print("res_fric_ineq", res_fric_ineq.shape)
        # print("res_normal_comp", res_normal_comp.shape)
        # print("res_b_comp", res_b_comp.shape)
        # print("res_psi_comp", res_psi_comp.shape)

        res = anp.concatenate(
            [
                dyn,  # dynamics residual # 3
                anp.atleast_1d(res_sd),  # normal gap residual # 1
                anp.atleast_1d(res_vT),  # tangential velocity residual (4,)
                anp.atleast_1d(res_fric_ineq),  # 1
                anp.atleast_1d(res_normal_comp),
                anp.atleast_1d(res_b_comp),  # 1
                anp.atleast_1d(res_psi_comp),  # 1
            ]
        )
        # print("res", res)
        # input()
        return res
