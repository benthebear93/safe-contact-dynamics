from autograd import numpy as anp
from autograd import jacobian
from src.helper import *
from src.solver.indices import IndicesOptimization, IndicesTheta, IndicesZ
from src.simulator.dynamics import dynamics_auto


class Pusher:

    def __init__(
        self,
        nq=7,
        nu=2,
        nw=0,
        nc=5,
        r_box=0.1,
        r_pusher=0.025,
        mb=1.0,
        mp=10.0,
        I=None,
        mu_body=0.5,
        mu_pusher=0.5,
        gravity=9.81,
        initial_configuration=None,
    ):
        self.nq = nq
        self.nu = nu
        self.nw = nw
        self.nc = nc
        self.mb = mb
        self.mp = mp
        self.r_box = r_box
        self.r_pusher = r_pusher
        self.box_height = 2 * r_box
        self.pusher_height = 2 * r_pusher
        if I is None:
            I = 1.0 / 12.0 * mb * ((2.0 * r_box) ** 2 + (2.0 * r_box) ** 2)
        self.I = I
        self.mu_body = mu_body
        self.mu_pusher = mu_pusher
        self.gravity = gravity

        self.friction_coeff = anp.array([self.mu_body, self.mu_pusher])
        self.nf = len(self.friction_coeff)
        self.lc_fric = 4
        self.nb = self.lc_fric * self.nc

        self.idx_q1 = anp.arange(0, nq)
        self.idx_q2 = anp.arange(nq, 2 * nq)
        self.idx_u = anp.arange(2 * nq, 2 * nq + nu)
        self.idx_w = anp.arange(2 * nq + nu, 2 * nq + nu + nw)
        self.idx_f = anp.arange(2 * nq + nu + nw, 2 * nq + nu + nw + self.nf)
        self.idx_h = anp.arange(
            2 * nq + nu + nw + self.nf, 2 * nq + nu + nw + self.nf + 1
        )

        self.contact_corner_offset = [
            anp.array([+self.r_box, +self.r_box, -self.r_box]),
            anp.array([-self.r_box, +self.r_box, -self.r_box]),
            anp.array([+self.r_box, -self.r_box, -self.r_box]),
            anp.array([-self.r_box, -self.r_box, -self.r_box]),
        ]
        self._nominal_configuration = anp.array(
            [
                0.0,
                0.0,
                self.r_box + 1e-8,
                0.0,  # theta
                0.0,
                -(self.r_box + self.r_pusher) - 1e-8,
                self.r_box,
            ]
        )
        if initial_configuration is not None:
            self.set_nominal_configuration(initial_configuration)

    def M_func(self, q):
        return anp.diag([self.mb, self.mb, self.mb, self.I, self.mp, self.mp, self.mp])

    def C_func(self, q, qdot):
        return anp.array([0.0, 0.0, self.mb * self.gravity, 0.0, 0.0, 0.0, 0.0])

    def kinematics(self, q):
        return anp.array(q)

    def signed_distance(self, q):
        box_pos = q[0:3]
        box_yaw = q[3]
        push_pos = q[4:]
        delta = push_pos - box_pos

        R = rotation_matrix_yaw(box_yaw)

        # existing corner heights
        phi = []
        for corner in self.contact_corner_offset:
            corner_world = box_pos + R @ corner
            phi.append(corner_world[2])

        # Lp approximation (p = s) of a 2D box SDF in the box frame.
        p_rel = delta[0:2]  # [dx, dy] in world frame
        c = anp.cos(box_yaw)
        s = anp.sin(box_yaw)
        R_T = anp.array([[c, s], [-s, c]])  # transpose of [[c, -s],[s, c]]
        p_local = R_T @ p_rel  # 2D point in box frame

        s_exp = 10  # even power for smooth Lp norm; higher ≈ L_inf
        lp_base = anp.sum(p_local**s_exp)
        lp = anp.maximum(lp_base, 1e-12) ** (1.0 / s_exp)
        sdf_box_point = lp - (self.r_box + self.r_pusher)
        phi.append(sdf_box_point)
        return anp.array(phi)  # (5,)

    def input_jacobian(self, q):
        return anp.array(
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [1, 0],
                [0, 1],
                [0, 0],
            ]
        )

    def contact_jacobian(self, q):
        phi_fun = lambda x: self.signed_distance(x)
        return jacobian(phi_fun)(q)

    def nominal_configuration(self):
        return anp.array(self._nominal_configuration)

    def set_nominal_configuration(self, q):
        q_array = anp.array(q)
        if q_array.shape != (self.nq,):
            raise ValueError(
                f"Nominal configuration must have shape ({self.nq},), got {q_array.shape}"
            )
        self._nominal_configuration = q_array

    def friction_coefficients(self):
        # Return current friction coefficients in the order [mu_body, mu_pusher].
        return anp.array([self.mu_body, self.mu_pusher])

    def p_box_function(self, q):
        map_mat = anp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])

        def p(x):
            box_position = x[:3]  # (x, y, z)
            box_rot = x[3]
            R = rotation_matrix_yaw(box_rot)

            blocks = []
            for corner in self.contact_corner_offset:
                corner = anp.array(corner)  # shape (3,)
                corner_world = box_position + R @ corner  # world frame position
                xy = corner_world[:2]  # (x, y)
                blocks.append(map_mat @ xy)  # shape (4,)
            return anp.concatenate(blocks)

        J = jacobian(p)(q)
        # print("J", J.shape)
        return J

    def P_function(self, q):
        P_box = self.p_box_function(q)

        phi_fun = lambda x: self.signed_distance(x)

        N = jacobian(phi_fun)(q)
        N_pusher = N[-1][4:7]
        n_size = anp.sqrt(N_pusher[0] ** 2 + N_pusher[1] ** 2 + N_pusher[2] ** 2)
        n_dir = N_pusher / (n_size + 1e-12)
        t_dir = anp.array([-n_dir[1], n_dir[0]])  # tangent (tx, ty)
        map_mat = anp.array(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
        )  # four pyramid axes
        rows = []
        for ax, ay in map_mat * t_dir:  # each row gives (ax, ay)
            # print("ax ", ax, "ay ", ay)
            r = q[4:6] - q[0:2]
            m = r[0] * ay - r[1] * ax
            row = anp.array([-ax, -ay, 0.0, -m, ax, ay, 0.0])
            rows.append(row)
        P_pusher = anp.vstack(rows)  # 4x7
        P = anp.vstack([P_box, P_pusher])  # 20x7

        return P

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
        eps = 3.16227766e-2
        z[self.q] = q
        # z[self.gamma] = 1
        # z[self.s_gamma] = 1
        # z[self.psi] = 1
        # z[self.b] = 0.1
        # z[self.spsi] = 1
        # z[self.sb] = 0.1
        z[self.gamma] = eps
        z[self.s_gamma] = eps
        z[self.psi] = eps
        z[self.b] = eps
        z[self.spsi] = eps
        z[self.sb] = eps
        return z

    def indices_theta(self):
        return IndicesTheta(self)

    def indices_optimization(self):
        # Ensure index attributes exist.
        if not hasattr(self, "q"):
            self.indices_z()

        nq = self.nq
        nz = self.num_var()
        nc = self.nc
        nb = self.nb
        n_delta = nz
        ortz = [
            anp.concatenate([self.gamma, self.b, self.psi]),
            anp.concatenate([self.s_gamma, self.sb, self.spsi]),
        ]
        ort_delta = ortz
        socz = []
        soc_delta = []
        iimp = nc
        imdp = nb
        ifri = nc
        equr = anp.arange(0, nq + iimp + imdp + ifri)
        # All complementarity terms live in the orthant block.
        ortr = anp.arange(
            nq + iimp + imdp + ifri, nq + iimp + imdp + ifri + nc + nb + nc
        )
        socr = []
        socri = socr  # residual index grouping per cone
        # bilinear block tracks all complementarity residuals (orthant)
        bil = ortr
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
        # Ensure index attributes are initialized even if indices_z() wasn't called earlier.
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

        v1 = (q2 - q1) / h  # generalized object velocity
        P = self.P_function(q2)
        vT_stack = P @ v1

        # Generalized contact wrench: normals via J (q1), tangentials via P (q1).
        J = self.contact_jacobian(q1)
        P_force = self.P_function(q1)
        Lambda_normal = J.T @ gamma1
        Lambda_tangent = P_force.T @ b1
        Lambda1 = Lambda_normal + Lambda_tangent
        dyn = dynamics_auto(self, h, q0, q1, u1, w1, Lambda1, q2)

        phi = self.signed_distance(q2)  # (nc,)

        E = anp.zeros((self.nc, self.nb))
        for j in range(self.nc):
            start = j * 4
            end = start + 4
            E[j, start:end] = 1.0
        self.E = anp.array(E)

        psi_stack = E.T @ psi1
        res_sd = phi - sgamma1
        res_vT = sb1 - vT_stack - psi_stack

        # Use per-contact friction from theta: four box corners use mu_body, pusher uses mu_pusher.
        mu_body_world = mu[0]
        mu_pusher_world = mu[1]
        mu_contact = anp.concatenate(
            [anp.full(self.nc - 1, mu_body_world), anp.array([mu_pusher_world])]
        )
        res_fric_ineq = spsi1 - (mu_contact * gamma1 - E @ b1)

        res_normal_comp = gamma1 * sgamma1 - kappa
        res_b_comp = b1 * sb1 - kappa
        res_psi_comp = psi1 * spsi1 - kappa
        # print("dyn", dyn.shape)

        # util = np.linalg.norm(E @ b1, ord=np.inf) / (mu_body_world * anp.max(gamma1))
        # print(f"friction utilization: {util}")
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
                anp.atleast_1d(res_b_comp),
                anp.atleast_1d(res_psi_comp),
            ]
        )

        return res
