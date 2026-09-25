import numpy as np
from src.helper import num_data, num_var, nc_impact_check
from src.solver.indices import linearization_term_index, linearization_var_index
from .schur import schur_factorize, schur_solve, schur


def _safe_pos_denom(x, eps):
    """Return a finite positive denominator vector with lower bound eps."""
    eps = max(float(eps), 1.0e-12)
    x_arr = np.asarray(x, dtype=float)
    # Replace non-finite values first, then clamp away from zero.
    x_arr = np.where(np.isfinite(x_arr), x_arr, eps)
    return np.maximum(x_arr, eps)


class RLin:
    def __init__(self, s, z0, theta0, r0, rz0, r_theta0, verbose=False):
        model = s.model
        env = s.env

        nq = model.nq
        nc = model.nc
        nb = nc_impact_check(env, nc)
        # nc * friction_dim(env)

        nz = num_var(model, env)
        n_theta = num_data(model)
        nx = nq  # state
        ny = 2 * nc + nb  # contact + psi + friction
        nn = nc + nb

        idyn, irst, ibil, ialt = linearization_term_index(
            model, env, verbose=verbose
        )  # dyn, residual stact, bilinear, alterantive
        ix, iy1, iy2 = linearization_var_index(
            model, env, verbose=verbose
        )  # state, complementary, slack
        i_theta = np.arange(n_theta)

        # for symbolic
        # self.rdyn0 = r0[0][idyn]
        # self.rrst0 = r0[0][irst]
        # self.rbil0 = r0[0][ibil]

        self.rdyn0 = r0[idyn]
        self.rrst0 = r0[irst]
        self.rbil0 = r0[ibil]

        self.rdyn = np.zeros(nx)
        self.rrst = np.zeros(ny)
        self.rbil = np.zeros(ny)
        self.rbil_cache = np.zeros(ny)

        self.Dx = rz0[np.ix_(idyn, ix)]
        self.Dy1 = rz0[np.ix_(idyn, iy1)]

        # impact, contact force, friciton vel
        self.Rx = rz0[np.ix_(irst, ix)]
        self.Ry1 = rz0[np.ix_(irst, iy1)]
        self.Ry2 = np.diag(rz0[np.ix_(irst, iy2)])

        self.r_theta_dyn = r_theta0[np.ix_(idyn, i_theta)]
        self.r_theta_rst = r_theta0[np.ix_(irst, i_theta)]
        self.r_theta_bil = r_theta0[np.ix_(ibil, i_theta)]

        if verbose:
            np.set_printoptions(suppress=True, precision=4, floatmode="maxprec_equal")
            print(f"Dx \n {self.Dx}")
            print(f"Dy1 \n{self.Dy1}")
            print(f"Rx \n{self.Rx}")
            print(f"Ry1 \n{self.Ry1}")
            print(f"Ry2 \n{self.Ry2}")

            print(f"r_theta_dyn \n{self.r_theta_dyn}")
            print(f"r_theta_rst \n{self.r_theta_rst}")
            print(f"r_theta_bil \n{self.r_theta_bil}")

        self.x0 = [z0[i] for i in ix]
        self.y10 = [z0[i] for i in iy1]  # z0[iy1]
        self.y20 = [z0[i] for i in iy2]  # z0[iy2]
        self.theta0 = [theta0[i] for i in i_theta]  # theta0[i_theta]

        self.x = np.zeros(nx)
        self.y1 = np.zeros(ny)
        self.y2 = np.zeros(ny)
        self.theta = np.zeros(n_theta)

        self.nz = nz
        self.n_theta = n_theta
        self.ix = ix
        self.iy1 = iy1
        self.iy2 = iy2
        self.i_theta = i_theta
        self.idyn = idyn
        self.irst = irst
        self.ibil = ibil
        self.ialt = ialt

        self.alt = np.zeros(nc)
        self.alt_zeros = np.zeros(nn)


class RZLin:
    def __init__(self, s, rz0, eps: float = 0.0):
        model = s.model
        env = s.env

        nq = model.nq
        nc = model.nc
        nspi = nc
        nb = nc_impact_check(env, nc)

        nz = num_var(model, env)
        nx = nq
        ny = nc + nspi + nb

        # print(f"Rzlin nz  {nz} nx {nx}  ny {ny}")
        # Terms
        # dyn = [dyn]
        # rst = [s1  - ..., ≡ ialt
        #        η1  - ...,
        #        s2  - ...,]
        # bil = [γ1 .* s1 .- κ;
        #        b1 .* η1 .- κ;
        #        ψ1 .* s2 .- κ]
        idyn, irst, ibil, ialt = linearization_term_index(
            model, env, quat=False, verbose=False
        )

        # Vars
        # ix = [dyn]
        # iy1 = [γ1 .* s1 .- κ;
        #        b1 .* η1 .- κ;
        #        ψ1 .* s2 .- κ]
        # iy2 = [s1  - ..., ≡ ialt
        #        η1  - ...,
        #        s2  - ...,]
        ix, iy1, iy2 = linearization_var_index(
            model, env
        )  # dyn, impact+friction+contact_vel, slack variables

        # Fill the matrix blocks rz0s
        self.Dx = rz0[np.ix_(idyn, ix)]  # nx*nx
        self.Dy1 = rz0[np.ix_(idyn, iy1)]  # nx*ny
        self.Rx = rz0[np.ix_(irst, ix)]  # ny*nx
        self.Ry1 = rz0[np.ix_(irst, iy1)]  # ny*ny
        self.Ry2 = np.diag(rz0[np.ix_(irst, iy2)])  # ny
        # z = eta, s_phi, s_psi
        self.y1 = np.diag(rz0[np.ix_(ibil, iy2)])  # ny  y = gamma, psi, beta
        self.y2 = np.diag(rz0[np.ix_(ibil, iy1)])  # ny
        self.eps = max(float(eps), 1.0e-12)
        y1_safe = _safe_pos_denom(self.y1, self.eps)
        y2_safe = _safe_pos_denom(self.y2, self.eps)

        # print("self.y1", self.y1)
        # print("self.y2", self.y2)
        self.D = self.Ry1 - np.diag(self.Ry2 * y2_safe / y1_safe)
        self.nx = nx
        self.ny = ny

        M = np.block([[self.Dx, self.Dy1], [self.Rx, self.D]])

        # Schur complement
        self.S = schur(M, n=nx, m=ny)

        self.M = np.array(M, order="F", dtype=float)
        self.M1 = self.M[:nx, :nx]
        self.M2 = self.M[:nx, nx : nx + ny]
        self.M3 = self.M[nx : nx + ny, :nx]
        self.M4 = self.M[nx : nx + ny, nx : nx + ny]

        A = np.block(
            [
                [self.Dx, self.Dy1, np.zeros((nx, ny))],
                [self.Rx, self.Ry1, np.diag(self.Ry2)],
            ]
        )

        # Least square
        AA = A.T @ np.linalg.inv(A @ A.T)

        self.A1 = AA[:nx, :nx]
        self.A2 = AA[:nx, nx : nx + ny]
        self.A3 = AA[nx : nx + ny, :nx]
        self.A4 = AA[nx : nx + ny, nx : nx + ny]
        self.A5 = AA[nx + ny : nx + ny * 2, :nx]
        self.A6 = AA[nx + ny : nx + ny * 2, nx : nx + ny]

        self.ix = ix
        self.iy1 = iy1
        self.iy2 = iy2
        self.idyn = idyn
        self.irst = irst
        self.ibil = ibil


class RThetaLin:
    def __init__(self, s, r_theta0):
        model = s.model
        env = s.env

        nq = model.nq
        nc = model.nc
        nb = nc_impact_check(env, nc)
        # nc * friction_dim(env) - 1

        self.n_theta = num_data(model)
        nx = nq
        ny = 2 * nc + nb

        idyn, irst, ibil, ialt = linearization_term_index(model, env, False)
        i_theta = np.arange(self.n_theta)

        row_idx = idyn + irst
        self.r_theta0 = np.array(r_theta0[np.ix_(row_idx, i_theta)])
        self.r_theta_dyn0 = r_theta0[np.ix_(idyn, i_theta)]
        self.r_theta_rst0 = r_theta0[np.ix_(irst, i_theta)]
        self.r_theta_bil0 = r_theta0[np.ix_(ibil, i_theta)]

        self.i_theta = i_theta
        self.idyn = idyn
        self.irst = irst
        self.ibil = ibil


def rlin(r, z, theta, kappa):
    z = np.array(z)
    theta = np.array(theta)

    r.x = z[r.ix]
    r.y1 = z[r.iy1]
    r.y2 = z[r.iy2]
    r.theta = theta[r.i_theta]

    r.rdyn = (
        r.rdyn0
        + r.Dx @ (np.array(r.x) - np.array(r.x0))
        + r.Dy1 @ (np.array(r.y1) - np.array(r.y10))
        + r.r_theta_dyn @ (np.array(r.theta) - np.array(r.theta0))
    )

    temp = np.array(r.x) - np.array(r.x0)

    r.rrst = (
        r.rrst0
        + r.Rx @ (np.array(r.x) - np.array(r.x0))
        + r.Ry1 @ (np.array(r.y1) - np.array(r.y10))
        + r.Ry2 * (np.array(r.y2) - np.array(r.y20))
        + r.r_theta_rst @ (np.array(r.theta) - np.array(r.theta0))
        + np.concatenate([r.alt, r.alt_zeros])
    )
    r.rbil = np.array(r.y1) * np.array(r.y2) - kappa


def rzlin(rz, z, theta, reg=0.0):
    z = np.array(z)
    rz.y1 = z[rz.iy1]
    rz.y2 = z[rz.iy2]
    eps = max(float(reg), float(getattr(rz, "eps", 1.0e-12)), 1.0e-12)
    y1_reg = _safe_pos_denom(rz.y1, eps)
    y2_reg = _safe_pos_denom(rz.y2, eps)
    rz.D = rz.Ry1 - np.diag(rz.Ry2 * y2_reg / y1_reg)
    schur_factorize(rz.S, rz.D)
    # Update the assembled block matrix used by the linear solver.
    nx, ny = rz.nx, rz.ny
    rz.M[:nx, :nx] = rz.Dx
    rz.M[:nx, nx : nx + ny] = rz.Dy1
    rz.M[nx : nx + ny, :nx] = rz.Rx
    rz.M[nx : nx + ny, nx : nx + ny] = rz.D


def general_correction_term_lin(r, Delta, ortr, socr, socri, ortDelta, socDelta):
    Delta_o1 = Delta[ortDelta[0]]
    Delta_o2 = Delta[ortDelta[1]]
    r.rbil_cache = Delta_o1 * Delta_o2  # adding second to add predictor and! centering
    r.rbil += r.rbil_cache


def linear_solve(Delta, rz, r, reg=0.0):
    rdyn = r.rdyn
    rrst = r.rrst
    rbil = r.rbil
    Ry2 = rz.Ry2
    y1 = rz.y1
    y2 = rz.y2
    eps = max(float(reg), float(getattr(rz, "eps", 1.0e-12)), 1.0e-12)
    y1_reg = _safe_pos_denom(y1, eps)
    y2_reg = _safe_pos_denom(y2, eps)

    u = rdyn
    v = rrst - Ry2 * rbil / y1_reg

    # solve jacobian
    schur_solve(rz.S, u, v)  # get dx, dy1
    Delta[rz.ix] = rz.S.x
    Delta[rz.iy1] = rz.S.y
    Delta[rz.iy2] = (rbil - y2_reg * Delta[rz.iy1]) / y1_reg  # y2 using dx, dy1


def linear_solve_theta(delta_z, rz, r_theta, reg=0.0):
    r_theta_dyn = r_theta.r_theta_dyn0
    r_theta_rst = r_theta.r_theta_rst0
    r_theta_bil = r_theta.r_theta_bil0
    Ry2 = rz.Ry2
    y1 = rz.y1
    y2 = rz.y2
    eps = max(float(reg), float(getattr(rz, "eps", 1.0e-12)), 1.0e-12)
    y1_reg = _safe_pos_denom(y1, eps)
    y2_reg = _safe_pos_denom(y2, eps)
    ix = rz.ix
    iy1 = rz.iy1
    iy2 = rz.iy2

    for i in range(r_theta.n_theta):
        u = r_theta_dyn[:, i]
        v = r_theta_rst[:, i]
        schur_solve(rz.S, u, v)
        delta_z[ix, i] = rz.S.x
        delta_z[iy1, i] = rz.S.y
        delta_z[iy2, i] = (r_theta_bil[:, i] - y2_reg * delta_z[iy1, i]) / y1_reg
    return delta_z


def r_theta(r_theta, z, theta):
    pass
