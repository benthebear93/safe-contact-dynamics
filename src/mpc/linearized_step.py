import numpy as np
from src.helper import num_data, num_var


class LinearizedStep:
    def __init__(self, z, theta, kappa, r, rz, r_theta):
        self.z = z
        self.theta = theta
        self.kappa = kappa
        self.r = r
        self.rz = rz
        self.r_theta = r_theta


def linearized_step(s, z, theta, kappa):
    # print(f"lin step z {z}")
    # print(f"lin step theta {theta}")
    model = s.model
    env = s.env

    nz = num_var(model, env)
    n_theta = num_data(model)
    # print(f"linearized step nz {nz} n_theta {n_theta}\n")

    z0 = z.copy()
    theta0 = theta.copy()
    kappa0 = kappa

    r0 = np.zeros(nz)
    rz0 = np.zeros((nz, nz))
    r_theta0 = np.zeros((nz, n_theta))

    r0 = s.res.r(z0, theta0, kappa0)
    rz0 = s.res.rz(z0, theta0)
    r_theta0 = s.res.rtheta(z0, theta0)

    # print(f"lin step r0\n {r0}")
    # print(f"lin step rz0\n {rz0}")
    return LinearizedStep(z0, theta0, kappa0, r0, rz0, r_theta0)
