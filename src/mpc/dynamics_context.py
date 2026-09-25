"""Model/environment context for CI-MPC."""

from copy import deepcopy
import numpy as np


def create_dynamics_context(model, env):
    model_obj = deepcopy(model)
    env_obj = deepcopy(env)
    sim = _make_dynamics_context(model_obj, env_obj)
    return sim


class DynamicsContext:
    def __init__(self, model, env, con, res, rz, rtheta, phi):
        self.model = model
        self.env = env
        self.con = con
        self.res = res
        self.rz = rz
        self.rtheta = rtheta
        self.phi = phi


def _make_dynamics_context(model, env):
    return DynamicsContext(
        model,
        env,
        ContactMethods(),
        ResidualMethods(),
        np.zeros((0, 0)),
        np.zeros((0, 0)),
        lambda q: None,
    )


class ContactMethods:
    def __init__(self):
        self.J = 0
        self.phi = 0
        self.cf = 0
        self.vs = 0
        self.d = 0


class ResidualMethods:
    def __init__(self):
        def not_implemented(*args, **kwargs):
            raise NotImplementedError("Not Implemented: use instantiate_residual!")

        self.r = not_implemented
        self.rz = not_implemented
        self.rtheta = not_implemented


def z_initialize_warmstart(z, idx, q):
    z[:] = [1.0] * len(z)
    for i, val in zip(idx, q):
        z[i] = val
    return None

