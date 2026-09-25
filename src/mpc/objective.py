import numpy as np


class Objective:
    pass  # Abstract base class


class TrackingObjective(Objective):
    def __init__(self, q, u, gamma, b):
        self.q = q
        self.u = u
        self.gamma = gamma
        self.b = b


def tracking_objective(model, friction_dim, H, q=None, u=None, gamma=None, b=None):
    if q is None:
        q = [np.diag(np.zeros(model.nq)) for _ in range(H)]
    if u is None:
        u = [np.diag(np.zeros(model.nu)) for _ in range(H)]
    if gamma is None:
        gamma = [np.diag(np.zeros(model.nc)) for _ in range(H)]
    if b is None:
        b = [np.diag(np.zeros(model.nc * friction_dim)) for _ in range(H)]
    return TrackingObjective(q, u, gamma, b)


class TrackingVelocityObjective(Objective):
    def __init__(self, q, v, u, gamma, b, v_target, q_target):
        self.q = q
        self.v = v
        self.u = u
        self.gamma = gamma
        self.b = b
        self.v_target = v_target
        self.q_target = q_target


