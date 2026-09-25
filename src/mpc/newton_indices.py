import numpy as np
from abc import ABC
from src.helper import friction_dim


class NewtonIndices(ABC):
    pass


class NewtonIndicesConfigurationForce(NewtonIndices):
    def __init__(self, nd, nr, iq, iu, iγ, ib, iν, iz, iθ, Iq, Iu, Iγ, Ib, Iν, Iz, Iθ):
        self.nd = nd
        self.nr = nr
        self.iq = iq
        self.iu = iu
        self.iγ = iγ
        self.ib = ib
        self.iν = iν
        self.iz = iz
        self.iθ = iθ
        self.Iq = Iq
        self.Iu = Iu
        self.Iγ = Iγ
        self.Ib = Ib
        self.Iν = Iν
        self.Iz = Iz
        self.Iθ = Iθ


def newton_indices_configuration_force(model, env, H):
    nq = model.nq  # configuration
    nu = model.nu  # control
    nw = model.nw  # disturbance
    nc = model.nc  # contact
    nb = nc * friction_dim(env)  # linear friction
    nd = nq + nc + nb  # implicit dynamics constraint
    nr = nq + nu + nc + nb + nd  # size of a one-time-step block

    off = 0
    iq = np.arange(off + 1, off + nq + 1)  # index of the configuration q2
    off += nq
    iu = np.arange(off + 1, off + nu + 1)  # index of the control u1
    off += nu
    iγ = np.arange(off + 1, off + nc + 1)  # index of the impact γ1
    off += nc
    ib = np.arange(off + 1, off + nb + 1)  # index of the linear friction b1
    off += nb
    iν = np.arange(
        off + 1, off + nd + 1
    )  # index of the dynamics lagrange multiplier ν1
    off += nd
    iz = np.concatenate((iq, iγ, ib))  # index of the IP solver solution [q2, γ1, b1]
    iθ = np.concatenate(
        (iq - 2 * nr, iq - nr, iu)
    )  # index of the IP solver data [q0, q1, u1]

    Iq = [(t - 1) * nr + iq for t in range(1, H + 1)]
    Iu = [(t - 1) * nr + iu for t in range(1, H + 1)]
    Iγ = [(t - 1) * nr + iγ for t in range(1, H + 1)]
    Ib = [(t - 1) * nr + ib for t in range(1, H + 1)]
    Iν = [(t - 1) * nr + iν for t in range(1, H + 1)]
    Iz = [(t - 1) * nr + iz for t in range(1, H + 1)]
    Iθ = [(t - 1) * nr + iθ for t in range(1, H + 1)]

    return NewtonIndicesConfigurationForce(
        nd, nr, iq, iu, iγ, ib, iν, iz, iθ, Iq, Iu, Iγ, Ib, Iν, Iz, Iθ
    )


class NewtonIndicesConfiguration(NewtonIndices):
    def __init__(self, nd, nr, iq, iu, iν, iz, iθ, Iq, Iu, Iν, Iz, Iθ):
        self.nd = nd
        self.nr = nr
        self.iq = iq
        self.iu = iu
        self.iν = iν
        self.iz = iz
        self.iθ = iθ
        self.Iq = Iq
        self.Iu = Iu
        self.Iν = Iν
        self.Iz = Iz
        self.Iθ = Iθ


def newton_indices_configuration(model, env, H):
    #
    nq = model.nq  # configuration
    nu = model.nu  # control
    nw = model.nw  # disturbance
    nc = model.nc  # contact
    nd = nq  # implicit dynamics constraint
    nr = nq + nu + nd  # size of a one-time-step block

    off = 0
    iq = np.arange(off, off + nq)  # index of the configuration q2
    off += nq
    iu = np.arange(off, off + nu)  # index of the control u1
    off += nu
    iν = np.arange(off, off + nd)  # index of the dynamics lagrange multiplier ν1
    off += nd
    iz = np.copy(iq)  # index of the IP solver solution [q2]
    iθ = np.concatenate(
        (iq - 2 * nr, iq - nr, iu)
    )  # index of the IP solver data [q0, q1, u1]

    Iq = [(t - 1) * nr + iq for t in range(1, H + 1)]
    Iu = [(t - 1) * nr + iu for t in range(1, H + 1)]
    Iν = [(t - 1) * nr + iν for t in range(1, H + 1)]
    Iz = [(t - 1) * nr + iz for t in range(1, H + 1)]
    Iθ = [(t - 1) * nr + iθ for t in range(1, H + 1)]

    return NewtonIndicesConfiguration(nd, nr, iq, iu, iν, iz, iθ, Iq, Iu, Iν, Iz, Iθ)


def newton_indices(model, env, H, mode="configurationforce"):
    if mode == "configurationforce":
        return newton_indices_configuration_force(model, env, H)
    elif mode == "configuration":
        return newton_indices_configuration(model, env, H)
    else:
        raise ValueError("mode not implemented")
