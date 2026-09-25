import numpy as np
from abc import ABC, abstractmethod


class Policy(ABC):
    @abstractmethod
    def __call__(self, traj, t):
        pass


class EmptyPolicy(Policy):
    def __init__(self, u):
        self.u = u

    def __call__(self, traj, t):
        return self.u


def empty_policy(model):
    return EmptyPolicy(np.zeros(model.nu))
