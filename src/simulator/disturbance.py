import numpy as np
from abc import ABC, abstractmethod


class Disturbances(ABC):
    @abstractmethod
    def __call__(self, x, t):
        pass


class EmptyDisturbances(Disturbances):
    def __init__(self, w):
        self.w = w

    def __call__(self, x, t):
        return self.w


def empty_disturbances(model):
    return EmptyDisturbances(np.zeros(model.nw))
