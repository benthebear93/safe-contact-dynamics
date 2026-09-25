import numpy as np
import sympy as sp

# world
# Abstract types are represented as strings in Environment class attributes

# friction cone
# Similarly, cone types as strings


class Environment:
    def __init__(self, rotation_type, cone_type, surf, surf_grad, nc_impact):
        self.rotation_type = rotation_type
        self.cone_type = cone_type
        self.surf = surf
        self.surf_grad = surf_grad
        self.nc_impact = nc_impact


def _is_sympy_array(x):
    """Return True if x contains any SymPy object."""
    if isinstance(x, (sp.Basic, sp.MatrixBase)):
        return True
    try:
        return any(isinstance(xi, sp.Basic) for xi in x)
    except TypeError:
        return False


def environment_3d_flat(nc_impact=0, cone="LinearizedCone"):
    # z = 0 flat plane (3D)
    def surf(x):
        # For flat surface, value is always 0
        if _is_sympy_array(x):
            return sp.Integer(0)
        else:
            return 0.0

    def surf_grad(x):
        # [∂z/∂x, ∂z/∂y] = [0, 0]
        if _is_sympy_array(x):
            return sp.Matrix([0, 0])
        else:
            x = np.asarray(x)
            return np.zeros_like(x)

    return Environment("R3", cone, surf, surf_grad, nc_impact=nc_impact)
