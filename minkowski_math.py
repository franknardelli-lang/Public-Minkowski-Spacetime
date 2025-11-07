import numpy as np

def lorentz_transform(points, v):
    """Apply Lorentz transformation to a set of (t, x) points."""
    gamma = 1 / np.sqrt(1 - v*v)
    t, x = points[:, 0], points[:, 1]
    t_p = gamma * (t - v * x)
    x_p = gamma * (x - v * t)
    return np.column_stack((t_p, x_p))


def relative_velocity(u, v):
    """Velocity addition formula for special relativity."""
    return (u - v) / (1 - u * v)


def spacetime_intervals(events, c=1, v_frame=0):
    """Compute invariant intervals between events under given frame velocity."""
    from minkowski_math import lorentz_transform  # safe local import
    Atp = lorentz_transform(events, v_frame)
    (tA, xA), (tB, xB), (tC, xC) = Atp

    def S2(a, b):
        return (c*(b[0]-a[0]))**2 - (b[1]-a[1])**2

    S2_AB = S2((tA, xA), (tB, xB))
    S2_AC = S2((tA, xA), (tC, xC))
    S2_BC = S2((tB, xB), (tC, xC))

    return S2_AB, S2_AC, S2_BC
