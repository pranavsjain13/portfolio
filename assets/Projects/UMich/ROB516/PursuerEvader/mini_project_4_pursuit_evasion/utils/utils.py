import numpy as np

def saturate_norm(u: np.ndarray, u_max: float, eps: float = 1e-12) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    n = np.linalg.norm(u)
    if n <= u_max:
        return u
    if n < eps:
        return np.zeros_like(u)
    return (u_max / n) * u

def accel_constraints(u, u_max):
    return [
        u[0] <= u_max,
        u[0] >= -u_max,
        u[1] <= u_max,
        u[1] >= -u_max,
    ]

def vel_constraints(u, v_param, dt, v_max):
    return [
        v_param[0] + dt * u[0] <= v_max,
        v_param[0] + dt * u[0] >= -v_max,
        v_param[1] + dt * u[1] <= v_max,
        v_param[1] + dt * u[1] >= -v_max
    ]
