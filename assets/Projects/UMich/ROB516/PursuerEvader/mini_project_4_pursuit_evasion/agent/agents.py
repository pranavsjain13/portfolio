import numpy as np
import cvxpy as cp

from config.config import Config
from utils.utils import saturate_norm, accel_constraints, vel_constraints


class Agent:
    def __init__(self, x_init: np.ndarray, u_max: float, v_max: float, dt: float):
        self.state = x_init.astype(float).copy()  # [px, py, vx, vy]
        self.u_max = u_max
        self.v_max = v_max
        self.dt = dt
        self.u_last = np.zeros(2)

    @property
    def p(self):
        return self.state[:2]

    @property
    def v(self):
        return self.state[2:]

    def step(self, u: np.ndarray):
        self.state[:2] += self.state[2:] * self.dt
        self.state[2:] += u * self.dt
        self.u_last = u


class Pursuer(Agent):
    def __init__(self, config: Config):
        super().__init__(config.x_P_init, config.uP_max, config.vP_max, config.dt)
        self.cfg = config
        self._setup_qp()

    def _setup_qp(self):
        # TODO aligned with ta_version/agent/agents.py -> Pursuer._setup_qp().
        # Replace the placeholder objective below with the worst-case pursuer
        # objective from Problem 4.1(b). Keep the same variable/parameter setup.
        self.u_var = cp.Variable(2)
        self.p_rel_param = cp.Parameter(2)
        self.vP_param = cp.Parameter(2)

        # obj_expr = cp.sum_squares(self.u_var)  # placeholder
        obj_expr = -self.p_rel_param.T @ self.u_var

        constraints = []
        constraints += accel_constraints(self.u_var, self.u_max)
        constraints += vel_constraints(self.u_var, self.vP_param, self.dt, self.v_max)

        obj = cp.Minimize(obj_expr)
        self.prob = cp.Problem(obj, constraints)

    def compute_control(self, p_E, v_E):
        # TODO aligned with ta_version/agent/agents.py -> Pursuer.compute_control().
        # This should return the worst-case pursuer input u_P^*.
        # Suggested steps:
        # 1. Compute the relative states you need.
        # 2. Set the CVXPY parameters.
        # 3. Solve the QP.
        # 4. If infeasible, use a bounded fallback controller.

        # raise NotImplementedError(
        #     "Fill in template_version/agent/agents.py -> Pursuer._setup_qp() and Pursuer.compute_control()."
        # )
        p_rel = p_E - self.p
        self.p_rel_param.value = p_rel
        self.vP_param.value = self.v
        
        try:
            self.prob.solve(solver=cp.OSQP, warm_start=True)
            if self.prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return self.u_var.value.copy()
        except:
            pass
            
        # Fallback if infeasible: aggressive pursuit saturated to max bounds
        u = p_rel / (np.linalg.norm(p_rel) + 1e-6) * self.u_max
        return saturate_norm(u, self.u_max)


class Evader(Agent):
    def __init__(self, config: Config):
        super().__init__(config.x_E_init, config.uE_max, config.vE_max, config.dt)
        self.cfg = config
        self._setup_qp()

    def _setup_qp(self):
        # TODO aligned with ta_version/agent/agents.py -> Evader._setup_qp().
        # Keep the same primary QP structure and replace any placeholder pieces
        # with your Problem 4.1(c) controller.
        self.u_var = cp.Variable(2)
        self.u_nom_param = cp.Parameter(2)
        self.A_param = cp.Parameter(2)
        self.b_param = cp.Parameter(1)
        self.vE_param = cp.Parameter(2)

        # TODO: replace this placeholder with the objective that keeps
        # the safe input close to the nominal controller.
        # obj = cp.Minimize(cp.sum_squares(self.u_var))
        obj = cp.Minimize(cp.sum_squares(self.u_var - self.u_nom_param))

        self.constraints_safety = []
        self.constraints_safety.append(self.A_param @ self.u_var <= self.b_param)
        self.constraints_safety += accel_constraints(self.u_var, self.u_max)
        self.constraints_safety += vel_constraints(self.u_var, self.vE_param, self.dt, self.v_max)

        self.prob = cp.Problem(obj, self.constraints_safety)

        self.u_fallback = cp.Variable(2)
        # TODO: replace this placeholder fallback objective with your
        # best-effort fallback from the handout if you choose to implement it.
        # obj_fallback = cp.Minimize(cp.sum_squares(self.u_fallback))
        obj_fallback = cp.Minimize(self.A_param @ self.u_fallback)

        constraints_fallback = []
        constraints_fallback += accel_constraints(self.u_fallback, self.u_max)
        constraints_fallback += vel_constraints(self.u_fallback, self.vE_param, self.dt, self.v_max)

        self.prob_fallback = cp.Problem(obj_fallback, constraints_fallback)

    def nominal_control(self, p_target=None, v_target=None):
        # TODO aligned with ta_version/agent/agents.py -> Evader.nominal_control().
        # Implement the nominal PD controller used in Problem 4.1(c).
        if p_target is None:
            p_target = self.cfg.p_goal
        if v_target is None:
            v_target = np.zeros(2)

        # Suggested steps:
        # 1. Compute position and velocity tracking errors.
        # 2. Form the PD acceleration command.
        # 3. Saturate the result before returning it.
        # raise NotImplementedError(
        #     "Fill in template_version/agent/agents.py -> Evader.nominal_control()."
        # )
        err_p = self.p - p_target
        err_v = self.v - v_target
        
        # PD Controller
        u_nom = -self.cfg.kp_E * err_p - self.cfg.kd_E * err_v
        
        return saturate_norm(u_nom, self.u_max)

    def compute_control(self, p_P, v_P, u_P):
        # TODO aligned with ta_version/agent/agents.py -> Evader.compute_control().
        # Build the robust HOCBF inequality A*u <= b and solve the safe evader QP.
        # Suggested steps:
        # 1. Compute the nominal input.
        # 2. Compute the relative position/velocity quantities.
        # 3. Build the HOCBF terms and rearrange them into A_val and b_val.
        # 4. Set the QP parameters and solve.
        # 5. If infeasible, solve the fallback or return u_nom.

        # raise NotImplementedError(
        #     "Fill in template_version/agent/agents.py -> Evader.compute_control()."
        # )
        u_nom = self.nominal_control()
        
        dp = self.p - p_P
        dv = self.v - v_P
        
        gamma1 = self.cfg.gamma1
        gamma2 = self.cfg.gamma2
        R = self.cfg.R
        
        # Formulate HOCBF A * u <= b constraint
        A_val = -2.0 * dp
        b_val = 2.0 * np.dot(dv, dv) - 2.0 * np.dot(dp, u_P) \
                + 2.0 * (gamma1 + gamma2) * np.dot(dp, dv) \
                + gamma1 * gamma2 * (np.dot(dp, dp) - R**2)

        self.u_nom_param.value = u_nom
        self.A_param.value = A_val
        self.b_param.value = np.array([b_val])
        self.vE_param.value = self.v

        try:
            self.prob.solve(solver=cp.OSQP, warm_start=True)
            if self.prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return self.u_var.value.copy()
        except:
            pass

        # Fallback evaluation
        try:
            self.prob_fallback.solve(solver=cp.OSQP, warm_start=True)
            if self.prob_fallback.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return self.u_fallback.value.copy()
        except:
            pass

        return u_nom
