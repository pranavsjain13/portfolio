import cvxpy as cp
import numpy as np

from config.config import Config
from agent.agents import Evader
from utils.utils import accel_constraints, vel_constraints


class FormationEvader(Evader):
    def __init__(self, x_init, config: Config, formation_index):
        self.num_pursuers = config.num_pursuers
        self.idx = formation_index
        self.N = config.num_evaders
        self.cfg = config

        super().__init__(config)

        self.state = x_init.astype(float).copy()

    def get_formation_point(self, t):
        # TODO aligned with ta_version/agent/formation.py -> get_formation_point().
        # Implement the moving formation center and the desired point for evader i.
        # Suggested steps:
        # 1. Compute the moving center at time t.
        # 2. Compute evader i's angular offset on the formation circle.
        # 3. Return the desired position and desired velocity.
        # raise NotImplementedError(
        #     "Fill in template_version/agent/formation.py -> FormationEvader.get_formation_point()."
        # )

        # Moving center
        center = self.cfg.center_start + self.cfg.center_vel * t
        
        # Angular offset on formation circle
        angle = self.idx * 2 * np.pi / self.N
        offset = np.array([self.cfg.formation_radius * np.cos(angle), 
                           self.cfg.formation_radius * np.sin(angle)])
        
        p_des = center + offset
        v_des = self.cfg.center_vel
        
        return p_des, v_des

    def _setup_qp(self):
        # TODO aligned with ta_version/agent/formation.py -> FormationEvader._setup_qp().
        # Use the same multi-constraint QP structure as the TA version and fill in
        # the missing controller details.
        self.u_var = cp.Variable(2)
        self.u_nom_param = cp.Parameter(2)
        self.vE_param = cp.Parameter(2)

        self.A_param = cp.Parameter((self.num_pursuers, 2))
        self.b_param = cp.Parameter(self.num_pursuers)

        # TODO: replace this placeholder with the formation-tracking
        # objective that stays close to the nominal input.
        # obj = cp.Minimize(cp.sum_squares(self.u_var))
        obj = cp.Minimize(cp.sum_squares(self.u_var - self.u_nom_param))

        self.constraints = []
        self.constraints.append(self.A_param @ self.u_var <= self.b_param)
        self.constraints += accel_constraints(self.u_var, self.u_max)
        self.constraints += vel_constraints(self.u_var, self.vE_param, self.dt, self.v_max)

        self.prob = cp.Problem(obj, self.constraints)

        self.u_fallback = cp.Variable(2)

        # TODO: replace this placeholder with your Part II fallback if needed.
        # obj_fb = cp.Minimize(cp.sum_squares(self.u_fallback))
        obj_fb = cp.Minimize(cp.sum(self.A_param @ self.u_fallback))

        self.constraints_fb = []
        self.constraints_fb += accel_constraints(self.u_fallback, self.u_max)
        self.constraints_fb += vel_constraints(self.u_fallback, self.vE_param, self.dt, self.v_max)

        self.prob_fallback = cp.Problem(obj_fb, self.constraints_fb)

    def compute_control(self, t, pursuers_list, u_P_list):
        # TODO aligned with ta_version/agent/formation.py -> FormationEvader.compute_control().
        # Build one robust HOCBF constraint per pursuer and solve the evader QP.
        # Suggested steps:
        # 1. Get the desired formation state and compute u_nom.
        # 2. For each pursuer, derive one pairwise HOCBF inequality.
        # 3. Append each pairwise inequality as one row of A_vals / b_vals.
        # 4. Set the QP parameters and solve.
        # 5. If infeasible, use the fallback controller.

        # for p_agent, u_P in zip(pursuers_list, u_P_list):
        #     # TODO: compute the pairwise relative quantities and build
        #     # one row A_ij and scalar b_ij for this evader-pursuer pair.
        #     raise NotImplementedError(
        #         "Fill in template_version/agent/formation.py -> FormationEvader.compute_control()."
        #     )

        p_des, v_des = self.get_formation_point(t)
        u_nom = self.nominal_control(p_target=p_des, v_target=v_des)

        A_vals = np.zeros((self.num_pursuers, 2))
        b_vals = np.zeros(self.num_pursuers)

        gamma1 = self.cfg.gamma1
        gamma2 = self.cfg.gamma2
        R = self.cfg.R

        for i, (p_agent, u_P) in enumerate(zip(pursuers_list, u_P_list)):
            dp = self.p - p_agent.p
            dv = self.v - p_agent.v

            A_val = -2.0 * dp
            b_val = 2.0 * np.dot(dv, dv) - 2.0 * np.dot(dp, u_P) \
                    + 2.0 * (gamma1 + gamma2) * np.dot(dp, dv) \
                    + gamma1 * gamma2 * (np.dot(dp, dp) - R**2)
            
            A_vals[i, :] = A_val
            b_vals[i] = b_val

        self.u_nom_param.value = u_nom
        self.A_param.value = A_vals
        self.b_param.value = b_vals
        self.vE_param.value = self.v

        try:
            self.prob.solve(solver=cp.OSQP, warm_start=True)
            if self.prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return self.u_var.value.copy()
        except:
            pass

        try:
            self.prob_fallback.solve(solver=cp.OSQP, warm_start=True)
            if self.prob_fallback.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return self.u_fallback.value.copy()
        except:
            pass

        return u_nom
