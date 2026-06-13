
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.animation import PillowWriter
from matplotlib.patches import Circle
from connectivity_and_resilience import *
from config import *
from plot_h_functions import plotter


# -------------------- Simulation setup --------------------
plt.ion()
fig, ax = plt.subplots()
ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-2.5, 6.5)

# Video settings
fps = 20
metadata = dict(title='Resilient_aware Simulation', artist='Matplotlib', comment='Multi-robot CBF simulation')
# writer = FFMpegWriter(fps=fps, metadata=metadata)
writer = PillowWriter(fps=fps, metadata=metadata)
video_name = "video_4_4b.gif"


# CBF alphas - tuned for resilience-aware control but can be adjusted 
alpha_conn = 0.8
alpha_coll = 6.0
alpha_obs = 6.0

# Resilience parameter
F = 1

# Storage for h function evolution
h_res_history = []
time_steps = []

# -------------------- Main loop --------------------
with writer.saving(fig, video_name, dpi=100):
    for step in range(num_steps):
        sigma = compute_sigma(robots, R)
        A = adjacency_matrix(robots, R, sigma)
        L = laplacian_matrix(A)

        # Reference control toward goal (unit vector)
        u_ref = nominal_controller(n, robots, goals)

        # CVXPY QP
        u = cp.Variable((n, 2))
        constraints = []

        # F-resilient connectivity CBF
        h_res = f_resilient_CBF(L, F, epsilon)
        h_res_history.append(h_res)
        time_steps.append(step * dt)
        print("time step, h_res:", step, h_res)


        # Approximate gradient of lambda2 numerically (simplification)
        # Here we use identity to allow QP feasibility; can improve with proper mu_m
        constraints.append(cp.sum(mu_m(robots, u, R, sigma)) >= -alpha_conn * h_res)

        # Inter-robot collision
        # TODO: # Construct the CBF constraints for inter-robot collision avoidance
        for i in range(n):
            for j in range(i+1, n):
                h, dh_dxi, dh_dxj = collision_CBF(robots[i], robots[j], d_min)
                constraints.append(dh_dxi @ u[i] + dh_dxj @ u[j] >= -alpha_coll * h) # Add CBF constraint
                if h < 0:
                    print(f"Collision warning between robot {i} and robot {j}, h={h}")

        # Obstacle avoidance
        # TODO: # Construct the CBF constraints for obstacle avoidance
        for i in range(n):
            for obs in obstacles:
                obs_center = np.array(obs[:2])
                h, dh_dxi = obstacle_CBF(robots[i], obs_center, obs[2], r_s)
                constraints.append(dh_dxi @ u[i] >= -alpha_obs * h) # Add CBF constraint
                if h < 0:
                    print(f"Obstacle avoidance warning for robot {i} and obstacle at {obs[:2]}, h={h}")

        # Objective: stay close to reference
        objective = cp.Minimize(cp.sum_squares(u - u_ref))
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, verbose=False)

        # Update robots
        u_val = u.value
        robots += u_val * dt

        # Plot robots and obstacles
        ax.clear()
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-2.5, 6.5)
        for r in robots:
            circle = Circle((r[0], r[1]), 0.15, color='blue', fill=True, alpha=0.5)
            ax.add_patch(circle)
        for obs in obstacles:
            circle = Circle(obs[:2], obs[2], color='k', alpha=1)
            ax.add_artist(circle)
        plt.pause(0.01)

        writer.grab_frame()

plotter(time_steps, h_res_history, epsilon,'Resilience-Aware Control: h Function Evolution')

