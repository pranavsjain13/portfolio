"""
University of Michigan
ROB 416/516 Winter 2026
Mini Project #4, Problem 4.1 Part I
Template Version
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

FIGURE_DIR = Path(__file__).resolve().parent / "figure"

from config.config import Config
from agent.agents import Evader, Pursuer


class Simulation:
    def __init__(self):
        self.cfg = Config()
        self.evader = Evader(self.cfg)
        self.pursuer = Pursuer(self.cfg)
        self.hist = {'pE': [], 'pP': [], 't': [], 'status': 'Running'}

    def run(self):
        print("Starting Adversarial Evasion Simulation...")

        for step in range(self.cfg.max_steps):
            t = step * self.cfg.dt

            # Record
            self.hist['pE'].append(self.evader.p.copy())
            self.hist['pP'].append(self.pursuer.p.copy())
            self.hist['t'].append(t)

            # Check Status
            if self.check_status(t):
                break

            # TODO: fill in Problem 4.1(b) here via
            # agent/agents.py -> Pursuer.compute_control().
            u_P = self.pursuer.compute_control(self.evader.p, self.evader.v)

            # TODO: fill in Problem 4.1(c) here via
            # agent/agents.py -> Evader.nominal_control() and Evader.compute_control().
            u_E = self.evader.compute_control(self.pursuer.p, self.pursuer.v, u_P)

            # 4. Step
            self.evader.step(u_E)
            self.pursuer.step(u_P)

    def check_status(self, t):
        dist_goal = np.linalg.norm(self.evader.p - self.cfg.p_goal)
        dist_agents = np.linalg.norm(self.evader.p - self.pursuer.p)

        if dist_goal < self.cfg.goal_tol:
            print(f"[{t:.2f}s] SUCCESS: Evader reached the goal!")
            self.hist['status'] = 'Success'
            return True

        if dist_agents < self.cfg.R:
            print(f"[{t:.2f}s] FAILURE: Evader was caught!")
            self.hist['status'] = 'Caught'
            return True

        return False


def animate_simulation(hist, sim: Simulation):
    pE_arr = np.array(hist['pE'])
    pP_arr = np.array(hist['pP'])
    t_arr = np.array(hist['t'])

    cfg = sim.cfg
    dt = cfg.dt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.grid(False)

    ax.plot(cfg.p_goal[0], cfg.p_goal[1], 'g*', markersize=15, label='Goal')

    line_e, = ax.plot([], [], 'b-', alpha=0.5, label='Evader')
    line_p, = ax.plot([], [], 'r-', alpha=0.5, label='Pursuer')
    dot_e, = ax.plot([], [], 'bo', markersize=8)
    dot_p, = ax.plot([], [], 'ro', markersize=8)

    circle = Circle((0, 0), cfg.R, color='b', fill=False, linestyle=':')
    ax.add_patch(circle)

    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)

    all_p = np.vstack([pE_arr, pP_arr, cfg.p_goal])
    if len(all_p) > 0:
        ax.set_xlim(all_p[:, 0].min() - 1, all_p[:, 0].max() + 1)
        ax.set_ylim(all_p[:, 1].min() - 1, all_p[:, 1].max() + 1)

    def init():
        dot_e.set_data([], [])
        dot_p.set_data([], [])
        line_e.set_data([], [])
        line_p.set_data([], [])
        time_text.set_text('')
        return dot_e, dot_p, line_e, line_p, circle, time_text

    def update(frame):
        pe = pE_arr[frame]
        pp = pP_arr[frame]

        dot_e.set_data([pe[0]], [pe[1]])
        dot_p.set_data([pp[0]], [pp[1]])

        line_e.set_data(pE_arr[:frame, 0], pE_arr[:frame, 1])
        line_p.set_data(pP_arr[:frame, 0], pP_arr[:frame, 1])

        circle.center = (pe[0], pe[1])
        time_text.set_text(f"Time: {t_arr[frame]:.2f}s")

        return dot_e, dot_p, line_e, line_p, circle, time_text

    ani = animation.FuncAnimation(fig, update, frames=len(t_arr),
                                  init_func=init, blit=True, interval=30)

    plt.legend()
    ani.save(FIGURE_DIR / "simulation.gif", writer='pillow', fps=int(1 / dt))

    # Deliverable: plot h(t) with the h = 0 boundary.
    dist_sq = np.sum((pE_arr - pP_arr)**2, axis=1)
    h_val = dist_sq - cfg.R**2

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(t_arr, h_val, 'b-', label='h(x)')
    ax2.axhline(0, color='r', linestyle='-', label='h=0')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Barrier Function h(x)')
    ax2.legend()
    ax2.grid(False)
    plt.savefig(FIGURE_DIR / "h_function.png")

    # Deliverable: plot 2D trajectory plot.
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    ax3.plot(pE_arr[:, 0], pE_arr[:, 1], 'b-', label='Evader')
    ax3.plot(pP_arr[:, 0], pP_arr[:, 1], 'r-', label='Pursuer')
    ax3.plot(cfg.p_goal[0], cfg.p_goal[1], 'g*', markersize=15, label='Goal')
    ax3.plot(pE_arr[0, 0], pE_arr[0, 1], 'c*', markersize=15, label='Start (Evader)')
    ax3.plot(pP_arr[0, 0], pP_arr[0, 1], 'c*', markersize=15, label='Start (Pursuer)')
    ax3.set_xlabel('X [m]')
    ax3.set_ylabel('Y [m]')
    ax3.legend()
    ax3.grid(False)
    plt.savefig(FIGURE_DIR / "trajectory.png")

    plt.show()


if __name__ == "__main__":
    sim = Simulation()
    sim.run()
    animate_simulation(sim.hist, sim)
