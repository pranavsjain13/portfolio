"""
University of Michigan
ROB 416/516 Winter 2026
Mini Project #4, Problem 4.1 Part II
Template Version
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

FIGURE_DIR = Path(__file__).resolve().parent / "figure"

from config.config import Config
from agent.agents import Pursuer
from agent.formation import FormationEvader


def assign_pairs(pursuers, evaders):
    """
    Greedy assignment without replacement.
    1. Compute all pairwise distances.
    2. Find the global minimum distance pair (P_i, E_j).
    3. Assign P_i -> E_j.
    4. Remove P_i and E_j from consideration.
    5. Repeat.

    This matches the Part II fixed t=0 assignment used in miniproj4_prob1_2.py.
    """
    n_p = len(pursuers)
    n_e = len(evaders)

    all_pairs = []
    for i in range(n_p):
        for j in range(n_e):
            dist = np.linalg.norm(pursuers[i].p - evaders[j].p)
            all_pairs.append((dist, i, j))

    all_pairs.sort(key=lambda x: x[0])

    assignments = [-1] * n_p
    assigned_p = set()
    assigned_e = set()

    for dist, p_idx, e_idx in all_pairs:
        if p_idx not in assigned_p and e_idx not in assigned_e:
            assignments[p_idx] = e_idx
            assigned_p.add(p_idx)
            assigned_e.add(e_idx)

    return assignments


class MultiAgentSimulation:
    def __init__(self):
        self.cfg = Config()
        self.cfg.max_steps = 1000

        # Evaders
        self.evaders = []
        N_e = self.cfg.num_evaders
        for i in range(N_e):
            angle = i * 2 * np.pi / N_e
            c_ev = self.cfg.evader_init_center
            r_ev = self.cfg.evader_init_radius
            pos = np.array([c_ev[0] + r_ev * np.cos(angle), c_ev[1] + r_ev * np.sin(angle), 0, 0])
            self.evaders.append(FormationEvader(pos, self.cfg, i))

        # Pursuers
        self.pursuers = []
        N_p = self.cfg.num_pursuers

        for i in range(N_p):
            angle = i * 2 * np.pi / N_p
            c_pur = self.cfg.pursuer_init_center
            r_pur = self.cfg.pursuer_init_radius
            pos = np.array([c_pur[0] + r_pur * np.cos(angle), c_pur[1] + r_pur * np.sin(angle), 0, 0])
            p = Pursuer(self.cfg)
            p.state = pos.astype(float)
            self.pursuers.append(p)

        self.hist = {'evaders': [[] for _ in range(N_e)],
                     'pursuers': [[] for _ in range(N_p)],
                     't': []}

    def run(self):
        print("Starting Multi-Agent Formation Escape Simulation...")

        assignments = assign_pairs(self.pursuers, self.evaders)

        print(f"Assignments (Pursuer -> Evader idx): {assignments}")

        for step in range(self.cfg.max_steps):
            t = step * self.cfg.dt
            self.hist['t'].append(t)

            # Store history
            for i, e in enumerate(self.evaders):
                self.hist['evaders'][i].append(e.p.copy())
            for i, p in enumerate(self.pursuers):
                self.hist['pursuers'][i].append(p.p.copy())

            # TODO: fill in the worst-case pursuer policy in
            # agent/agents.py -> Pursuer.compute_control().
            p_actions = []
            for i, p in enumerate(self.pursuers):
                target_idx = assignments[i]
                target = self.evaders[target_idx]
                u_p = p.compute_control(target.p, target.v)
                p_actions.append(u_p)

            # TODO: fill in the multi-constraint formation QP in
            # agent/formation.py -> FormationEvader.compute_control().
            e_actions = []
            for e in self.evaders:
                u_e = e.compute_control(t, self.pursuers, p_actions)
                e_actions.append(u_e)

            # 3. step
            for i, p in enumerate(self.pursuers):
                p.step(p_actions[i])
            for i, e in enumerate(self.evaders):
                e.step(e_actions[i])

            if step % 100 == 0:
                print(f"Step {step}/{self.cfg.max_steps}")


def animate_multi(hist, sim):
    t_arr = np.array(hist['t'])
    dt = sim.cfg.dt

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_xlim(-5, 18)
    ax.set_ylim(-5, 18)

    N_e = sim.cfg.num_evaders
    N_p = sim.cfg.num_pursuers

    lines_e = [ax.plot([], [], 'b-', alpha=0.3)[0] for _ in range(N_e)]
    lines_p = [ax.plot([], [], 'r-', alpha=0.3)[0] for _ in range(N_p)]

    dots_e = [ax.plot([], [], 'bo')[0] for _ in range(N_e)]
    dots_p = [ax.plot([], [], 'ro')[0] for _ in range(N_p)]

    ghosts = [ax.plot([], [], 'gx', alpha=0.5)[0] for _ in range(N_e)]

    circles = [Circle((0, 0), sim.cfg.R, color='b', fill=False, linestyle=':', alpha=0.2) for _ in range(N_e)]
    for c in circles:
        ax.add_patch(c)

    texts_e = [ax.text(0, 0, f"E{i}", fontsize=9, color='blue') for i in range(N_e)]

    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)

    def update(frame):
        time_text.set_text(f"Time: {t_arr[frame]:.2f}s")
        current_t = t_arr[frame]

        formation_center = sim.cfg.center_start + sim.cfg.center_vel * current_t

        for i in range(N_e):
            pe = hist['evaders'][i][frame]
            dots_e[i].set_data([pe[0]], [pe[1]])

            texts_e[i].set_position((pe[0], pe[1]))

            start_f = max(0, frame - 50)
            hist_e = np.array(hist['evaders'][i])
            lines_e[i].set_data(hist_e[start_f:frame, 0], hist_e[start_f:frame, 1])

            circles[i].center = (pe[0], pe[1])

            theta = 2 * np.pi * i / N_e
            offset = np.array([sim.cfg.formation_radius * np.cos(theta), sim.cfg.formation_radius * np.sin(theta)])
            ghost_pos = formation_center + offset
            ghosts[i].set_data([ghost_pos[0]], [ghost_pos[1]])

        for i in range(N_p):
            pp = hist['pursuers'][i][frame]
            dots_p[i].set_data([pp[0]], [pp[1]])

            start_f = max(0, frame - 50)
            hist_p = np.array(hist['pursuers'][i])
            lines_p[i].set_data(hist_p[start_f:frame, 0], hist_p[start_f:frame, 1])

        return lines_e + lines_p + dots_e + dots_p + ghosts + circles + [time_text] + texts_e

    ani = animation.FuncAnimation(fig, update, frames=len(t_arr), interval=30, blit=True)

    ani.save(FIGURE_DIR / "multi_agent_simulation.gif", writer='pillow', fps=int(1 / dt))

    fig_h, axes_h = plt.subplots(N_e, 1, figsize=(8, 4 * N_e), sharex=True)
    if N_e == 1:
        axes_h = [axes_h]

    for i in range(N_e):
        evader_hist = np.array(hist['evaders'][i])
        for j in range(N_p):
            pursuer_hist = np.array(hist['pursuers'][j])
            dist_sq = np.sum((evader_hist - pursuer_hist)**2, axis=1)
            h_val = dist_sq - sim.cfg.R**2
            axes_h[i].plot(t_arr, h_val, label=f'Pursuer {j}')

        axes_h[i].axhline(0, color='r', linestyle='--', label='h=0')
        axes_h[i].set_ylabel(f'h(x) Evader {i}')
        axes_h[i].legend()

    axes_h[-1].set_xlabel('Time [s]')
    fig_h.tight_layout()
    fig_h.savefig(FIGURE_DIR / "multi_agent_h_function.png")

    # Plot  2D trajectory plot with the formation circle or desired positions overlaid. Use different colors for pursuers and evaders. Add legends, titles, and axis labels for clarity.
    fig_traj, ax_traj = plt.subplots(figsize=(8, 8))
    ax_traj.set_aspect('equal')
    
    # 1. Plot Evader Trajectories (Blue)
    for i in range(N_e):
        # Convert history list to numpy array for slicing
        ev_pos = np.array(hist['evaders'][i])
        ax_traj.plot(ev_pos[:, 0], ev_pos[:, 1], 'b-', alpha=0.4, 
                    label=f'Evader Paths' if i == 0 else "")
        # Mark the final position
        ax_traj.plot(ev_pos[-1, 0], ev_pos[-1, 1], 'bo', markersize=6)
        
    # 2. Plot Pursuer Trajectories (Red)
    for i in range(N_p):
        pur_pos = np.array(hist['pursuers'][i])
        ax_traj.plot(pur_pos[:, 0], pur_pos[:, 1], 'r-', alpha=0.4, 
                    label=f'Pursuer Paths' if i == 0 else "")
        # Mark the final position
        ax_traj.plot(pur_pos[-1, 0], pur_pos[-1, 1], 'ro', markersize=6)

    # 3. Overlay Final Formation Circle and Desired Positions
    final_t = t_arr[-1]
    final_center = sim.cfg.center_start + sim.cfg.center_vel * final_t #
    
    # Draw the target formation circle
    formation_circle = Circle(final_center, sim.cfg.formation_radius, 
                              color='green', fill=False, linestyle='--', 
                              linewidth=1.5, label='Target Formation')
    ax_traj.add_patch(formation_circle)
    
    # Draw individual goal points (Ghosts)
    for i in range(N_e):
        theta = 2 * np.pi * i / N_e
        offset = np.array([sim.cfg.formation_radius * np.cos(theta), 
                          sim.cfg.formation_radius * np.sin(theta)])
        goal_pos = final_center + offset
        ax_traj.plot(goal_pos[0], goal_pos[1], 'gx', markersize=8, 
                    label='Desired Positions' if i == 0 else "")

    # 4. Formatting
    ax_traj.set_xlabel('X Position [m]', fontsize=12)
    ax_traj.set_ylabel('Y Position [m]', fontsize=12)
    ax_traj.set_title('Problem 4.1 Part II: Multi-Agent Trajectories', fontsize=14)
    ax_traj.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax_traj.grid(True, linestyle=':', alpha=0.6)
    
    # Save the figure
    fig_traj.tight_layout()
    fig_traj.savefig(FIGURE_DIR / "multi_agent_trajectories.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    sim = MultiAgentSimulation()
    sim.run()
    animate_multi(sim.hist, sim)
