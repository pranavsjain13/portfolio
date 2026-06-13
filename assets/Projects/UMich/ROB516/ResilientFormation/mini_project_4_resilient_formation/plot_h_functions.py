import matplotlib.pyplot as plt
import numpy as np

'''
Helper function to plot h function evolution over time
'''

def plotter(time_steps, h_history, threshold, title):
    # -------------------- Plot h function evolution --------------------
    fig_h, ax_h = plt.subplots(figsize=(12, 6))
    ax_h.plot(time_steps, h_history, 'g-', linewidth=2, label='h(x)')
    ax_h.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label='Safety threshold')
    ax_h.fill_between(time_steps, 0, h_history, where=(np.array(h_history) >= 0), 
                    alpha=0.3, color='green', label='Safe region')
    ax_h.fill_between(time_steps, 0, h_history, where=(np.array(h_history) < 0), 
                    alpha=0.3, color='red', label='Unsafe region')
    ax_h.set_xlabel('Time (seconds)', fontsize=12)
    ax_h.set_ylabel('h(x)', fontsize=12)
    ax_h.set_title(title, fontsize=13)
    ax_h.grid(True, alpha=0.3)
    ax_h.legend(fontsize=11, loc='best')
    plt.tight_layout()
    plt.savefig('h_res_evolution.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as 'h_res_evolution.png'")
    print(f"\nStatistics for h_res:")
    print(f"  Min: {np.min(h_history):.4f}")
    print(f"  Max: {np.max(h_history):.4f}")
    print(f"  Mean: {np.mean(h_history):.4f}")
    print(f"  Final value: {h_history[-1]:.4f}")
    plt.show()