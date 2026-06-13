import numpy as np
from dataclasses import dataclass, field

@dataclass
class Config:
    dt: float = 0.02
    max_steps: int = 1000
    
    # Limits
    uE_max: float = 1.0
    uP_max: float = 0.8
    vE_max: float = 1.5
    vP_max: float = 1.2
    
    # Pursuer PD
    kp_P: float = 2.0
    kd_P: float = 2.0 * np.sqrt(kp_P)

    # Evader nominal PD
    kp_E: float = 2.0
    kd_E: float = 2.0 * np.sqrt(kp_E)
    
    # Safety CBF
    R: float = 1.0
    gamma1: float = 2.0
    gamma2: float = 2.0
    
    #------- Problem 1.1 Parameters -------#

    # Scenario
    p_goal: np.ndarray = field(default_factory=lambda: np.array([5.0, 5.0]))
    goal_tol: float = 0.2
    
    x_E_init: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 0.0]))
    x_P_init: np.ndarray = field(default_factory=lambda: np.array([5.0, 0.0, 0.0, 0.0]))
    
    #------- Problem 1.2 Parameters -------#

    # Formation Config  
    center_start: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    center_vel: np.ndarray = field(default_factory=lambda: np.array([0.6, 0.4]))
    formation_radius: float = 3.0
    
    # Init pos for formation 
    evader_init_center: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    evader_init_radius: float = 2.0
    
    pursuer_init_center: np.ndarray = field(default_factory=lambda: np.array([10.0, 5.0]))
    pursuer_init_radius: float = 5.0

    # Multi-Agent Counts
    num_evaders: int = 4
    num_pursuers: int = 2

