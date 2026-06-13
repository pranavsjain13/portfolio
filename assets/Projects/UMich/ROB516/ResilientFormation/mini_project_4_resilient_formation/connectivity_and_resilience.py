import numpy as np
from scipy.linalg import eigh
import cvxpy as cp
from numpy.linalg import norm

"""
Helper functions to compute adjacency matrix, Laplacian, its derivatives, and μ_m for connectivity CBF
"""

def laplacian_matrix(A):
    # TODO: Compute Laplacian matrix from adjacency matrix A
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    return L

def algebraic_connectivity(L):
    # TODO: Compute the second smallest eigenvalue of Laplacian L
    eigenvalues, _ = eigh(L)
    lambda_2 = eigenvalues[1]

    return lambda_2

def connectivity_CBF(L, epsilon):
    # TODO: Define the CBF for connectivity h_conn
    lambda_2 = algebraic_connectivity(L)
    h_conn = lambda_2 - epsilon
    return h_conn

def f_resilient_CBF(L, F, epsilon):
    # TODO: Define the CBF for resilience h_res
    lambda_2 = algebraic_connectivity(L)
    h_res = lambda_2 - (4.0 + epsilon)
    return h_res

def collision_CBF(xi, xj, d_min):
    # TODO: Define the CBF for inter-robot collision avoidance
    diff = xi - xj
    h = np.dot(diff, diff) - d_min**2
    dh_dxi = 2.0 * diff
    dh_dxj = -2.0 * diff
    return h, dh_dxi, dh_dxj

def obstacle_CBF(xi, obs_center, r, d_safe):
    # TODO: Define the CBF for obstacle avoidance
    diff = xi - obs_center
    h = np.dot(diff, diff) - (r + d_safe)**2
    dh_dxi = 2.0 * diff
    return h, dh_dxi

def nominal_controller(n, x, goals):
    u_ref = np.zeros((n, 2))
    # TODO: Compute nominal control inputs toward goals given state x
    for i in range(n):
        diff = goals[i] - x[i]
        dist = np.linalg.norm(diff)
        if dist > 1e-4:
            u_ref[i] = diff / dist  # Desired speed of 1 m/s
        else:
            u_ref[i] = np.zeros(2)

    return u_ref

def adjacency_matrix(x, R, sigma):
    N = x.shape[0]
    A = np.zeros((N, N))

    # TODO:Compute adjacency matrix A based on positions x, communication radius R, and sigma
    for i in range(N):
        for j in range(i+1, N):
            dist = np.linalg.norm(x[i] - x[j])
            if dist <= R:
                val = np.exp((R**2 - dist**2)**2 / sigma) - 1.0
                A[i, j] = val
                A[j, i] = val
            
    return A


def compute_sigma(x, R):
    """
    Compute sigma so that A_ij <= 1 for all i,j neighbors
    """
    N = x.shape[0]
    g_max = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(x[i] - x[j])
            if d <= R:
                g = (R**2 - d**2)**2
                if g > g_max:
                    g_max = g
    sigma = g_max / np.log(2)
    return sigma

def dL_dx(x, R, sigma):
    N = x.shape[0]
    dL = [np.zeros((N,N)) for _ in range(2*N)]
    
    for i in range(N):
        for j in range(i+1, N):
            diff = x[i] - x[j]
            d = np.linalg.norm(diff)
            if d > R or d < 1e-8:
                continue
            g = (R**2 - d**2)**2
            exp_term = np.exp(g / sigma)
            da_dxi = -(4.0 / sigma)*(R**2 - d**2)*exp_term*diff
            da_dxj = -da_dxi

            # Update diagonal entries
            dL[i*2][i, i] += da_dxi[0]
            dL[i*2+1][i, i] += da_dxi[1]
            dL[j*2][j, j] += da_dxj[0]
            dL[j*2+1][j, j] += da_dxj[1]

            # Update off-diagonal entries
            for k in range(2):
                dL[i*2+k][i, j] = -da_dxi[k]
                dL[j*2+k][j, i] = -da_dxj[k]

    return dL

def mu_m(x, u, R, sigma, m=2):
    """
    Computes μ_m(x,u) for F-resilient CBF with CVXPY variable u
    Returns lambda_2 (float) and mu (CVXPY expression)
    """
    N = x.shape[0]
    A = adjacency_matrix(x, R, sigma)
    D = np.diag(A.sum(axis=1))
    L = D - A
    
    eigvals, eigvecs = eigh(L)
    Vm = eigvecs[:, 1:m+1]

    dL = dL_dx(x, R, sigma)

    # Build Delta_L as a CVXPY expression
    Delta_L_expr = 0
    for i in range(N):
        # Convert dL arrays to CVXPY constants
        dL_xi = cp.Constant(dL[2*i])
        dL_yi = cp.Constant(dL[2*i+1])
        Delta_L_expr += dL_xi * u[i,0] + dL_yi * u[i,1]

    # Project onto eigenvectors Vm
    mu_vals = []
    for v in Vm.T:
        v = v.reshape(-1,1)  # ensure column vector
        mu_vals.append(cp.quad_form(v, Delta_L_expr))  # v^T * Delta_L * v

    # μ_m is minimum over projections
    mu = cp.minimum(*mu_vals)

    return mu