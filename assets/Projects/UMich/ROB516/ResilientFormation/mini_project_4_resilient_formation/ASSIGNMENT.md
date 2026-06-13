# Multi-Robot Control with Connectivity Resilience - Assignment

## Overview
This assignment involves implementing Control Barrier Functions (CBF) for multi-robot coordination with connectivity/resilience and collision/obstacle avoidance constraints. You will complete the helper functions and control constraint implementations.

---

## Part 1: Helper Functions (`connectivity_and_resilience.py`)

### Task 1.1: Laplacian Matrix Computation
**File:** `connectivity_and_resilience.py`, 

Implement the `laplacian_matrix(A)` function:
- **Input:** Adjacency matrix `A` (N×N)
- **Output:** Laplacian matrix `L` (N×N)
- **Definition:** L = D - A, where D is the degree matrix (diagonal matrix with D[i,i] = sum of row i in A)


```python
def laplacian_matrix(A):
    # TODO: Compute Laplacian matrix from adjacency matrix A
    D = # Compute degree matrix
    L = # Compute Laplacian
    return L
```

### Task 1.2: Algebraic Connectivity (Second Smallest Eigenvalue)
**File:** `connectivity_and_resilience.py`, 

Implement the `algebraic_connectivity(L)` function:
- **Input:** Laplacian matrix `L`
- **Output:** λ₂ (second smallest eigenvalue of L)

```python
def algebraic_connectivity(L):
    # TODO: Compute the second smallest eigenvalue of Laplacian L
    lambda_2 = # Compute and return second smallest eigenvalue
    return lambda_2
```

### Task 1.3: Connectivity CBF
**File:** `connectivity_and_resilience.py`, 

Implement the `connectivity_CBF(L, epsilon)` function:
- **Input:** Laplacian matrix `L`, safety margin `epsilon`
- **Output:** h_conn

```python
def connectivity_CBF(L, epsilon):
    # TODO: Define the CBF for connectivity h_conn
    h_conn =
    return h_conn
```

### Task 1.4: F-Resilient CBF
**File:** `connectivity_and_resilience.py`, 

Implement the `f_resilient_CBF(L, F, epsilon)` function:
- **Input:** Laplacian matrix `L`, resilience parameter `F` (max number of failed agents), safety margin `epsilon`
- **Output:** h_res 

```python
def f_resilient_CBF(L, F, epsilon):
    # TODO: Define the CBF for resilience h_res
    h_res = 
    return h_res
```

### Task 1.5: Collision Avoidance CBF
**File:** `connectivity_and_resilience.py`, 

Implement the `collision_CBF(xi, xj, d_min)` function:
- **Input:** 
  - `xi`, `xj`: positions of robots i and j (2D vectors)
  - `d_min`: minimum allowed distance between robots
- **Output:** `h` (barrier function), `dh_dxi`, `dh_dxj` (gradients)


```python
def collision_CBF(xi, xj, d_min):
    # TODO: Define the CBF for inter-robot collision avoidance
    h = 
    dh_dxi = # Gradient w.r.t. xi
    dh_dxj = # Gradient w.r.t. xj
    return h, dh_dxi, dh_dxj
```

### Task 1.6: Obstacle Avoidance CBF
**File:** `connectivity_and_resilience.py`, 

Implement the `obstacle_CBF(xi, obs_center, r, d_safe)` function:
- **Input:**
  - `xi`: robot position (2D)
  - `obs_center`: obstacle center (2D)
  - `r`: obstacle radius
  - `d_safe`: safety distance from obstacle
- **Output:** `h` (barrier function), `dh_dxi` (gradient)


```python
def obstacle_CBF(xi, obs_center, r, d_safe):
    # TODO: Define the CBF for obstacle avoidance
    h = 
    dh_dxi = # Gradient w.r.t. xi
    return h, dh_dxi
```

### Task 1.7: Nominal Controller
**File:** `connectivity_and_resilience.py`, 

Implement the `nominal_controller(n, x, goals)` function:
- **Input:**
  - `n`: number of robots
  - `x`: current positions (N×2 array)
  - `goals`: goal positions (N×2 array)
- **Output:** `u_ref` (reference control inputs, N×2)


```python
def nominal_controller(n, x, goals):
    u_ref = np.zeros((n, 2))
    # TODO: Compute nominal control inputs toward goals given state x
    for i in range(n):
        # Compute vector toward goal
        # If far from goal, normalize; otherwise set to zero
    return u_ref
```

### Task 1.8: Adjacency Matrix
**File:** `connectivity_and_resilience.py`, 

Implement the `adjacency_matrix(x, R, sigma)` function:
- **Input:**
  - `x`: robot positions (N×2)
  - `R`: communication radius
  - `sigma`: bandwidth parameter
- **Output:** Adjacency matrix `A` (N×N, symmetric)


```python
def adjacency_matrix(x, R, sigma):
    N = x.shape[0]
    A = np.zeros((N, N))
    
    # TODO: Compute adjacency matrix A based on positions x, communication radius R, and sigma
    for i in range(N):
        for j in range(i+1, N):
            # Compute distance
            # If within range, compute adjacency weight
            # Ensure symmetry
    return A
```

---

## Part 2: Control Constraints Implementation

### Task 2.1: Collision Avoidance Constraints (connectivity_aware_control.py)
**File:** `connectivity_aware_control.py`, 

Add collision avoidance CBF constraints for the control optimization:
- **Location:** Inside the loop over pairs (i,j) where i < j
- **Steps:**
  1. Call `collision_CBF(robots[i], robots[j], d_min)` to get h and gradients
  2. Construct constraint: 
  3. Append to constraints list

```python
# Inter-robot collision
# TODO: Construct the CBF constraints for inter-robot collision avoidance
for i in range(n):
    for j in range(i+1, n):
        h, dh_dxi, dh_dxj =
        constraints.append()  # Add CBF constraint
        if h < 0:
            print(f"Collision warning between robot {i} and robot {j}, h={h}")
```

### Task 2.2: Obstacle Avoidance Constraints (connectivity_aware_control.py)
**File:** `connectivity_aware_control.py`, 

Add obstacle avoidance CBF constraints:
- **Location:** Inside the loop over obstacles
- **Steps:**
  1. Call `obstacle_CBF(robots[i], obs[:2], obs[2], r_s)` to get h and gradient
  2. Construct constraint: `dh_dxi @ u[i] >= -alpha_obs * h`
  3. Append to constraints list

```python
# Obstacle avoidance
# TODO: Construct the CBF constraints for obstacle avoidance
for i in range(n):
    for obs in obstacles:
        h, dh_dxi = # Call obstacle_CBF
        constraints.append()  # Add CBF constraint
        if h < 0:
            print(f"Obstacle avoidance warning for robot {i} and obstacle at {obs[:2]}, h={h}")
```

### Task 2.3: Collision Avoidance Constraints (resilience_aware_control.py)
**File:** `resilience_aware_control.py`,

Repeat Task 2.1 for resilience-aware control file:
```python
# Inter-robot collision
# TODO: Construct the CBF constraints for inter-robot collision avoidance
for i in range(n):
    for j in range(i+1, n):
        h, dh_dxi, dh_dxj = # Call collision_CBF
        constraints.append()  # Add CBF constraint
        if h < 0:
            print(f"Collision warning between robot {i} and robot {j}, h={h}")
```

### Task 2.4: Obstacle Avoidance Constraints (resilience_aware_control.py)
**File:** `resilience_aware_control.py`,

Repeat Task 2.2 for resilience-aware control file:
```python
# Obstacle avoidance
# TODO: Construct the CBF constraints for obstacle avoidance
for i in range(n):
    for obs in obstacles:
        h, dh_dxi = # Call obstacle_CBF
        constraints.append()  # Add CBF constraint
        if h < 0:
            print(f"Obstacle avoidance warning for robot {i} and obstacle at {obs[:2]}, h={h}")
```

---

## Testing & Validation

After implementing all tasks, run the control scripts to verify:

```bash
python connectivity_aware_control.py
python resilience_aware_control.py
```

### Expected Results:
1. **Robots should reach goals** without colliding with each other or obstacles
2. **h_conn plot** should stay positive 
3. **h_res plot** should stay positive
4. **Console output** should show evolution of h values over time

