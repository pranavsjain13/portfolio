import numpy as np

# Robot initial positions
robots = np.array([
    [-0.4, -1.0],
    [ 0.0, -1.0],
    [-0.2, -2.0],
    [ 0.6, -1.2],
    [-0.5, -1.3],
    [-0.4, -1.6],
    [ 0.5, -1.7],
    [ 0.3, -0.8],
])

n = len(robots)
R = 2.0           # communication radius
epsilon = 0.01    # small margin for CBF
r_s = 0.15      # radius of agents
d_min = 2*r_s       # inter-robot minimum distance
# Obstacles: list of [x, y, radius]
obstacles = []
x1 = -1.1
x2 = 1.1 
radius = 0.6
y_s = 0
y_increment = 0.3
for i in range(int( 5/y_increment )):
    obstacles.append( (x1,y_s,radius) ) # x,y,radius, ax, id
    obstacles.append( (x2,y_s,radius) )
    y_s = y_s + y_increment

obstacles.append((-0.6,0.5,radius))
obstacles.append(( 0.6,2.5,radius))
obstacles.append(( 0.0,4.0,.2))


# Goals for the robots
goals = np.array([
    [0, 5],
    [0, 5],
    [0, 5],
    [0, 5],
    [0, 5],
    [0, 5],
    [0, 5],
    [0, 5],
])

# discrete time step and number of steps of simulation
dt = 0.05

# adjust accodingly for the video
num_steps = 350
