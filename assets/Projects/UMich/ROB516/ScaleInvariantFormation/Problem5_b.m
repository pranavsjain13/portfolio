clc; clear; close all;

% Parameters
N = 5;           
unit_offset = zeros(5,2);  
for i=1:5
    unit_offset(i,1) =  cosd(72*i);
    unit_offset(i,2) =  sind(72*i);
end

% Relative distance to the center
alpha = 2;
formation_offset =alpha*unit_offset; 

% Target position for the center
target = [20, 10];       
% Obstacle positions and radii
obstacle = [10, 1; 15, 11;5,-4;15,2;5,6]; 
obstacle_radius = [1,1,3,2,1.5];     


first_figure = true;
collision = false;

% Initialize positions: center at origin, robots in formation
positions = zeros(5,2);
for i=1:5
    positions(i,1) = formation_offset(i,1);
    positions(i,2) = formation_offset(i,2);
end
center = mean(positions);

% Set up video writer
v = VideoWriter('problem_2.5_part_b_scale_invariant_formation');
v.FrameRate = 10;  % Set frames per second
open(v);  % Open video file

% Plot settings
first_figure_handle = figure;
xlim([-5, 25]); ylim([-5, 15]);
axis manual;
hold on;

% Plot obstacles
for i = 1:size(obstacle, 1)
    viscircles(obstacle(i, :), obstacle_radius(i), 'LineStyle', '--');
end

% Time step (welcome to change)
R_min = 1.2;      % Minimum distance between robots
D_safe = 0.5;     % Distance from obstacles (radius + margin)
K_goal = 0.4;     % Gain for target
K_avoid = 50.0;    % Gain for obstacle avoidance
K_inter = 10.0;    % Gain for inter-robot separation
dt = 0.1;
threshold = 0.5;
R_robot = 0.4;
delta = 0.1;

% Simulation loop
while (norm(center - target) > threshold)
    % TODO: Reference control to move toward the target
    new_velocities = zeros(N, 2);
    
    for i = 1:N
        % --- Force 1: Goal Attraction ---
        v_slot = (center + alpha * unit_offset(i,:)) - positions(i,:);
        v_target = target - center;
        % We weight the target pull so the formation actually moves forward
        f_goal = K_goal * (v_slot + 0.8 * (v_target / norm(v_target))); 

        % --- Force 2: Obstacle Repulsion ---
        f_obs = [0, 0];
        for j = 1:size(obstacle, 1)
            vec_to_obs = positions(i, :) - obstacle(j, :);
            dist = norm(vec_to_obs);
            d_safe = obstacle_radius(j) + R_robot + delta;
            
            if dist < d_safe
                % Use K_avoid (not ko)
                % Formula creates a "wall" of force that grows as dist -> 0
                f_obs = f_obs + K_avoid * (1/dist - 1/d_safe) * (vec_to_obs / dist^3);
            end
        end

        % --- Force 3: Inter-Robot Repulsion ---
        f_inter = [0, 0];
        for k = 1:N
            if i == k, continue; end
            vec_to_robot = positions(i, :) - positions(k, :);
            dist_rk = norm(vec_to_robot);
            
            if dist_rk < R_min
                % Use K_inter (not ki)
                f_inter = f_inter + K_inter * (1/dist_rk - 1/R_min) * (vec_to_robot / dist_rk^3);
            end
        end

        % Total Velocity Command
        new_velocities(i, :) = f_goal + f_obs + f_inter;
    end
    
    % Update positions
    positions = positions + new_velocities * dt;
    center = mean(positions);


    % Plot robots' positions
    clf; hold on;
    axis([-5 25 -5 15])
    plot(target(1), target(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2); % Target

    % Plot obstacles
    for i = 1:size(obstacle, 1)
        viscircles(obstacle(i, :), obstacle_radius(i), 'LineStyle', '--'); % Obstacles
    end

    % Center
    plot(center(1, 1), center(1, 2),'bx');
    % Followers
    viscircles(positions(1:5, :), 0.3, 'Color','g');
    % Draw edges
    plot(positions(1:5, 1), positions(1:5, 2),'k-', 'MarkerSize', 10,'LineWidth',2); 
    plot([positions(1, 1), positions(5, 1)], [positions(1, 2), positions(5, 2)],'k-', 'MarkerSize', 10,'LineWidth',2); 

    % Capture frame for video
    frame = getframe(gcf);
    if first_figure
        intial_position = get(first_figure_handle, 'Position');
        first_figure = false;
    end
    set(gcf, 'Position', [100, 100, intial_position(3), intial_position(4)]);
    writeVideo(v, frame);  % Write frame to video file
    
    % Pause to visualize
    pause(0.1);

    % Check for a collision
    collision = check_collision(positions, obstacle, obstacle_radius);
    if collision
        break
    end
end

% Close the video file
close(v);

if collision
    disp("Collision!");
else
    disp("No collision!");
end

% Check if there is any collision with the given positions of robots and
% obstacles
function yes_no = check_collision(positions, obstacle, obstacle_radius)
    yes_no = false;
    for i=1:size(positions, 1)
        for j=1:size(obstacle, 1)
            if norm(positions(i,:)-obstacle(j,:))<obstacle_radius(j)+0.4
                yes_no = true;
            end
        end
        if yes_no
            break
        end
    end
end