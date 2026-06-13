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
v = VideoWriter('problem_2.5_part_a_scale_invariant_formation');
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
dt = 0.1;
threshold = 0.2;
K_gain = 0.5;
D_margin = 0.5;

% Simulation loop
while (norm(center - target) > threshold)
    % TODO: Reference control to move toward the target
    center_velocity = K_gain * (target - center);
    center = center +  center_velocity*dt;

    alpha_safe = 2;

    % Agents' control to maintain formation
    % future_collision = check_collision(positions +  center_velocity*dt, obstacle, obstacle_radius);
    % if future_collision
    %     % TODO: if there will be a collision, adjust alpha accordingly
    % 
    % else
    %     % TODO: if there will be no collision, adjust alpha accordingly
    %     % (you can choose to do nothing, or you can try to expand back to
    %     % alpha=2).
    % end
    for i = 1:N
        di = unit_offset(i, :);
        for j = 1:size(obstacle, 1)
            oj = obstacle(j, :);
            rj = obstacle_radius(j);

            B = 2 * dot(di, (center - oj));
            C = norm(center - oj)^2 - (rj + D_margin)^2;

            poly_roots = roots([1, B, C]);
            real_roots = poly_roots(imag(poly_roots) == 0);
            
            for r = 1:length(real_roots)
                if real_roots(r) > 0 && real_roots(r) < alpha_safe
                    alpha_safe = real_roots(r);
                end
            end
        end
    end

    alpha = max(0.1, alpha_safe);

    % Update the positions of robots
    for i = 1:N
        positions(i, :)  = center +  alpha*unit_offset(i,:);
    end



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