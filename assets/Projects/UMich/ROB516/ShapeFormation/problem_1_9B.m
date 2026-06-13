function E = undirected_complete_graph(n)
E = [];
    for Vi=1:n
        for Vj=Vi+1:n
            E = [E; Vi Vj];
        end
    end
end

function E = undirected_path_graph(n)
E = zeros(0,2);
    for Vi=1:n-1
        Vj = Vi + 1;
        E = [E; Vi Vj];
    end
end

function D = degree_matrix(n, E)
    vertex_degrees = zeros(1,n);
    for Vi=1:n
        vertex_degrees(Vi) = sum(E(:,1) == Vi | E(:,2) == Vi);
    end
    D = diag(vertex_degrees);
end

function A = adjacency_matrix(n, E)
    A = zeros(n);
    for Vi=1:n
        for Vj=1:n
            if ismember([Vi Vj], E, 'rows') || ismember([Vj Vi], E, 'rows')
                A(Vi, Vj) = 1;
            else
                A(Vi, Vj) = 0;
            end
        end
    end
end

function L = laplacian_matrix(n, E)
    L = degree_matrix(n, E) - adjacency_matrix(n, E);
end

function L = directed_path_laplacian(n)
I = zeros(n, n-1);
for i = 1:n-1
    I(i, i) = -1;
    I(i+1, i) = 1;
end
L = I * I';
end

function problem_1_9B_1(n, L, graph_type, movie_speed)


% Initial variables
% n = 20;
maxsteps = 1000000; % You can change this
dt = 0.01;
epsilon = 0.01;

% State vector
% The first axis represents agent number.
% The second axis holds the (x,y) positions for each agent.
% The third axis represents time step.
% E.g. x[1,2,tt] is the y positions for agent 1 at time tt

x = zeros(n,2,maxsteps);
x(:,:,1) = -30 + 60*rand(n,2);

% xi vector
% Same structure as the state vector, except there is no time axis since
%   xi is constant.
R = 20;
theta = linspace(0, 2*pi, n+1); theta(end) = [];
xi = zeros(n,2);
xi(:,1) = R*cos(theta)';
xi(:,2) = R*sin(theta)';

% --- Generate your Laplacian matrices
% 1) Undirected Complete Graph
% E = undirected_complete_graph(n);
% L = laplacian_matrix(n,E);

% 2) Undirected Path Graph
% E = undirected_path_graph(n);
% L = laplacian_matrix(n,E);

% 3) Directed Path Graph
% L = directed_path_laplacian(n);


% --- Simulate the dynamics ---
e = zeros(1,maxsteps);
final_timestep = maxsteps;

for tt = 2:maxsteps
    x(:,1,tt) = x(:,1,tt-1) - dt*(L*x(:,1,tt-1) - L*xi(:,1));
    x(:,2,tt) = x(:,2,tt-1) - dt*(L*x(:,2,tt-1) - L*xi(:,2));
    
    displacement = x(:,:,tt) - xi;
    error_vector = displacement - mean(displacement, 1);
    e(tt) = max(max(abs(error_vector)));
    % disp(x(:,:,tt));
    % disp(e(tt));

    if e(tt) <= epsilon
        final_timestep = tt;
        fprintf('Consensus reached at t = %.2f seconds\n', final_timestep*dt);
        break;
    end
end

x = x(:,:,1:final_timestep);


% --- Create the Plot for part A here ---
figure;
plot(xi(:,1), xi(:,2), 'rx','MarkerSize',10,'LineWidth',2); hold on;
plot(x(:,1,end), x(:,2,end),'bo','MarkerSize',6,'LineWidth',1.5);
legend('Target','Robots');
axis equal; grid on;
xlabel('X (m)'); ylabel('Y (m)');
title('Final Circular Formation ('+graph_type+')');



% --- Use the make_movie function to make your movie ---

% Call make_movie(x) where x is your state vector. This will create an .avi
% movie in the folder where this code is.
% To change the movie name, change the variable "Title" in the
% make_movie(x) function.

make_movie(x, ['video_1_9_B_' char(graph_type) '.mp4'], movie_speed);
% make_movie(x, "video_1_9_B_undirected_complete.mp4");
% make_movie(x, "video_1_9_B_undirected_path.mp4");
% make_movie(x, "video_1_9_B_directed_path.mp4");


end




% --- Movie function ---

function make_movie(x, name, movie_speed)

maxsteps = size(x,3);
n = size(x,1);

% Make the movie object
mov = struct('cdata', [], 'colormap', []);

% Change "movie" to the name of your .avi movie
Title = name;
vidObj = VideoWriter(Title, "MPEG-4");

% Frame rate. Keep this at 24 if you don't care.
FR = 24;
vidObj.FrameRate = FR;
open(vidObj)

figure;
clf

% The first two numbers in the "position" matrix are the (x,y) position of the lower
% left corner of the plot window. The last two numbers are the (x,y) position
% of the upper right corner.
position = [100 50 700 700];
set(gcf, 'Position', position)

hold on
for ii=1:1:n
    plot(x(ii,1,1), x(ii,2,1),'ko');
end
hold off
drawnow update
mov= getframe(gcf);

% The next few lines make the first frame of the movie play for 2 seconds
first_frame_pause_secs = 2;
for k=1:1:first_frame_pause_secs*FR
    writeVideo(vidObj,mov)
end

% This main loop draws the plots for each remaining time step and saves
%   each plot to a frame of the movie.

% If the movie is going too slow, you can increase the "stride" value, e.g.
%   put "for j = 1:10:maxsteps" or some other number in the middle.
for jj=1:movie_speed:maxsteps
    
    clf
    set(gcf, 'Position', position)
    hold on
    
    for ii=1:1:n
        plot(x(ii,1,jj), x(ii,2,jj),'ko');
    end
    
    hold off
    
    drawnow update
    mov= getframe(gcf);
    writeVideo(vidObj,mov)
    clear mov
    mov = struct('cdata', [], 'colormap', []);
end
%Close the file when you're done
close(vidObj)

end

graphs = ["undirected_complete", "undirected_path", "directed_path"];
n = 20;
movie_speed = [1, 30, 30];

for graph=1:length(graphs)
    graph_type = graphs(graph);

    switch graph_type
        case "undirected_complete"
            E = undirected_complete_graph(n);
            L = laplacian_matrix(n, E);
        case "undirected_path"
            E = undirected_path_graph(n);
            L = laplacian_matrix(n, E);
        case "directed_path"
            L = directed_path_laplacian(n);
    end

    problem_1_9B_1(n, L, graph_type, movie_speed(graph));
end