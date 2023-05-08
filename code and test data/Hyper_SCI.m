function [new_vertex,A,S,time] = Hyper_SCI(data,vertex,C,means,adjust_factor)


% Start the timer
t0 = clock;

% Get the number of vertices and data dimensions
vertex_num = size(vertex,2);
dims = size(data);
n= dims(2);


% Initialize border vector
border_vector = zeros(dims(1),vertex_num*(vertex_num-1)/2);
bi_tilde = [];
for i = 1:vertex_num
    bi_tilde = [bi_tilde,find_hyper_norm(data(:,vertex),i)];
end

count = 1;

% Determine the vectors between purest pixels (borders)
for i = 1:vertex_num-1
    for j = i+1:vertex_num
        %finding border_vector
        border_vector(:,count) = data(:,vertex(1,i))-data(:,vertex(1,j));
        count = count+1;
    end
end

% Determine radius (r) of the region
R = border_vector.'*border_vector;
diagonal_elements = diag(R);
r = 0.5*sqrt(min(diagonal_elements));

% Find points inside the region
inside_circles = cell(1, vertex_num);
for i = 1:vertex_num
    distance = zeros(1,size(data,2));
    for j = 1:size(data,2)
        point = data(:, j);
        vec_to_vertex = point - data(:,vertex(1,i));
        distance(j) = norm(vec_to_vertex);
    end
    inside_circle = (distance <= r);
    inside_circles{i} = data(:, inside_circle);
end

p = zeros(dims(1),vertex_num-1);

count = 1;
%obtaining p (p12,p13...p21,p23...p(n)(n-1))
b_hat = []; % b_hat(:,1) is the hyperplane without vertex 1
h_hat = []; % h_hat is the distance of b_hat
for i = 1:vertex_num
    p=[];
    for j = 1:vertex_num
        if i == j
            continue;
        end
        [max_in_p, idx_in_p] = max(bi_tilde(:,i).'*inside_circles{j});
        p = [p,inside_circles{j}(:,idx_in_p)];
    end
    
    p = [zeros(size(data,1),1),p];
    b_hat = [b_hat,find_hyper_norm(p,1)];
    % Redirect b_hat
    if b_hat(:,i)'*data(:,vertex(i)) >0
        b_hat(:,i)= -b_hat(:,i);
    end
    p(:,1) = [];
    h = max(b_hat(:,i).'*data);
    h_hat = [h_hat,h];
end


%finding new vertex
new_vertex = zeros(dims(1),vertex_num);
for i = 1:vertex_num
    coef_matrix = [];
    cost_vector = [];
    for j = 1:vertex_num
        if i == j
            continue;
        end
        % Matrix and vector construction for calculating new vertex
        coef_matrix = [coef_matrix;b_hat(:,j).'];
        cost_vector = [cost_vector;h_hat(:,j)];
        
    end
    % Calculate new vertex
    new_vertex(:,i) = inv(coef_matrix)*cost_vector;
end

% bring hyperplanes closer to the center of data cloud
VV = C*new_vertex;
UU = means*ones(1,size(data,1)+1);
c_apostrophe = max( 1 , max(max( (-VV) ./ UU)) );
c = c_apostrophe/adjust_factor;
h_hat = h_hat/c;
new_vertex = new_vertex/c;

% Build matrix A(endmember matrix)
A = C*new_vertex+means;

% build matrix S
S=[];
for i = 1:vertex_num 
    %point which is not used
    hy_vertex = new_vertex;
    hy_vertex(:,i)=[];
    hy_vec = [];
    for col = 2:size(hy_vertex,2)
        temp_vec = hy_vertex(:,col)-hy_vertex(:,1);
        hy_vec = [hy_vec;temp_vec.'];
    end
    hy_nor = null(hy_vec);
    %dist form alpha_i to hyperplane_i
    dist1 = hy_nor'*(new_vertex(:,i)-hy_vertex(:,1));
    dist2 = hy_nor'*(data-hy_vertex(:,1));
    S = [S;max(0,dist2./dist1)];
end
% Calculate processing time
time = etime(clock,t0);
end

% Find index of point i and j in dimension dim
function result = find_idx(i,j,dim)
    if j>i
        result = (i-1)*(dim-1)+j-1;
    else
        result = (i-1)*(dim-1)+j;
    end
end

% Find the normal vector of the hyperplane that passes through the vertices
% except vertex i
function hy_nor = find_hyper_norm(vertex,i)
    % point of hyperplane
    hy_vertex = vertex;
    % Remove the vertex that is not used
    hy_vertex(:,i)=[];
    hy_vec = [];
    for col = 2:size(hy_vertex,2)
        temp_vec = hy_vertex(:,col)-hy_vertex(:,1);
        hy_vec = [hy_vec;temp_vec.'];
    end
    hy_nor = null(hy_vec);
    if vertex(:,i).'*hy_nor > 0
        hy_nor = -hy_nor;
    end
end
