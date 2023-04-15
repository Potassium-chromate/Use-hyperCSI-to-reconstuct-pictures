function [new_vertex,A,S] = Hyper_SCI_r(data,vertex,C,means)


%num of vertex is dimension +1
vertex_num = size(vertex,2);
dims = size(data);
n= dims(2);

X1 = data(1, :);
Y1 = data(2, :);
plot(X1,Y1)
hold on;




%number of borders
border_vector = zeros(dims(1),vertex_num*(vertex_num-1)/2);
nor_border_vector = zeros(dims(1),vertex_num*(vertex_num-1)/2);
count = 1;
% determine the vectors between purest pixels
% border12~border1n...border23~border2n....border(n-1)n
for i = 1:vertex_num-1
    for j = i+1:vertex_num
        %finding border_vector
        border_vector(:,count) = data(:,vertex(1,i))-data(:,vertex(1,j));
        %finding unit normal  vector
        vec_1 = -data(:,vertex(1,i));
        vec_2 = border_vector(:,count);
        pro_vec = (dot(vec_1,vec_2)/norm(vec_2)^2)*vec_2;
        pro_point = data(:,vertex(1,i))+pro_vec;
        %nor_border_vector is point outward the original
        nor_border_vector(:,count) = pro_point/norm(pro_point);
        count = count+1;
    end
end

%determine r
R = border_vector.'*border_vector;
diagonal_elements = diag(R);
r = 0.5*sqrt(min(diagonal_elements));
%finding point in the Region
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

%obtaining points in the circle
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

p = zeros(dims(1),vertex_num*(vertex_num-1));

count = 1;
%obtaining p (p12,p13...p21,p23...p(n)(n-1))
%and obtaining h (h12,h13...h21,h23...h(n)(n-1))
for i = 1:vertex_num
    for j = 1:vertex_num
        if i == j 
            continue;
        elseif i<j
            bor_vec_idx = (2*vertex_num-1-i+1)/2*(i-1)+j-i;
        else 
            bor_vec_idx = (2*vertex_num-1-j+1)/2*(j-1)+i-j;
        end
        in_porduct = nor_border_vector(:,bor_vec_idx).'*inside_circles{i};
        [max_in_p, idx_in_p] = max(in_porduct);
        p(:,count) = inside_circles{i}(:,idx_in_p);
        count = count+1;
    end
end

%find new border
new_bor_vec = zeros(dims(1),vertex_num*(vertex_num-1)/2);
count = 1;
for i = 1:vertex_num-1
    for j = i+1:vertex_num
        index1 = find_idx(i,j,vertex_num);
        index2 = find_idx(j,i,vertex_num);
        new_bor_vec(:,count) = p(:,index1)-p(:,index2);
        count = count +1;
    end
end

%obtaining b^ point away from original
b_carat = zeros(dims(1),vertex_num*(vertex_num-1)/2);
h = zeros(1,vertex_num*(vertex_num-1)/2);
count = 1;
for i = 1:vertex_num-1
    for j = i+1:vertex_num
        p_idx = (i-1)*(vertex_num-1)+j-1;
        %finding unit normal  vector
        vec_1 = -p(:,p_idx);
        vec_2 = -new_bor_vec(:,count);
        pro_vec = (dot(vec_1,vec_2)/norm(vec_2)^2)*vec_2;
        pro_point = p(:,p_idx)+pro_vec;
        %b^ is point outward the original
        b_carat(:,count) = pro_point/norm(pro_point);
        h(:,count) = (b_carat(:,count).'*-vec_1)/.9;
        count = count+1;
    end
end
%finding new vertex
new_vertex = zeros(dims(1),vertex_num);
for i = 1:vertex_num
    coef_matrix = [];
    cost_vector = [];
    for j = 1:vertex_num
        if i == j
            continue;
        elseif i<j
            idx = (2*vertex_num-1-i+1)/2*(i-1)+j-i;
        else
            idx = (2*vertex_num-1-j+1)/2*(j-1)+i-j;
        end
        coef_matrix = [coef_matrix;b_carat(:,idx).'];
        cost_vector = [cost_vector;h(:,idx)];
        
    end
    new_vertex(:,i) = inv(coef_matrix)*cost_vector;
end
    






%build matirx a 
A = C*new_vertex+means;

%build matrix S
S=[];
for i = 1:vertex_num %point which is not used
    %point of hyper plane
    hy_vertex = new_vertex;
    hy_vertex(:,i)=[];
    hy_vec = [];
    for col = 2:size(hy_vertex,2)
        temp_vec = hy_vertex(:,col)-hy_vertex(:,1);
        hy_vec = [hy_vec;temp_vec.'];
    end
    hy_nor = null(hy_vec);
    %dist form alpha_i to hyperplane_i
    dist1 = abs(hy_nor'*(new_vertex(:,i)-hy_vertex(:,1)));
    dist2 = abs(hy_nor'*(data-hy_vertex(:,1)));
    S = [S;dist2./dist1];
end


end
function result = find_idx(i,j,dim)
    if j>i
        result = (i-1)*(dim-1)+j-1;
    else
        result = (i-1)*(dim-1)+j;
    end
end

