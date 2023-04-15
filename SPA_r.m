function result = SPA_r(X)

dim = size(X);
%In N dimension, the simplex will have N+1 vertices
vertex_index = zeros(1,dim(1)+1);

for i = 1:dim(1)-1
    row_means = mean(X, 2);
    X = X - row_means;
    %calculate the distance to original
    Distance =vecnorm(X);
    %Choose the point with max distance
    [~, farthest_index] = max(Distance);
    vertex_index(i) = farthest_index;
    %coordinate of first_point
    farthest_index = X(:,farthest_index);
    %Shift all points and make the farthest point origin
    X = X - farthest_index;
    %normal_vector of the plane
    n = farthest_index / norm(farthest_index);
    %mapping all point to a hyperplane
    X_dot = X.'*n;
    X = X.'-X_dot*n.';
    X = X.';
end
%second to last vertex
row_means = mean(X, 2);
X = X - row_means;
Distance =vecnorm(X);
[~, last_two_index] = max(Distance);
vertex_index(dim(1)) = last_two_index;
%last vertex
X = X - X(:,last_two_index);
Distance =vecnorm(X);
[~, last_one_index] = max(Distance);
vertex_index(dim(1)+1) = last_one_index;

result = vertex_index;



