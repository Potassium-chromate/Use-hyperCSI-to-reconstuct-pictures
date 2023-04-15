clc;clear;
%set the dimension of the data
create_dim = 2;
% A is the vertex of the simplex
%And A should conclude create_dim+1 points
A=[];
for i = 1:create_dim+1
    vertex=[];
    for j = 1:create_dim
        random_number = 20 * rand() - 10; % -10~10
        vertex = [vertex;random_number];
    end
    A = [A,vertex];
end

%make sure the vertex exist
ver = eye(create_dim+1);
% Generate a random matrix to perform the component ratio of the matrix
S = rand(create_dim+1, 10000-(create_dim+1));
% Define the probability of an element being set to 0
p_zero = 0.01;  % 1% probability

% Generate a binary mask with the same dimensions as S
binary_mask = rand(create_dim+1, 10000-(create_dim+1)) >= p_zero;

% Once there is a component = 0
% the point will on the hyper plane of simplex
S_zeroed = S .* binary_mask;

S = [ver,S_zeroed];

% Normalize each column of S to make sure the sum of each component is 1
S = S ./ sum(S);
X3D = A*S;
%reshape to 244 * 10000
temp = rand(244,create_dim);
temp = temp ./ sum(temp,2);
X3D = temp*X3D;
%X3D = reshape(X3D,100,100,224);
%Y = reshape(X3D, size(X3D, 1)*size(X3D, 2), size(X3D, 3));

% Save the big matrix to a .mat file
save('data40.mat', 'X3D');
