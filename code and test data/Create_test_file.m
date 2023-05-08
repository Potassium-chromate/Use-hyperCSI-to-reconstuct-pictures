clc;clear;

load('Nevada.mat');

X = reshape(X, size(X, 1)*size(X, 2), size(X, 3))';
dim = size(X);

cols_to_masked = 2:2:dim(2)-2;

%cols_to_masked = [];

mask = ones(size(X));
mask(:,cols_to_masked) = 0;

cols_to_masked = 2:13:dim(2)-2;
mask(:,cols_to_masked) = 0;




X_c = X.*mask;

save('test.mat', 'X_c');
