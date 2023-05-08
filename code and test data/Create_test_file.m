clc;clear;

load('Nevada.mat');

X = reshape(X, size(X, 1)*size(X, 2), size(X, 3))';
dim = size(X);

%cols_to_masked = randperm(dim(2),int32(dim(2)*masked_num));
cols_to_masked = 2:3:dim(2)-2;
mask = ones(size(X));
mask(:,cols_to_masked) = 0;

X = X.*mask;

save('test.mat', 'X');
