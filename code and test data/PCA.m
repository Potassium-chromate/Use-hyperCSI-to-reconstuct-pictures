function [A_compress, A_decompress,top_eig_vec,row_means] = PCA(A,dim)
% This function compresses a 3D matrix using PCA and plots the compressed
% image as a scatter plot.

% Perform PCA compression
row_means = mean(A, 2);
A_re_centered = A - row_means;
C = A_re_centered * A_re_centered';
[V, D] = eig(C);
[~, sorted_idx] = sort(diag(D), 'descend');
top_eig_vec = V(:, sorted_idx(1:dim)); %top_eig_vec is matrix "C"
A_compress = top_eig_vec' * A_re_centered;

% Decompress the image
A_decompress = top_eig_vec * A_compress + row_means;


% Plot the compressed image
X = A_compress(1, :);
Y = A_compress(2, :);
plot(X,Y)
%hold on;
end