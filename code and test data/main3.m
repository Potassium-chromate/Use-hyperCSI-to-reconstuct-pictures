%%
clc; clear; close;


load('data40.mat');
%X3D = double(hcube.DataCube);
%Y = reshape(X3D, size(X3D, 1)*size(X3D, 2), size(X3D, 3))';
Y = X3D;
dim = size(Y);

r_re_num = 0.9; %the ratio of rows to be removed
c_re_num = 0.9; %the ratio of cols to be removed

rows_to_remove = randperm(dim(1), int32(dim(1)*r_re_num));
cols_to_remove = randperm(dim(2), int32(dim(2)*c_re_num));
remove_rate = r_re_num*c_re_num

%use mask to create Y1 and Y2
mask_Y1 = true(size(Y, 1), 1); % Create a logical array with the same number of rows as Y
mask_Y1(rows_to_remove) = false; % Set elements corresponding to rows_to_remove to false
Y1 = Y(mask_Y1, :); % Select only rows with a true value in the mask

mask_Y2 = true(1,size(Y, 2)); 
mask_Y2(cols_to_remove) = false; 
Y2 = Y(:, mask_Y2); 



% Y1 (row,10000)
[data_1,dec_data_1,C_1,means_1] = PCA(Y1,2);
purest_vertex_1 = SPA_r(data_1);
[Y1_vertex,a_1,S_1] = Hyper_SCI_r(data_1,purest_vertex_1,C_1,means_1);

% Y2 (row,10000)
[data_2,dec_data_2,C_2,means_2] = PCA(Y2,2);
purest_vertex_2 = SPA_r(data_2);
[Y2_vertex,a_2,S_2] = Hyper_SCI_r(data_2,purest_vertex_2,C_2,means_2);

difference = a_2*S_1 - Y;
result = a_2*S_1;
frobenius_norm_difference = norm(difference, 'fro')