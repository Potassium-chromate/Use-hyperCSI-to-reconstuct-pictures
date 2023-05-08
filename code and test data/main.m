%%
clc; clear; close;


load('Nevada.mat');
%X3D = double(hcube.DataCube);
X_c = X;
X = reshape(X, size(X, 1)*size(X, 2), size(X, 3))';
dim = size(X);

masked_num = 2; %the ratio of rows to be masked 0~1
%cols_to_masked = randperm(dim(2),int32(dim(2)*masked_num));
cols_to_masked = 2:3:dim(2)-2;
rows_to_masked = [];

%use mask to create Y1
mask = ones(size(X)); % Create a logical array with the same number of rows as Y
mask(:,cols_to_masked) = 0; % Set elements corresponding to rows_to_remove to false
mask(rows_to_masked,:) = 0;
total_sum = sum(sum(mask));
remove_rate = 100*(1-total_sum/dim(1)/dim(2))
X_copy = X.*mask; % Select only rows with a true value in the mask
X_reshaped = reshape(X_copy', 150, 150, 183);
%{
% Create a specific folder to save the images, e.g., 'output_images'
output_folder = 'C:\Users\88696\Desktop\hyperspectral picture';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% Separate the 3D array into 183 images and save them as grayscale images
for i = 1:183
    img = uint8(X_c(:, :, i)*255); % Extract the i-th image and convert it to uint8 for grayscale
    img_filename = fullfile(output_folder, sprintf('image_%03d.png', i)); % Create a filename for the i-th image
    imwrite(img, img_filename, 'png'); % Save the i-th image as a PNG file in the specified folder
end
%}
X_corrupt_remove = X;
X_corrupt_remove(:,cols_to_masked)=[];
X_corrupt_remove(rows_to_masked,:)=[];


% Y1 (row,10000)
[data_1,dec_data_1,C_1,means_1] = PCA(X_corrupt_remove,4);
[row1 ,col1] = size(data_1);
purest_vertex_1 = SPA(data_1,col1,row1+1);
[Y1_vertex,a_1,S_1,time_1] = Hyper_SCI(data_1,purest_vertex_1,C_1,means_1,.95);

temp = ones(dim);
ret = a_1*S_1;
for i = cols_to_masked
    temp(:,i)=0;
end
for j = rows_to_masked
    temp(j,:)=0;
end
ptr = 1;
unmasked_rows = setdiff((1:dim(1)), rows_to_masked);
for i = 1:dim(2)
    if temp(unmasked_rows,i)==0;
        temp(unmasked_rows,i) = (ret(:,ptr-1)+ret(:,ptr+1))/2;
    else
        temp(unmasked_rows,i)= ret(:,ptr);
        ptr =ptr+1;
    end
end
for j = 1:dim(1)
    if temp(j,:)==0;
        temp(j,:) = temp(j-1,:)+temp(j+1,:)/2;
    end
end
differences = temp - X;

RGBmax = max(max(max(X_c)));
RGBmin = min(min(min(X_c)));
X_c = (X_c-RGBmin)/(RGBmax-RGBmin)*255;

RGBmax = max(max(max(temp)));
RGBmin = min(min(min(temp)));
temp = (temp-RGBmin)/(RGBmax-RGBmin)*255;

temp = reshape(temp', size(X_c, 1),size(X_c, 2), size(X_c, 3));
img =uint8(temp(:,:,[18,8,2]));
subplot(1, 2, 1);
imshow(img);
title('Inpaint');

img_2 = uint8(X_c(:,:,[18,8,2]));
subplot(1, 2, 2);
imshow(img_2);
title('Ground truth');



squared_differences = differences.^2;
frobenius_norm_difference = norm(differences, 'fro')
mean_squared_difference = mean(squared_differences(:));
RMSE = sqrt(mean_squared_difference);
% Display RMSE
disp(['The RMSE is ', num2str(RMSE)]);
