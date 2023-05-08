%%
clc; clear; close;

load('test.mat');
load('Nevada.mat');

X = reshape(X, size(X, 1)*size(X, 2), size(X, 3))';
dim = size(X);

% Find columns with all zeros (masked areas)
cols_to_masked = find(all(X_c == 0));
dim2 = size(cols_to_masked);

% Calculate the remove rate of columns with all zeros
remove_rate = 100 * dim2(2) / dim(2);

% Remove columns with all zeros from the data
X_c(:, cols_to_masked) = [];

% Start the timer
t0 = clock;

% HyperCSI Algorithm
[data, dec_data, C, means] = PCA(X_c, 4);
[row, col] = size(data);
purest_vertex = SPA(data, col, row + 1);
[Y_vertex, a, S, time] = Hyper_SCI(data, purest_vertex, C, means, 2);
time = etime(clock,t0);

% Create a temporary matrix to store the reconstructed data
temp = ones(dim);
ret = a * S;

for i = cols_to_masked
    temp(:,i)=0;
end

% Fill the temporary matrix with the reconstructed data
ptr = 1;
for i = 1:dim(2)
    if temp(:, i) == 0;
        continue;
    else
        temp(:, i) = ret(:, ptr);
        ptr = ptr + 1;
       
    end
end
corrupt = temp;

% Inpaint the masked areas in the temporary matrix
for i = 1:dim(2)
    if temp(:, i) == 0;
        index = i;
        while temp(:, index) == 0
            index = index + 1;
        end

        temp(:, i) = (temp(:, i - 1) + temp(:, index)) / 2;
    end
end

% Calculate the differences between the inpainted and ground truth data
differences = temp - X;

% Normalize the RGB values
RGBmax = max(max(max(X)));
RGBmin = min(min(min(X)));
X = (X - RGBmin) / (RGBmax - RGBmin) * 255;

RGBmax = max(max(max(temp)));
RGBmin = min(min(min(temp)));
temp = (temp - RGBmin) / (RGBmax - RGBmin) * 255;
corrupt = (corrupt - RGBmin) / (RGBmax - RGBmin) * 255;

% Reshape the matrices for visualization
temp = reshape(temp', 150, 150, 183);
corrupt = reshape(corrupt', 150, 150, 183);
X = reshape(X', 150, 150, 183);

% Display the inpainted image and ground truth
img_1 = uint8(corrupt(:, :, [18, 8, 2]));
subplot(1, 3, 1);
imshow(img_1);
title('Corrupt');

img_2 = uint8(temp(:, :, [18, 8, 2]));
subplot(1, 3, 2);
imshow(img_2);
title('Inpaint');

img_3 = uint8(X(:, :, [18, 8, 2]));
subplot(1, 3, 3);
imshow(img_3);
title('Ground truth');



% Calculate error metrics
squared_differences = differences .^ 2;
frobenius_norm_difference = norm(differences, 'fro');
mean_squared_difference = mean(squared_differences(:));
RMSE = sqrt(mean_squared_difference);

% Display RMSE
disp(['The RMSE is ', num2str(RMSE)]);
disp(['The F_norm is ', num2str(frobenius_norm_difference)]);
disp(['The remove rate is ', num2str(remove_rate)]);
disp(['The cost time is ', num2str(time)]);
