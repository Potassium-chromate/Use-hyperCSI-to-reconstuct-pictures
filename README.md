# Hyperspectral Image Inpainting

## Purposes
The purpose of this repository is to provide a MATLAB implementation of a hyperspectral image inpainting algorithm, which combines Principal Component Analysis (PCA), Successive Projection Algorithm (SPA), and Hyper_CSI to improve the performance of image inpainting tasks. This implementation is suitable for researchers and practitioners working with hyperspectral images who require an efficient method for image inpainting and restoration.

## Requirements

- MATLAB R2020a or later
- Hyperspectral image dataset

## Usage

1. `PCA.m`: Function to compress and decompress a 3D matrix using Principal Component Analysis (PCA).

2. `SPA.m`: Successive Projection Algorithm (SPA) for finding the purest pixels in a hyperspectral image.

3. `Hyper_CSI.m`: Implementation of Hyper_CSI algorithm for image inpainting.

4. `main.m`: The main script that demonstrates the usage of PCA, SPA, and Hyper_CSI for hyperspectral image inpainting.

## Preprocessing
In the `main_script.m`, the original hyperspectral image data is first loaded and reshaped into a 2D matrix. The purpose of reshaping is to transform the 3D hyperspectral image data into a 2D matrix, where each column represents a pixel and each row represents a spectral band. This makes it easier to perform the subsequent PCA, SPA, and Hyper_CSI operations.

Next, the script masks the original data by removing the columns (pixels) with all zero values:  
``` matlab
cols_to_masked = find(all(X_c == 0));
X_c(:,cols_to_masked) =[];
```
In this step, `cols_to_masked` contains the indices of columns (pixels) with all zero values, and these columns are removed from the reshaped data matrix `X_c`. This masking operation is essential to focus on the relevant pixels in the hyperspectral image and improve the efficiency of the subsequent inpainting algorithm.


## Algorithm Overview
The provided MATLAB code implements a hyperspectral image inpainting algorithm that combines PCA, SPA, and Hyper_CSI techniques to recover missing or corrupted data in the image. Here's a brief overview of the main steps in the algorithm:
1. **PCA (Principal Component Analysis):** The hyperspectral data is compressed by applying PCA, which reduces the dimensionality of the data while retaining most of the information. This step allows for more efficient processing in the subsequent stages.
``` matlab
[data,dec_data,C,means] = PCA(X_c,4);
```
2. **SPA (Successive Projection Algorithm):** SPA is used to identify the purest pixels in the compressed data, which are considered as endmembers. These endmembers are important for the unmixing process in the next step.
``` matlab
purest_vertex = SPA(data,col,row+1);
```
3. **Hyper_CSI (Hyperspectral Subspace Constrained Identification):** Given the endmembers found by SPA, Hyper_CSI is applied to unmix the data and estimate the abundance of each endmember in each pixel. This step recovers the missing or corrupted data in the hyperspectral image.
``` matlab
[Y_vertex,a,S,time] = Hyper_CSI(data,purest_vertex,C,means,1);
```
4. **Reconstruction and inpainting:** After applying Hyper_CSI, the reconstructed data is combined with the original (masked) data. The missing or corrupted data is then inpainted by interpolating neighboring pixels' values.
``` matlab
for i = 1:dim(2)
    if temp(:,i) == 0;
        temp(:,i) = (temp(:,i-1)+temp(:,i+1))/2;
    end
end
```
5.**Visualization and evaluation:** The inpainted hyperspectral image is visualized and compared to the ground truth image. Additionally, the algorithm's performance is assessed by computing the RMSE (Root Mean Square Error) between the inpainted and original images.
``` matlab
differences = temp - X;
RMSE = sqrt(mean(squared_differences(:)));
```


## Customization
In this repository, you can easily customize the inpainting algorithm by adjusting the `adjust_factor` variable in the `Hyper_CSI` function call. Changing the `adjust_factor` will affect the convergence of the hyperplanes towards the center of the data cloud, which in turn can impact the overall performance of the algorithm.

## Additional Information
### Test Data
The original hyperspectral data has dimensions 183x22500. To create a masked version of this data for testing purposes, we apply two masks to remove specific columns of data.

First, we remove every second column in the dataset by setting the columns to zero:
``` matlab
cols_to_masked = 2:2:dim(2)-2;
mask(:, cols_to_masked) = 0;
```

Then, we further mask the data by removing every 13th column:
``` matlab
cols_to_masked = 2:13:dim(2)-2;
mask(:, cols_to_masked) = 0;
```

After applying both masks, we obtain a masked version of the original hyperspectral data, which can be used to test and evaluate the inpainting algorithm's performance. By comparing the inpainted data with the original data, we can assess the quality of the reconstruction and the effectiveness of the algorithm.

The below is the best we have tried:   
- **Number of endmember:** 5
- **adjust_factor:** 2

| Original size| Masked Column Count | Corrupt Rate | RMSE | F_norm |
| :----:       | :----------:        | :---:        |:----:|:----:  |
| 183x22500    | 12114               |   53.84%     | 0.015|  31.123|

| Corrupt| Inpaint | Ground truth | 
| :----:       | :----------:        | :---:        |
| <img src="https://github.com/Potassium-chromate/Use-hyperCSI-to-reconstuct-pictures/blob/main/picture/Corrupt%2053%25/Corrupt%2053%25/Corrupt.png?raw=true" alt="Corrupt Image" width="300"/>  | <img src="https://github.com/Potassium-chromate/Use-hyperSCI-to-reconstuct-pictures/blob/main/picture/Corrupt%2053%25/Corrupt%2053%25/inpaint.png?raw=true" alt="Corrupt Image" width="300"/> |<img src="https://github.com/Potassium-chromate/Use-hyperSCI-to-reconstuct-pictures/blob/main/picture/Corrupt%2053%25/Corrupt%2053%25/ground%20truth.png?raw=true" alt="Corrupt Image" width="300"/> |

## Futher test

| Original size| Masked Column Count | Corrupt Rate | RMSE | F_norm |
| :----:       | :----------:        | :---:        |:----:|:----:  |
| 183x22500    | 0                   |   0    %     | 0.005|  9.6354|

| Corrupt| Inpaint | Ground truth | 
| :----:       | :----------:        | :---:        |
| <img src="https://github.com/Potassium-chromate/Use-hyperSCI-to-reconstuct-pictures/blob/main/picture/Corrupt%2053%25/Corrupt%200%25/Corrupt.png?raw=true" alt="Corrupt Image" width="300"/>  | <img src="https://github.com/Potassium-chromate/Use-hyperSCI-to-reconstuct-pictures/blob/main/picture/Corrupt%2053%25/Corrupt%200%25/Inpaint.png?raw=true" alt="Corrupt Image" width="300"/> |<img src="https://github.com/Potassium-chromate/Use-hyperSCI-to-reconstuct-pictures/blob/main/picture/Corrupt%2053%25/Corrupt%2053%25/ground%20truth.png?raw=true" alt="Corrupt Image" width="300"/> |


| Original size| Masked Column Count | Corrupt Rate | RMSE | F_norm |
| :----:       | :----------:        | :---:        |:----:|:----:  |
| 183x22500    | 13597               |   60.43%     | 0.019|  39.271|

| Corrupt| Inpaint | Ground truth | 
| :----:       | :----------:        | :---:        |
| <img src="https://github.com/Potassium-chromate/Use-hyperSCI-to-reconstuct-pictures/blob/main/picture/Corrupt%2053%25/Corrupt%2060%25/Corrupt.png?raw=true" alt="Corrupt Image" width="300"/>  | <img src="https://github.com/Potassium-chromate/Use-hyperSCI-to-reconstuct-pictures/blob/main/picture/Corrupt%2053%25/Corrupt%2060%25/Inpaint.png?raw=true" alt="Corrupt Image" width="300"/> |<img src="https://github.com/Potassium-chromate/Use-hyperSCI-to-reconstuct-pictures/blob/main/picture/Corrupt%2053%25/Corrupt%2060%25/Ground%20truth.png?raw=true" alt="Corrupt Image" width="300"/> |

| Original size| Masked Column Count | Corrupt Rate | RMSE | F_norm |
| :----:       | :----------:        | :---:        |:----:|:----:  |
| 183x22500    | 14120               |   62.75%     | 0.020|  51.54 |

| Corrupt| Inpaint | Ground truth | 
| :----:       | :----------:        | :---:        |
| <img src="https://github.com/Potassium-chromate/Use-hyperSCI-to-reconstuct-pictures/blob/main/picture/Corrupt%2053%25/Corrupt%2063%25/Corrupt.png?raw=true" alt="Corrupt Image" width="300"/>  | <img src="https://github.com/Potassium-chromate/Use-hyperSCI-to-reconstuct-pictures/blob/main/picture/Corrupt%2053%25/Corrupt%2063%25/Inpaint.png?raw=true" alt="Corrupt Image" width="300"/> |<img src="https://github.com/Potassium-chromate/Use-hyperSCI-to-reconstuct-pictures/blob/main/picture/Corrupt%2053%25/Corrupt%2063%25/Ground%20Truth.png?raw=true" alt="Corrupt Image" width="300"/> |

| Original size| Masked Column Count | Corrupt Rate | RMSE | F_norm |
| :----:       | :----------:        | :---:        |:----:|:----:  |
| 183x22500    | 15797               |   70.20%     | 0.332|  673.9 |

| Corrupt| Inpaint | Ground truth | 
| :----:       | :----------:        | :---:        |
| <img src="https://github.com/Potassium-chromate/Use-hyperSCI-to-reconstuct-pictures/blob/main/picture/Corrupt%2053%25/Corrupt%2070%25/Corrupt.png?raw=true" alt="Corrupt Image" width="300"/>  | <img src="https://github.com/Potassium-chromate/Use-hyperSCI-to-reconstuct-pictures/blob/main/picture/Corrupt%2053%25/Corrupt%2070%25/Inpaint.png?raw=true" alt="Corrupt Image" width="300"/> |<img src="https://github.com/Potassium-chromate/Use-hyperSCI-to-reconstuct-pictures/blob/main/picture/Corrupt%2053%25/Corrupt%2070%25/Ground%20Truth.png?raw=true" alt="Corrupt Image" width="300"/> |

### Experiment for `adjust rate` = 0.9 

| Original size| Masked Column Count | Corrupt Rate | RMSE | F_norm | adjust rate |
| :----:       | :----------:        | :---:        |:----:|:----:  | :----------:|
| 183x22500    | 12114               |   53.84%     | 0.064|  **131.51** | **0.9**|

| Corrupt| Inpaint | Ground truth | 
| :----:       | :----------:        | :---:        |
| <img src="https://github.com/Potassium-chromate/Use-hyperSCI-to-reconstuct-pictures/blob/main/picture/Corrupt%2053%25/0.9%20adjust%20factor/Corrupt.png?raw=true" alt="Corrupt Image" width="300"/>  | <img src="https://github.com/Potassium-chromate/Use-hyperSCI-to-reconstuct-pictures/blob/main/picture/Corrupt%2053%25/0.9%20adjust%20factor/Inpaint.png?raw=true" alt="Corrupt Image" width="300"/> |<img src="https://github.com/Potassium-chromate/Use-hyperSCI-to-reconstuct-pictures/blob/main/picture/Corrupt%2053%25/0.9%20adjust%20factor/Ground%20Truth.png?raw=true" alt="Corrupt Image" width="300"/> |

-Conclusion: the performance sucks!

## Abundance maps

![map 1](https://github.com/Potassium-chromate/Use-hyperSCI-to-reconstuct-pictures/blob/main/picture/Corrupt%2053%25/abundance%20maps/map%201.png)
![map 2](https://github.com/Potassium-chromate/Use-hyperSCI-to-reconstuct-pictures/blob/main/picture/Corrupt%2053%25/abundance%20maps/map%202.png)
![map 3](https://github.com/Potassium-chromate/Use-hyperSCI-to-reconstuct-pictures/blob/main/picture/Corrupt%2053%25/abundance%20maps/map%203.png)
![map 4](https://github.com/Potassium-chromate/Use-hyperSCI-to-reconstuct-pictures/blob/main/picture/Corrupt%2053%25/abundance%20maps/map%204.png)
![map 5](https://github.com/Potassium-chromate/Use-hyperSCI-to-reconstuct-pictures/blob/main/picture/Corrupt%2053%25/abundance%20maps/map%205.png)


