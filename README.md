# Hyperspectral Image Inpainting

## Purposes
The purpose of this repository is to provide a MATLAB implementation of a hyperspectral image inpainting algorithm, which combines Principal Component Analysis (PCA), Successive Projection Algorithm (SPA), and Hyper_SCI to improve the performance of image inpainting tasks. This implementation is suitable for researchers and practitioners working with hyperspectral images who require an efficient method for image inpainting and restoration.

## Requirements

- MATLAB R2020a or later
- Hyperspectral image dataset

## Usage

1. `PCA.m`: Function to compress and decompress a 3D matrix using Principal Component Analysis (PCA).

2. `SPA.m`: Successive Projection Algorithm (SPA) for finding the purest pixels in a hyperspectral image.

3. `Hyper_SCI.m`: Implementation of Hyper_SCI algorithm for image inpainting.

4. `main.m`: The main script that demonstrates the usage of PCA, SPA, and Hyper_SCI for hyperspectral image inpainting.

## Preprocessing
In the `main_script.m`, the original hyperspectral image data is first loaded and reshaped into a 2D matrix. The purpose of reshaping is to transform the 3D hyperspectral image data into a 2D matrix, where each column represents a pixel and each row represents a spectral band. This makes it easier to perform the subsequent PCA, SPA, and Hyper_SCI operations.

Next, the script masks the original data by removing the columns (pixels) with all zero values:  
``` matlab
cols_to_masked = find(all(X_c == 0));
X_c(:,cols_to_masked) =[];
```
In this step, `cols_to_masked` contains the indices of columns (pixels) with all zero values, and these columns are removed from the reshaped data matrix `X_c`. This masking operation is essential to focus on the relevant pixels in the hyperspectral image and improve the efficiency of the subsequent inpainting algorithm.


## Algorithm Overview
The provided MATLAB code implements a hyperspectral image inpainting algorithm that combines PCA, SPA, and Hyper_SCI techniques to recover missing or corrupted data in the image. Here's a brief overview of the main steps in the algorithm:
1. **PCA (Principal Component Analysis):** The hyperspectral data is compressed by applying PCA, which reduces the dimensionality of the data while retaining most of the information. This step allows for more efficient processing in the subsequent stages.
``` matlab
[data,dec_data,C,means] = PCA(X_c,4);
```
2. **SPA (Successive Projection Algorithm):** SPA is used to identify the purest pixels in the compressed data, which are considered as endmembers. These endmembers are important for the unmixing process in the next step.
``` matlab
purest_vertex = SPA(data,col,row+1);
```
3. **Hyper_SCI (Hyperspectral Subspace Constrained Identification):** Given the endmembers found by SPA, Hyper_SCI is applied to unmix the data and estimate the abundance of each endmember in each pixel. This step recovers the missing or corrupted data in the hyperspectral image.
``` matlab
[Y_vertex,a,S,time] = Hyper_SCI(data,purest_vertex,C,means,1);
```
4. **Reconstruction and inpainting:** After applying Hyper_SCI, the reconstructed data is combined with the original (masked) data. The missing or corrupted data is then inpainted by interpolating neighboring pixels' values.
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
To adjust the training and testing data, you can modify the following variables:
- `corruption_level`: Controls the percentage of pixels to set to 0 when corrupting the images.
- `arg_factor`: Controls the number of augmented samples generated for each image in the dataset.
- `val_ratio`: The variable represents the proportion of data that will be used for testing.For example, if you set `val_ratio` to 0.8, 80% of the data will be used for testing, and the remaining 20% will be used for training.

For example, to increase the corruption level to 95% and the augmentation factor to 25:  
And test images do not need augmentation, so the `arg_factor` remain as 1.
```python
train_corrupt , train_complete = add_noise(train_img, 0.95, 25)
test_corrupt , test_complete = add_noise(test_img, 0.95, 1)
```

To adjust the model's hyperparameters, you can modify the following lines:
```python
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.009, beta_1=0.6, clipnorm=0.001, epsilon=0.001), loss=combined_loss, metrics=['accuracy'])
```
To adjust the number of training epochs and batch size, you can modify the following line:
```python
model.fit(train_corrupt, train_complete, epochs=15, batch_size=32, verbose=1)
```

## Additional Information
### Model Architecture
The U-Net architecture used in this script consists of an encoder (contraction) path and a decoder (expansion) path.
- **Encoder** (contraction) path: This path consists of 3 blocks. Each block has 2 Convolutional layers followed by a Batch Normalization layer and a Leaky ReLU activation function. The output of each block is then downsampled using a MaxPooling2D layer.
- **Bottom layer**: After the encoder, there is a bottom layer containing 3 Convolutional layers, each followed by a Batch Normalization layer and a Leaky ReLU activation function.
- **Decoder (expansion) path**: The decoder path consists of 3 blocks. Each block has an UpSampling2D layer followed by a Convolutional layer and a concatenation operation with the corresponding layer from the encoder path. This is followed by 2 Convolutional layers, each with a Batch Normalization layer and a Leaky ReLU activation function.
- **Output layer**: The output layer is a Conv2D layer with a sigmoid activation function to produce the inpainted image.
![Alt Text](https://github.com/Potassium-chromate/Hyperspectral-Image-Inpainting/blob/main/pictures/U-net.png)

### Combined Loss Function
The combined loss function used in this script is a linear combination of Mean Squared Error (MSE) and Structural Similarity Index Measure (SSIM). This helps the model to focus on both pixel-level reconstruction as well as preserving structural information in the images. The weighting factor `alpha` can be adjusted to control the trade-off between MSE and SSIM.

### Potential Improvements
- Experiment with different model architectures or pre-trained models to improve inpainting performance.
- Adjust the corruption level and augmentation factor to create a more diverse set of training and testing samples.
- Perform data normalization or other preprocessing techniques to improve model performance.
- Implement early stopping or learning rate scheduling to improve training efficiency and prevent overfitting.
- Experiment with different loss functions or combination of loss functions to further optimize the model's performance.

### Recover rate
- **Recovery rate**:  that can describe how much data has been recovered in the inpainting process.  
- Recovery rate = `val_ratio` *  `corruption_level` * 100  
- In the example, Recovery rate = 0.65 * 0.65 * 100 = 42.25%   

## Result
![Alt Text](https://github.com/Potassium-chromate/Hyperspectral-Image-Inpainting/blob/main/pictures/Result%201.png)
![Alt Text](https://github.com/Potassium-chromate/Hyperspectral-Image-Inpainting/blob/main/pictures/Result%202.png)
![Alt Text](https://github.com/Potassium-chromate/Hyperspectral-Image-Inpainting/blob/main/pictures/Result%203.png)

## Futher test

| category  | quantity| corrupt rate | Avg RMSE  | params used |
| :----     | :----:  | :----------: | ---:  |----------------:|
| train     | 55      | 0            |       |                 |
| test      | 128     | 0.7          |       |                 |
| total     | 183     | 0.49         | 0.068 | 8,053,105       |

| category  | quantity| corrupt rate | Avg RMSE  | params used |
| :----     | :----:  | :----------: | ---:  |----------------:|
| train     | 37      | 0            |       |                 |
| test      | 152     | 0.7          |       |                 |
| total     | 183     | 0.56         | 0.039 |   9,592,497     |

| category  | quantity| corrupt rate | Avg RMSE  | params used |
| :----     | :----:  | :----------: | ---:  |----------------:|
| train     | 37      | 0            |       |                 |
| test      | 152     | 0.8          |       |                 |
| total     | 183     | 0.64         | 0.025 |   9,592,497     |

| category  | quantity| corrupt rate | Avg RMSE  | params used |  
| :----     | :----:  | :----------: | ---:  |----------------:|
| train     | 28      | 0            |       |                 |
| test      | 155     | 0.8          |       |                 |
| total     | 183     | 0.68         | 0.044 |   9,592,497     |

| category  | quantity| corrupt rate | Avg RMSE  | params used |   
| :----     | :----:  | :----------: | ---:  |----------------:|
| train     | 28      | 0            |       |                 |
| test      | 155     | 0.85         |       |                 |
| total     | 183     | 0.72         | 0.079 |   9,592,497     |

| category  | quantity| corrupt rate | Avg RMSE  | params used |   
| :----     | :----:  | :----------: | ---:  |----------------:|
| train     | 19      | 0            |       |                 |
| test      | 164     | 0.9          |       |                 |
| total     | 183     | 0.81         | 0.024 |   10,069,313    |

| category  | quantity| corrupt rate | Avg RMSE  | params used |   
| :----     | :----:  | :----------: | ---:  |----------------:|
| train     | 15      | 0            |       |                 |
| test      | 168     | 0.95         |       |                 |
| total     | 183     | 0.874        | 0.034 |   10,069,313    |


