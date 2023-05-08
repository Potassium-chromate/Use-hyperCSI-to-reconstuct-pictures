# Hyperspectral Image Inpainting

## Purposes
This is a Python script to inpaint hyperspectral images of size (150,150,183) using a U-Net based architecture.

## Requirements

- Python 3
- Tensorflow 2.x
- Scipy
- NumPy
- Matplotlib

## Usage

1. Make sure you have the required libraries installed.

2. Change the `file_name` variable to the path of your .mat file containing the hyperspectral data.

```python
file_name  = "path/to/your/data.mat"
```
3. Run the script. This will load the hyperspectral data, preprocess the images, create a model, and train it on the provided dataset. After training, the script will show the results for the test images including the corrupted image, predicted image, and ground truth.  
```python 
python hyperspectral_inpainting.py
```

## Preprocessing
The input hyperspectral images have a shape of (150, 150). In order to perform convolution and upsampling operations effectively, each side of the image should be a multiple of 8. This is because the specific architecture used in this code involves multiple downsampling (using MaxPooling2D) and upsampling (using UpSampling2D) layers, which require dimensions that are divisible by the downsampling/upsampling factors.

To achieve this, we pad each side of the input image by 1 pixel, resulting in a new image shape of (152, 152). Padding is done using the `np.pad()` function with a `pad_width` of 1 and a constant value of 0. This ensures that the dimensions of the image are compatible with the convolution and upsampling operations in the model while minimizing the impact of padding on the inpainting process.

## Algorithm Overview
The script is based on a U-Net architecture, which is commonly used for image inpainting tasks. The algorithm performs the following steps:
1. Load and preprocess the hyperspectral data.
2. Create and compile the U-Net model with a custom combined loss function (Mean Squared Error and Structural Similarity Index Measure).
3. Split the hyperspectral data into training and testing sets.
4. Add noise to the training and testing images.
5. Train the model on the corrupted training images.
6. Test the model on the corrupted test images, and compare the predictions with the ground truth.
7. Display the results for the test images and compute the Root Mean Squared Error (RMSE) for each test image.
![Alt Text](https://github.com/Potassium-chromate/Hyperspectral-Image-Inpainting/blob/main/pictures/Process.png)  

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


