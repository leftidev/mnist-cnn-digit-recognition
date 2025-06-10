# Project Overview

This project is to learn about neural networks, specifically focusing on the application of Convolutional Neural Networks (CNNs) for visual recognition. It utilizes the MNIST dataset to train a CNN for recognizing handwritten digits. The dataset consists of 60,000 training images and 10,000 test images, serving as a benchmark for image classification tasks.

## Summary of Scripts

1. **train_mnist.py**: Trains a CNN model on the MNIST dataset for digit recognition.
2. **predict_digits.py**: Loads the trained model to recognize handwritten digits from images in a specified directory.
3. **visualize_mnist_dataset.py**: Visualizes a selection of handwritten digits from the MNIST dataset.
4. **label_digits.py**: Generates a label file for handwritten digit images stored in a specified directory.

## Script Descriptions

### train_mnist.py

This script implements a Convolutional Neural Network (CNN) for handwritten digit recognition using the MNIST dataset. 

#### Data Preprocessing

- The images are normalized to a range of [0, 1] by dividing the pixel values by 255.0.
- The training and test images are reshaped to include a channel dimension, resulting in a shape of (28, 28, 1).
- Labels are converted to one-hot encoding to facilitate multi-class classification.

#### Training and Validation Split

- The model is trained using 80% of the training data, with 20% reserved for validation to monitor the model's performance during training.

#### Model architecture

- Input Layer: Accepts images of shape (28, 28, 1).
- Convolutional Layers: Three convolutional layers with ReLU activation, followed by max pooling layers.
- Flatten Layer: Converts the 3D output to a 1D vector.
- Dense Layers: Includes a hidden layer with 64 units and an output layer with 10 units (one for each digit) using softmax activation.

#### Model Compilation

- Optimizer: Adam
- Loss Function: Categorical crossentropy
- Metrics: Accuracy

#### Model Saving and Visualization

- The model is trained for 10 epochs with a batch size of 64, and the trained model is saved as `mnist_model.h5`. Training history is plotted to visualize accuracy over epochs.

### predict_digits.py

This script recognizes handwritten digits from images stored in a specified directory.

1. **Load Pre-trained Model**: Loads the model from `trained_models/mnist_model.h5`.
2. **Image Loading**: Scans the `./own_digits/` directory for JPEG images.
3. **Image Preprocessing**: Inverts colors, resizes images to 28x28 pixels, normalizes pixel values, and reshapes them for model input.
4. **Prediction**: Feeds preprocessed images into the model to predict the digit with the highest probability.
5. **Display Results**: Shows the original and processed images alongside the predicted digit using Matplotplib.

_To use this script, ensure your images are placed in the `./own_digits/` directory and are in JPEG format. Run the script to see the predictions for each image._

### visualize_mnist_dataset.py

This script visualizes handwritten digits from the MNIST dataset.

1. **Load MNIST Dataset**: Loads the dataset from `./datasets/mnist.npz`.
2. **Extract Images and Labels**: Retrieves training images and their corresponding labels.
3. **Display Images**: Defines a function to visualize a specified number of images.

_To visualize the MNIST dataset, ensure the dataset is available in the specified directory and run the script._

### label_digits.py

This script generates a label file for handwritten digit images stored in a specified directory.

1. **Specify Directory**: Sets the directory containing the handwritten digit images (default is `./own_digits/`).
2. **Process Images**: Scans the directory for `.jpg` images and extracts labels from the filenames via underscore.
3. **Create Label Pairs**: Prepares a list of filename and label pairs in the format `filename label`.
4. **Write to File**: Saves the label pairs to a text file named `labels.txt` in the same directory.

_To use this script, ensure your images are named in the format `label_filename.jpg` and run the script to generate the labels._

## Libraries and Frameworks

- This project utilizes TensorFlow and Keras for building and training the CNN model.
- Matplotlib is used for visualizing the training history.