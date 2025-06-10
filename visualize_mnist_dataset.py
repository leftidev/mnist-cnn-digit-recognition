import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset
mnist_data = np.load('./datasets/mnist.npz')

# Extract images and labels
x_train = mnist_data['x_train']  # Training images
y_train = mnist_data['y_train']  # Training labels

# Print the shapes of the datasets
print(f'Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}')

# Function to display a specified number of images from the dataset
def display_images(images, labels, num_images=100):
    # Set grid size for 10x10 layout
    grid_size = 10
    plt.figure(figsize=(10, 10))  # Set figure size

    for i in range(num_images):
        plt.subplot(grid_size, grid_size, i + 1)  # Create subplots
        plt.imshow(images[i], cmap='gray')
        plt.title(f'Label: {labels[i]}')
        plt.axis('off')  # Hide axis

    plt.tight_layout()  # Adjust layout
    plt.show()

# Display 100 images from the training set
display_images(x_train, y_train, num_images=100)
