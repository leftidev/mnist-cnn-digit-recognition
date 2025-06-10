import subprocess
import numpy as np
import cv2 
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt

# Run the labeling.py script to make labels from the handwritten digits in folder 'own_digits'
subprocess.run(['python', 'label_digits.py'], check=True)

# Load the trained model
model = load_model('./trained_models/mnist_model.h5')

# Print the current working directory
print("Current working directory:", os.getcwd())

# Load and preprocess the image
def preprocess_image(image):
    # Invert colors: white background to black and dark digits to white
    inverted_image = cv2.bitwise_not(image)
    
    # Resize to 28x28 pixels
    img_resized = cv2.resize(inverted_image, (28, 28))
    
    # Normalize to [0, 1]
    img_normalized = img_resized.astype('float32') / 255.0
    
    # Reshape to (1, 28, 28, 1)
    img_reshaped = img_normalized.reshape((1, 28, 28, 1))
    
    return img_reshaped, img_resized  # Return both reshaped and resized images

# Function to display the images side by side with predictions
def display_images(original_image, processed_image, predicted_class, predicted_probabilities, true_digit=None, title=''):
    plt.figure(figsize=(8, 8))  # Set the figure size
    plt.subplot(1, 2, 1)  # First subplot
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('on')  # Hide axis

    plt.subplot(1, 2, 2)  # Second subplot
    plt.imshow(processed_image, cmap='gray')
    plt.title('Processed Image')
    plt.axis('on')  # Hide axis

    # Add predicted class and probabilities below the images
    # Color the predicted digit based on correctness
    predicted_color = 'green' if true_digit is not None and predicted_class == true_digit else 'red'
    plt.figtext(0.7, 0.01, f'Predicted Digit: {predicted_class}', ha='left', fontsize=12, color=predicted_color)
    
    # Display the true digit in standard color
    if true_digit is not None:
        plt.figtext(0.7, 0.05, f'True Digit: {true_digit}', ha='left', fontsize=12)

    prob_text = 'Predicted Probabilities:\n' + '\n'.join([f'Digit {i}: {prob:.4f}' for i, prob in enumerate(predicted_probabilities)])
    plt.figtext(0.5, 0.01, prob_text, ha='center', fontsize=10)

    plt.suptitle(title)  # Overall title
    plt.show()

# Load labels from the labels file
labels = {}
with open('./own_digits/labels.txt', 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            filename, label = parts
            labels[filename] = int(label)

# Path to the digits folder
digits_folder = './own_digits/'  # Adjust this path as necessary

# Initialize a counter for correct predictions
correct_predictions = 0
total_predictions = 0

# Process all .jpg images in the digits folder
for filename in os.listdir(digits_folder):
    if filename.endswith('.jpg'):
        image_path = os.path.join(digits_folder, filename)
        
        # Load the original image
        original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
        if original_image is None:
            print(f"Error: Unable to load image at {image_path}")
            continue
        
        # Preprocess the image
        preprocessed_image, resized_image = preprocess_image(original_image)

        # Make a prediction
        predictions = model.predict(preprocessed_image)
        predicted_class = np.argmax(predictions)  # Get the index of the highest probability
        predicted_probabilities = predictions[0]  # Get the probabilities for each class

        # Get the true label if it exists
        true_digit = labels.get(filename, None)

        # Increment the total predictions counter
        total_predictions += 1
        
        # Check if the prediction is correct
        if true_digit is not None and predicted_class == true_digit:
            correct_predictions += 1

        # Display the original and processed images side
        # Display the original and processed images side by side with predictions
        display_images(original_image, resized_image, predicted_class, predicted_probabilities, true_digit=true_digit, title=f'Images for {filename}')

# Print the total number of predictions and the number of correct predictions
print(f"Total predictions: {total_predictions}")
print(f"Correct predictions: {correct_predictions}")
print(f"Accuracy: {correct_predictions / total_predictions * 100:.2f}%")
