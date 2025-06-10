import subprocess
import numpy as np
import cv2 
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt

# Run the labeling.py script to make labels from the handwritten digits in folder 'own_digits'
subprocess.run(['python', 'label_digits.py'], check=True)

model = load_model('./trained_models/mnist_model.keras')

print("Current working directory:", os.getcwd())

def preprocess_image(image):
    inverted_image = cv2.bitwise_not(image)                     # Invert colors: white background to black and dark digits to white
    blurred_image = cv2.GaussianBlur(inverted_image, (5, 5), 0) # Apply Gaussian Blur to reduce noise
    
    # Apply adaptive thresholding to create a binary image
    binary_image = cv2.adaptiveThreshold(blurred_image, 255, 
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 
                                         11, 2)
    
    img_resized = cv2.resize(binary_image, (28, 28))        # Resize to 28x28 pixels
    img_normalized = img_resized.astype('float32') / 255.0  # Normalize to [0, 1]
    img_reshaped = img_normalized.reshape((1, 28, 28, 1))   # Reshape to (1, 28, 28, 1)
    
    return img_reshaped, img_resized 


def display_images(original_image, processed_image, predicted_class, predicted_probabilities, true_digit=None, title=''):
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('on')

    plt.subplot(1, 2, 2)
    plt.imshow(processed_image, cmap='gray')
    plt.title('Processed Image')
    plt.axis('on')

    # Add predicted class and probabilities below the images and color the predicted digit based on correctness
    predicted_color = 'green' if true_digit is not None and predicted_class == true_digit else 'red'
    plt.figtext(0.7, 0.01, f'Predicted Digit: {predicted_class}', ha='left', fontsize=12, color=predicted_color)
    
    if true_digit is not None:
        plt.figtext(0.7, 0.05, f'True Digit: {true_digit}', ha='left', fontsize=12)

    prob_text = 'Predicted Probabilities:\n' + '\n'.join([f'Digit {i}: {prob:.4f}' for i, prob in enumerate(predicted_probabilities)])
    plt.figtext(0.5, 0.01, prob_text, ha='center', fontsize=10)

    plt.suptitle(title)
    plt.show()

# Load labels from the labels file
labels = {}
with open('./own_digits/labels.txt', 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            filename, label = parts
            labels[filename] = int(label)

digits_folder = './own_digits/'

correct_predictions = 0
total_predictions = 0

# Process all .jpg images in the digits folder
for filename in os.listdir(digits_folder):
    if filename.endswith('.jpg'):
        image_path = os.path.join(digits_folder, filename)
        
        # Load the original image
        original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if original_image is None:
            print(f"Error: Unable to load image at {image_path}")
            continue
        
        # Preprocess the image
        preprocessed_image, resized_image = preprocess_image(original_image)

        # Make a prediction
        predictions = model.predict(preprocessed_image)
        predicted_class = np.argmax(predictions)  # Get the index of the highest probability
        predicted_probabilities = predictions[0]  # Get the probabilities for each class

        true_digit = labels.get(filename, None)
        total_predictions += 1
        
        if true_digit is not None and predicted_class == true_digit:
            correct_predictions += 1

        # Display the original and processed images side by side with predictions
        display_images(original_image, resized_image, predicted_class, predicted_probabilities, true_digit=true_digit, title=f'Images for {filename}')

print(f"Total predictions: {total_predictions}")
print(f"Correct predictions: {correct_predictions}")
print(f"Accuracy: {correct_predictions / total_predictions * 100:.2f}%")
