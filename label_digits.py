import os

# Specify the directory containing the handwritten digit images
digits_folder = './own_digits/'  # Change this to your directory

# Prepare a list to hold the filename and label pairs
label_pairs = []

# Process all .jpg images in the digits folder
for filename in os.listdir(digits_folder):
    if filename.endswith('.jpg'):
        # Split the filename to extract the label
        label = filename.split('_')[0]  # Get the part before the underscore
        # Create the entry in the desired format
        entry = f"{filename} {label}"
        label_pairs.append(entry)

# Specify the output file path
output_file_path = './own_digits/labels.txt'

# Write the filename and label pairs to the output file
with open(output_file_path, 'w') as f:
    for entry in label_pairs:
        f.write(entry + '\n')

print(f"Labels have been written to {output_file_path}")
