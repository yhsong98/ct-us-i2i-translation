from PIL import Image
import os

# Define the source and destination directories
source_dir = 'cycle_gan_5_6/test_179/fake_B'
destination_dir = 'cycle_gan_5_6/test_179/fake_B_grayscale'

# Ensure the destination directory exists, create if not
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Iterate over all files in the source directory
for filename in os.listdir(source_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
        # Construct full file path
        file_path = os.path.join(source_dir, filename)
        # Open the image
        with Image.open(file_path) as img:
            # Convert the image to grayscale
            grayscale_img = img.convert('L')
            # Construct the destination file path
            dest_file_path = os.path.join(destination_dir, os.path.splitext(filename)[0] + '_grayscale.png')
            # Save the grayscale image
            grayscale_img.save(dest_file_path, 'PNG')

print("Conversion to grayscale completed for all images.")
