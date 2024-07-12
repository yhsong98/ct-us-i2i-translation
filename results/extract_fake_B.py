import os
import shutil

# Define the source and destination directories
source_dir = '../checkpoints/cyclegan_6_2/web/images'
destination_dir = '../checkpoints/cyclegan_6_2/web/fake_B'

# source_dir = 'cycle_gan_5_6/test_179/images'
# destination_dir = 'cycle_gan_5_6/test_179/fake_B'
# Ensure destination directory exists, create if not
if os.path.exists(destination_dir):
    shutil.rmtree(destination_dir)
os.makedirs(destination_dir)

# List all files in the source directory
files = os.listdir(source_dir)

# Filter files that end with 'fake_B'
filtered_files = [file for file in files if file.endswith('fake_B.png')]

# Copy filtered files to the destination directory
for file in filtered_files:
    shutil.copy(os.path.join(source_dir, file), os.path.join(destination_dir, file))

print(f"Successfully copied {len(filtered_files)} images to {destination_dir}")
