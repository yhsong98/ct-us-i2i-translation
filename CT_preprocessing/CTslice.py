import nibabel as nib
import numpy as np
import cv2
import os
from tqdm import tqdm
import random
import shutil


def apply_fan_mask(ct_img):
    # Load the CT and ultrasound images

    # Determine the center of the fan - this is specific to the geometry of the ultrasound image
    #center_X_candidate = [ct_img.shape[1] // 2, ct_img.shape[1] // 2 +100, ct_img.shape[1] // 2 -100]
    center_x = ct_img.shape[1] // 2 # Since the vertex is at the top center
    center_y = -20  # Since the vertex is at the top center

    # The radius is the distance from the vertex to the bottom center of the image
    radius = 500

    # Approximate the angle range of the ultrasound image fan
    # These values need to be determined based on the actual ultrasound image
    angle_range = (60, 120)  # You might need to adjust these values

    # Create a meshgrid of coordinates for the CT image
    xx, yy = np.meshgrid(range(ct_img.shape[1]), range(ct_img.shape[0]))

    # Convert cartesian coordinates to polar coordinates
    r = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    theta = np.arctan2(yy - center_y, xx - center_x) * 180 / np.pi
    theta[theta < 0] += 360  # Make sure angles are between 0 and 360 degrees

    # Initialize the fan mask
    fan_mask = np.zeros_like(ct_img, dtype=np.uint8)

    # Apply the conditions to create the binary mask
    fan_mask[(r <= radius) & (theta >= angle_range[0]) & (theta <= angle_range[1])] = 255

    # Apply the fan mask to the CT image
    ct_img_masked = cv2.bitwise_and(ct_img, ct_img, mask=fan_mask)

    return ct_img_masked
# Path to your .nii or .nii.gz file
ct_dir = 'CT/Subtask1/TrainImage/'
mask_dir = 'CT/Subtask1/TrainMask/'

nii_files = os.listdir(ct_dir)
mask_files = os.listdir(mask_dir)

# nii_files  = os.listdir('slice_test/image')
# mask_files = os.listdir('slice_test/mask')

save_dir = 'trainA/'
save_dir_mask = 'trainmaskA/'

if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.makedirs(save_dir)

if os.path.exists(save_dir_mask):
    shutil.rmtree(save_dir_mask)
os.makedirs(save_dir_mask)

for nii_file_name in tqdm(nii_files):
    nii_file_path  = ct_dir + nii_file_name
    mask_file_path = mask_dir + nii_file_name

    # nii_file_path = 'slice_test/image/' + nii_file_name
    # mask_file_path = 'slice_test/mask/' + nii_file_name


    # Load the NIfTI file
    nii = nib.load(nii_file_path)
    mask = nib.load(mask_file_path)

    # Convert NIfTI file to numpy array
    volume = nii.get_fdata()
    mask_volume = mask.get_fdata()

    # Depending on the orientation, you might need to transpose or flip the array
    # Here, we assume the volume needs no such adjustment

    # Slice the volume
    # For example, to get the 50th axial slice:
    slice_len=volume.shape[2]

    save_file_name = nii_file_name.split('.')[0]

    index_list = random.sample(range(0, slice_len), 30)

    for i in index_list:
        flip = random.choice([True, False])
        rotate = random.choice([0])

        #Save mask slices, skip the no information ones
        mask_slice = mask_volume[:, :, i]
        if np.sum(mask_slice) == 0:
            # while True:
            #     new = random.randint(0, slice_len-1)
            #     if new not in index_list:
            #         break
            # index_list.append(new)
            continue
        # Assign an RGB value to each unique elements of the  mask,0 to (0,0,0), 1 to (100,0,100), 2 to (255, 255, 0), 3 to (255, 0, 255), 4 to (255,0,0)
        colored_slice = np.zeros((mask_slice.shape[0], mask_slice.shape[1], 3), dtype=np.uint8)
        colored_slice[mask_slice == 0] = [0, 0, 0]
        colored_slice[mask_slice == 1] = [100, 0, 100] #liver, purple
        colored_slice[mask_slice == 2] = [0, 255, 255] #kidney,yellow
        colored_slice[mask_slice == 3] = [255, 0, 255] #spleen, pink
        colored_slice[mask_slice == 4] = [255, 0, 0] #pancreas, blue

        colored_slice = cv2.rotate(colored_slice, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if flip:
            colored_slice = cv2.flip(colored_slice, 1)
        if rotate == 1:
            colored_slice = cv2.rotate(colored_slice, cv2.ROTATE_90_CLOCKWISE)
        elif rotate == 2:
            colored_slice = cv2.rotate(colored_slice, cv2.ROTATE_90_COUNTERCLOCKWISE)


        for j in range(colored_slice.shape[2]):
            colored_slice[:, :, j] = apply_fan_mask(colored_slice[:, :, j])

        # colored_slice = apply_fan_mask(colored_slice)
        cv2.imwrite(save_dir_mask + save_file_name + '_' + str(i).zfill(3) + '.png', colored_slice)

        ct_slice = volume[:, :, i]
        ct_slice = (255 * (ct_slice - np.min(ct_slice)) / (np.ptp(ct_slice)+0.1)).astype(int)
        ct_slice = cv2.rotate(ct_slice, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if flip:
            ct_slice = cv2.flip(ct_slice, 1)
        if rotate == 1:
            ct_slice = cv2.rotate(ct_slice, cv2.ROTATE_90_CLOCKWISE)
        elif rotate == 2:
            ct_slice = cv2.rotate(ct_slice, cv2.ROTATE_90_COUNTERCLOCKWISE)


        ct_slice = apply_fan_mask(ct_slice)
        cv2.imwrite(save_dir + save_file_name + '_' + str(i).zfill(3) + '.png', ct_slice)


        # plt.imshow(axial_slice, cmap='gray')
        # plt.savefig('ct_slice/train_0001' + str(i) + '.png')
        # plt.close()
        # mask_slice = mask[:, :, i]
        # plt.imshow(mask_slice, cmap='gray')
        # plt.savefig('mask_slice/train_0001'+str(i)+'.png')
        # plt.close()


    # # slice_index = 50
    # axial_slice = volume[:, :, slice_index]
    #
    # # Display the slice
    # plt.imshow(axial_slice, cmap='gray')
    # plt.axis('off')  # Remove axis for better visualization
    # plt.show()
