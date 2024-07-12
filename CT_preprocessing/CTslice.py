import os
import random
import numpy as np
import cv2
import nibabel as nib
from tqdm import tqdm
import shutil
import argparse

def apply_fan_mask(ct_img, center_x, center_y, radius, angle_range):
    xx, yy = np.meshgrid(range(ct_img.shape[1]), range(ct_img.shape[0]))
    r = np.sqrt((xx - center_x)**2 + (yy - center_y)**2)
    theta = np.arctan2(yy - center_y, xx - center_x) * 180 / np.pi
    theta[theta < 0] += 360

    fan_mask = np.zeros_like(ct_img, dtype=np.uint8)
    fan_mask[(r <= radius) & (theta >= angle_range[0]) & (theta <= angle_range[1])] = 255
    ct_img_masked = cv2.bitwise_and(ct_img, ct_img, mask=fan_mask)
    return ct_img_masked

def main(args):
    ct_dir = args.ct_dir
    mask_dir = args.mask_dir
    save_dir = args.save_dir
    save_dir_mask = args.save_dir_mask

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_dir_mask):
        os.makedirs(save_dir_mask)

    nii_files = os.listdir(ct_dir)
    for nii_file_name in tqdm(nii_files):
        nii_file_path = os.path.join(ct_dir, nii_file_name)
        mask_file_path = os.path.join(mask_dir, nii_file_name)

        nii = nib.load(nii_file_path)
        mask = nib.load(mask_file_path)

        volume = nii.get_fdata()
        mask_volume = mask.get_fdata()
        slice_len = volume.shape[2]
        index_list = random.sample(range(0, slice_len), 30)

        for i in index_list:
            ct_slice = volume[:, :, i]
            mask_slice = mask_volume[:, :, i]

            if np.sum(mask_slice) == 0:
                continue

            # Modify color mappings and rotations as required
            # Save the slices after applying transformations and masks

            # Sample code for saving images, adjust as per actual usage
            ct_slice_img = apply_fan_mask(ct_slice, args.center_x, args.center_y, args.radius, (args.angle_min, args.angle_max))
            mask_slice_img = apply_fan_mask(mask_slice, args.center_x, args.center_y, args.radius, (args.angle_min, args.angle_max))

            cv2.imwrite(os.path.join(save_dir, f"{nii_file_name}_{i:03d}.png"), ct_slice_img)
            cv2.imwrite(os.path.join(save_dir_mask, f"{nii_file_name}_{i:03d}.png"), mask_slice_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CT slices with a fan mask.")
    parser.add_argument("--ct_dir", type=str, required=True, help="Directory containing CT .nii files")
    parser.add_argument("--mask_dir", type=str, required=True, help="Directory containing mask .nii files")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save processed CT images")
    parser.add_argument("--save_dir_mask", type=str, required=True, help="Directory to save processed mask images")
    parser.add_argument("--center_x", type=int, default=256, help="Center X coordinate for fan mask")
    parser.add_argument("--center_y", type=int, default=-20, help="Center Y coordinate for fan mask")
    parser.add_argument("--radius", type=int, default=500, help="Radius for fan mask")
    parser.add_argument("--angle_min", type=float, default=60.0, help="Minimum angle for fan mask")
    parser.add_argument("--angle_max", type=float, default=120.0, help="Maximum angle for fan mask")
    args = parser.parse_args()

    main(args)
