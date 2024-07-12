import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
import torch
import torchvision

class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        #mask_path
        self.mask_dir_A = os.path.join(opt.dataroot, opt.phase + 'maskA')
        self.mask_dir_B = os.path.join(opt.dataroot, opt.phase + 'maskB')


        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        self.mask_A_paths = [os.path.join(self.mask_dir_A,mask_path.split("/")[-1]) for mask_path in self.A_paths]
        self.mask_B_paths = [os.path.join(self.mask_dir_B,mask_path.split("/")[-1]) for mask_path in self.B_paths]


        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)

        B_path = self.B_paths[index_B]

        mask_A_path = self.mask_A_paths[index % self.A_size]
        mask_B_path = self.mask_B_paths[index_B]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # convert = torchvision.transforms.ToTensor()
        #
        # A = convert(A_img)
        # B = convert(B_img)
        #
        mask_A_img = Image.open(mask_A_path).convert('RGB')
        mask_B_img = Image.open(mask_B_path).convert('RGB')

        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        # A = np.asarray(A_img)
        # B= np.asarray(B_img)
        mask_A = rgb_to_one_hot(mask_A_img)
        mask_B = rgb_to_one_hot(mask_B_img)

        mask_A_index = rgb_to_index_mask(mask_A_img)
        mask_B_index = rgb_to_index_mask(mask_B_img)




        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'A_M': mask_A, 'B_M': mask_B, 'A_M_index': mask_A_index, 'B_M_index':mask_B_index,'A_M_paths': mask_A_path, 'B_M_paths': mask_B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)


# Define your color to class index mapping here, including the background as class 0



def rgb_to_index_mask(mask_image):
    color_to_index = {
        (0, 0, 0): 0,       # Background
        (100, 0, 100): 1,   # Liver, purple
        (255, 255, 0): 2,   # Kidney, yellow
        (255, 0, 255): 3,   # Spleen, pink
        (0, 0, 255): 4      # Pancreas, blue
    }
    mask_array = np.array(mask_image)
    index_mask = np.zeros((mask_array.shape[0], mask_array.shape[1]), dtype=np.int32)

    for color, index in color_to_index.items():
        matches = (mask_array == color).all(axis=-1)
        index_mask[matches] = index

    # Convert to a PyTorch tensor, add a channel dimension, and permute to C, H, W
    index_mask_tensor = torch.from_numpy(index_mask).long()  # Shape will be [H, W]
    index_mask_tensor = index_mask_tensor.unsqueeze(0)  # Shape becomes [1, H, W], adding a singleton channel dimension

    return index_mask_tensor

def rgb_to_one_hot(mask_image):
    color_to_index = {
        (0, 0, 0): 0,  # Background
        (100, 0, 100): 1,  # Liver, purple
        (255, 255, 0): 2, #kidney, yellow
        (255, 0, 255): 3, #spleen, pink
        (0, 0, 255): 4, #pancreas, blue
        # Add additional classes as needed
    }
    """
    Convert an RGB mask image to a one-hot encoded mask with C channels,
    where C is the number of classes.

    Args:
    - mask_image (PIL Image): The RGB image to convert.
    - color_to_index (dict): A mapping from color tuples to class indices.

    Returns:
    - torch.Tensor: The one-hot encoded mask.
    """
    # Convert PIL Image to numpy array
    mask_array = np.array(mask_image)
    # Create a one-hot encoded mask with shape (H, W, C)
    one_hot_mask = np.zeros((mask_array.shape[0], mask_array.shape[1], len(color_to_index)), dtype=np.float32)

    for color, index in color_to_index.items():
        # Find pixels matching the color and set the corresponding index
        matches = (mask_array == color).all(axis=-1)
        one_hot_mask[matches, index] = 1.0

    # Convert to a PyTorch tensor and permute to (C, H, W)
    one_hot_mask = torch.from_numpy(one_hot_mask).permute(2, 0, 1)
    #one_hot_mask = torch.argmax(one_hot_mask, dim=0)

    return one_hot_mask