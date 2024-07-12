"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import torch.nn.functional as F


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def masktensor2im(input_image, imtype=np.uint8):

    """
    Convert a tensor of shape [B, H, W], where each entry is a class index,
    to a numpy array representing images with class-specific colors.

    Args:
    - indices_tensor (torch.Tensor): Tensor of class indices.
    - index_to_color (dict): Mapping from class indices to RGB colors.

    Returns:
    - numpy.ndarray: Array of shape [B, H, W, 3] with RGB colors.
    """

    class_indices = indices_to_rgb(torch.argmax(F.softmax(input_image, dim=1), dim=1))
    class_indices = class_indices[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    # Initialize an empty RGB image array

    return class_indices.astype(imtype)

def indices_to_rgb(tensor):
    colormap = {
        0: (0, 0, 0),  # Background
        1: (100, 0, 100),  # Liver, purple
        2: (255, 255, 0),  # kidney, yellow
        3: (255, 0, 255),  # spleen, pink
        4: (0, 0, 255),  # pancreas, blue
        # Add additional classes as needed
    }
    """
    Convert a tensor of class indices to an RGB image.

    Args:
    - tensor (torch.Tensor): A tensor of shape [B, H, W] containing class indices.
    - colormap (dict): A mapping from class indices to RGB values.

    Returns:
    - torch.Tensor: A tensor of shape [B, 3, H, W] containing RGB values.
    """
    # Initialize the RGB tensor
    rgb_tensor = torch.zeros((tensor.size(0), 3, tensor.size(1), tensor.size(2)), dtype=torch.float)

    # Apply the colormap
    for class_index, color in colormap.items():
        # Create a mask for the current class index
        mask = tensor == class_index
        # Place the RGB values
        for channel, rgb_value in enumerate(color):
            rgb_tensor[:, channel, :, :][mask] = rgb_value

    return rgb_tensor

def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
