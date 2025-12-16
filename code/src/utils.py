"""utils.py

    This module provides utility functions for MRI data processing, including coil compression, image cropping, montage creation, and error calculation.
    Functions:
    ----------
    coil_compression(ksp, percentile=None)
        Performs coil compression on k-space data, supporting both CPU and GPU arrays. Optionally reports the number of coils containing a specified percentile of total energy and visualizes energy contributions before and after compression.
    crop_center(img, target_shape)
        Crops a 4D MRI image (x, y, z, N) from the center to a specified target shape (target_x, target_y, target_z, N).
    crop_2d(img, target_shape)
        Crops a 2D MRI image (x, y) from the center to a specified target shape (target_x, target_y).
    create_montage(input, slices=None, cmap='gray', percentile=None, vmin=0, vmax=1)
        Creates a 2D montage from a 3D array (N_pixel x N_pixel x N_slice), stacking selected slices horizontally for visualization.
    nrmse(image1, image2, mask=None, norm='range')
        Calculates the Normalized Root Mean Squared Error (NRMSE) between two images, with optional masking and normalization strategies ('range', 'mean', 'std').
"""


import numpy as np
import sigpy as sp
import sys
import os
import time
from tqdm.auto import tqdm
import scipy.io as sio
import matplotlib.pyplot as plt
sys.path.append('../src')
sys.path.append('../lib')


def coil_compression(ksp, percentile=None):
    """
    Coil compression code that supports CPU and GPU arrays.
    
    Parameters:
    ksp (numpy.ndarray or cupy.ndarray): 3D array of shape (coils, readout, samples per readout).
    percentile (float, optional): Percentile to print the number of coils that contain the percent amount of total energy. 
    
    Returns:
    numpy.ndarray or cupy.ndarray: ksp array with coils compressed and reordered from highest energy to lowest energy.
    """
    device = sp.get_device(ksp)
    ksp_shape = ksp.shape
    nc = ksp_shape[0]

    # Compute energy contribution of each coil before compression
    coil_energies = device.xp.linalg.norm(ksp, axis=(1, 2))**2  # L2 norm squared per coil
    coil_energies /= coil_energies.sum()  # Normalize to get percentage contribution

    # Reshape and calculate EHE
    E = device.xp.reshape(ksp, [nc, -1])
    E = device.xp.swapaxes(E, 0, 1)

    EHE = device.xp.matmul(E.conj().T, E)
    eigval, eigvec = device.xp.linalg.eigh(EHE)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = device.xp.argsort(eigval)[::-1]
    eigval = eigval[idx]
    eigvec = eigvec[:, idx]

    # Calculate the compressed ksp
    E = device.xp.matmul(E, eigvec)
    ksp = device.xp.reshape(device.xp.swapaxes(E, 0, 1), ksp_shape)
    
    # Move data to CPU for plotting
    coil_energies = sp.to_device(coil_energies, -1)
    eigval = sp.to_device(eigval, -1)

    # Plot coil energy contributions before and after compression
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    # Before compression: Energy contribution per coil
    axes[0].bar(range(nc), coil_energies * 100, color='blue')
    axes[0].set_title("Energy contribution per coil \n(before compression)", fontsize=16)
    axes[0].set_xlabel("Coil Index", fontsize=16)
    axes[0].set_ylabel("Energy contribution (%)", fontsize=16)
    axes[0].grid(True)

    # After compression: Energy from eigenvalues
    energy_total = np.sum(eigval)
    energy_percent = (eigval / energy_total) * 100
    axes[1].bar(range(nc), energy_percent, color='red')
    axes[1].set_title("Energy contribution per coil \n(after compression)", fontsize=16)
    axes[1].set_xlabel("Compressed coil index", fontsize=16)
    axes[1].set_ylabel("Energy contribution (%)", fontsize=16)
    axes[1].grid(True)

    plt.show()
    
    # Calculate the cumulative energy to estimate XX% energy threshold
    if percentile is not None:
        cumulative_energy = np.cumsum(eigval) / energy_total
        num_coils_percentile = np.searchsorted(cumulative_energy, percentile / 100) + 1  
        num_coils_percentile = min(num_coils_percentile, nc)  # Cap at total coils
        
        print(f"Percent energy per coil: {np.round(energy_percent, decimals=2)}")
        print(f'{num_coils_percentile} coils contain {percentile}% of the total energy (for reference only).')

        return ksp, eigval, num_coils_percentile
    
    else:
        return ksp, eigval


def crop_center(img, target_shape):
    """
    Crops the input 4D MRI image `img` (x, y, z, N) from the center to the given `target_shape`.
    
    Parameters:
    img (numpy.ndarray): 4D array of shape (x, y, z, N).
    target_shape (tuple): Desired output shape (target_x, target_y, target_z, N).
    
    Returns:
    numpy.ndarray: Cropped image of shape `target_shape`.
    """
    x, y, z, N = img.shape
    target_x, target_y, target_z = target_shape # Not including number of resp phases

    # Calculate the cropping start and end points for each dimension
    start_x = (x - target_x) // 2
    end_x = start_x + target_x

    start_y = (y - target_y) // 2
    end_y = start_y + target_y

    start_z = (z - target_z) // 2
    end_z = start_z + target_z

    # Crop the image in the x, y, z dimensions (N is unchanged)
    cropped_img = img[start_x:end_x, start_y:end_y, start_z:end_z, ]

    return cropped_img


def crop_2d(img, target_shape):
    """
    Crops the input 2D MRI image `img` (x, y) from the center to the given `target_shape`.
    
    Parameters:
    img (numpy.ndarray): 2D array of shape (x, y).
    target_shape (tuple): Desired output shape (target_x, target_y).
    
    Returns:
    numpy.ndarray: Cropped image of shape `target_shape`.
    """
    x, y = img.shape
    target_x, target_y = target_shape # Not including number of resp phases

    # Calculate the cropping start and end points for each dimension
    start_x = (x - target_x) // 2
    end_x = start_x + target_x

    start_y = (y - target_y) // 2
    end_y = start_y + target_y

    # Crop the image in the x, y, z dimensions 
    cropped_img = img[start_x:end_x, start_y:end_y]

    return cropped_img

def create_montage(input, slices=None, cmap='gray', percentile=None, vmin=0, vmax=1):
    """
    A function that creates a 2D N_pixel x (N_slice * N_pixel) array from a 3D N_pixel x N_pixel x N_slice array. 

    Args:
        input (ndarray): 3D array, in form N_pixel x N_pixel x N_slice.
        slices (list of ints): slices to make into montage. Defaults to middle 5 slices.
    """
    if np.size(np.shape(input)) != 3:
        raise ValueError("Incorrect data dimensions.")

    if slices == None:
        median = np.shape(input)[2]//2
        slices = [median-2, median-1, median, median+1, median+2]

    output = input[:, :, slices[0]]
    for i in slices[1:]:
        temp = input[:, :, i]
        output = np.hstack((output, temp))

    if percentile is not None:
        montage = output / np.percentile(np.ravel(output), percentile)
    else:
        montage = output

    img_array_size = (2*len(slices) + 1, 2)
    plt.figure(figsize=img_array_size, dpi=100)
    plt.imshow(montage, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis('off')
    cbar = plt.colorbar(location='right',fraction=0.046, pad=0.04)
    cbar.set_ticks(ticks=[vmin, vmax])
    cbar.ax.set_yticklabels([f'{vmin:.1f}', f'{vmax:.1f}'])
    cbar.ax.tick_params(labelsize=16)
    plt.show()

    return output



def nrmse(image1, image2, mask=None, norm='range'):
    """
    Calculate the Normalized Root Mean Squared Error (NRMSE) between two images.

    Parameters:
        image1 (np.ndarray): The first image.
        image2 (np.ndarray): The second image.
        mask (np.ndarray, optional): A binary mask to specify the region of interest.
                                     Should have the same dimensions as the images.
        norm (str): The normalization strategy. Options are:
            'range' (default): Normalize by the range (max - min) of the region.
            'mean': Normalize by the mean of the region.
            'std': Normalize by the standard deviation of the region.
    
    Returns:
        float: The NRMSE value.
    """
    # Ensure images are NumPy arrays
    image1 = np.array(image1)
    image2 = np.array(image2)

    # Check if images have the same shape
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions.")

    # Apply mask if provided
    if mask is not None:
        mask = np.array(mask, dtype=bool)
        if mask.shape != image1.shape:
            raise ValueError("Mask must have the same dimensions as the images.")
        image1 = image1[mask]
        image2 = image2[mask]

    # Compute RMSE
    mse = np.mean((image1 - image2) ** 2)
    rmse = np.sqrt(mse)

    # Compute normalization factor
    if norm == 'range':
        normalization_factor = np.ptp(image1)  # Peak-to-peak range
    elif norm == 'mean':
        normalization_factor = np.mean(image1)
    elif norm == 'std':
        normalization_factor = np.std(image1)
    else:
        raise ValueError("Invalid normalization strategy. Choose from 'range', 'mean', or 'std'.")

    # Handle division by zero in normalization
    if normalization_factor == 0:
        raise ValueError("Normalization factor is zero. Check your input data or mask.")

    # Calculate NRMSE
    nrmse = rmse / normalization_factor
    return nrmse
