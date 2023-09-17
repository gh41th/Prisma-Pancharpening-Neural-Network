# Import necessary libraries
import numpy as np
import h5py
import scipy
from sklearn.preprocessing import RobustScaler
from skimage import exposure
import math

# Function to load hyperspectral image data from an HDF5 file
def load_image_from_file(filename):
    """
    Load hyperspectral image data from an HDF5 file.

    Args:
        filename (str): The path to the HDF5 file containing hyperspectral data.

    Returns:
        tuple: A tuple containing three hyperspectral data arrays: SWIR, VNIR, and PAN.
    """
    with h5py.File(filename, "r") as f:
        # Load different spectral bands
        SWIR = np.array(f['HDFEOS']['SWATHS']['PRS_L2D_HCO']['Data Fields']['SWIR_Cube'])
        VNIR = np.array(f['HDFEOS']['SWATHS']['PRS_L2D_HCO']['Data Fields']['VNIR_Cube'])
        PAN = np.array(f['HDFEOS']['SWATHS']['PRS_L2D_PCO']['Data Fields']['Cube'])
        return np.transpose(SWIR, (0, 2, 1)), np.transpose(VNIR, (0, 2, 1)), PAN


# Function to downsample an image by a specified ratio
def downsample(image, ratio):
    """
    Downsample an image by a specified ratio.

    Args:
        image (numpy.ndarray): The input image to be downsampled.
        ratio (int): The downsampling ratio.

    Returns:
        numpy.ndarray: The downsampled image.
    """
    if len(image.shape) == 2:
        return image[::ratio, ::ratio]
    else:
        return image[::ratio, ::ratio, :]



# Function to upsample an image by a specified ratio
def upsample(image, ratio):
    """
    Upsample an image by a specified ratio using interpolation.

    Args:
        image (numpy.ndarray): The input image to be upsampled.
        ratio (int): The upsampling ratio.

    Returns:
        numpy.ndarray: The upsampled image.
    """
    k, l, m = image.shape
    X = np.zeros((k * ratio, l * ratio, m))
    for i in range(m):
        X[:, :, i] = scipy.ndimage.zoom(image[:, :, i], ratio, order=3)
    return X



# Function to clip 1% of histogram values for each band to prevent extreme values
def clip(input):
    """
    Clip 1% of histogram values for each band to prevent extreme values.

    Args:
        input (numpy.ndarray): The input hyperspectral data.

    Returns:
        numpy.ndarray: The clipped hyperspectral data.
    """
    rows, cols, bands = np.shape(input)
    input2 = np.zeros((rows, cols, bands), dtype='uint16')
    for i in range(bands):
        p1 = np.percentile(input[:, :, i], 0.5)
        p99 = np.percentile(input[:, :, i], 99.5)
        min_val = np.min(input[:, :, i])
        max_val = np.max(input[:, :, i])
        input2[:, :, i] = exposure.rescale_intensity(input[:, :, i], in_range=(p1, p99), out_range=(min_val, max_val))
    return input2



# Function to rescale data using RobustScaler
def rescale(data, scaler=None, pan_scaler=None, input_scaling=True):
    """
    Rescale hyperspectral data using RobustScaler.

    Args:
        data (numpy.ndarray): The input hyperspectral data.
        scaler (sklearn.preprocessing.RobustScaler, optional): An optional pre-defined scaler for feature bands.
        pan_scaler (sklearn.preprocessing.RobustScaler, optional): An optional pre-defined scaler for the PAN band.
        input_scaling (bool): Flag to indicate whether input scaling should be applied.

    Returns:
        tuple: A tuple containing the scaler for feature bands, scaler for PAN band, and scaled hyperspectral data.
    """
    rows, cols, bands = data.shape
    data_reshaped = data.reshape(rows * cols, bands)

    if input_scaling:
        cube, pan = data_reshaped[:, :-1], data_reshaped[:, -1].reshape(-1, 1)
        if scaler is None:
            scaler = RobustScaler(quantile_range=(0.5, 99.5))
            cube = scaler.fit_transform(cube)
        else:
            cube = scaler.transform(cube)
        if pan_scaler is None:
            pan_scaler = RobustScaler(quantile_range=(0.5, 99.5))
            pan = pan_scaler.fit_transform(pan)
        else:
            pan = pan_scaler.transform(pan)

        scaled_data = np.concatenate([cube, pan], axis=1).reshape(rows, cols, bands)
        return scaler, pan_scaler, scaled_data
    else:
        if scaler is None:
            raise ValueError("You should specify an already defined scaler when input_scaling is False")

        scaled_data = scaler.transform(data_reshaped)
        return scaled_data.reshape(rows, cols, bands)




# Function to create training patches from input and output arrays
def create_patches(input_arr, output_arr, patch_size):
    """
    Create training patches from input and output hyperspectral data arrays.

    Args:
        output_arr (numpy.ndarray): The output (target) hyperspectral data.
        input_arr (numpy.ndarray): The input (source) hyperspectral data.
        patch_size (int): The size of the square patches to be created.

    Returns:
        tuple: A tuple containing input training patches and output training patches.
    """
    # Get the dimensions of the input image
    rows, cols, bandsin = input_arr.shape

    # Calculate the size of padded image (divisible by patch_size)
    y_size = math.ceil(rows / patch_size) * patch_size
    x_size = math.ceil(cols / patch_size) * patch_size

    # Calculate padding
    y_pad = y_size - rows
    x_pad = x_size - cols

    # Create padded input and output images
    input_arraypad = np.pad(input_arr, ((0, y_pad), (0, x_pad), (0, 0)), mode='constant')
    output_arraypad = np.pad(output_arr, ((0, y_pad), (0, x_pad), (0, 0)), mode='constant')

    # Create input and output training patches with overlap
    input_list_patches = []
    output_list_patches = []

    for i in range(0, y_size - patch_size + 1,  patch_size // 2):  # 50% overlap
        for j in range(0, x_size - patch_size + 1,  patch_size // 2):
            input_patch = input_arraypad[i:i + patch_size, j:j + patch_size, :]
            output_patch = output_arraypad[i:i + patch_size, j:j + patch_size, :]

            # Check if the PAN layer of the input patch is not empty (contains non-zero pixels)
            if np.count_nonzero(input_patch[:, :, -1]) > patch_size ** 2 * 0.5:
                input_list_patches.append(input_patch)
                output_list_patches.append(output_patch)

    input_patches = np.array(input_list_patches, dtype='float16')
    output_patches = np.array(output_list_patches, dtype='float16')

    return input_patches, output_patches
