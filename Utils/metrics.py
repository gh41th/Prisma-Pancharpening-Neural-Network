import numpy as np
import cv2
import tensorflow as tf
from keras import backend as K

def tf_log10(x):
    """
    Compute logarithm base 10 of a tensor.
    
    Args:
    - x: Input tensor.
    
    Returns:
    - Logarithm base 10 of the input tensor.
    """
    n = tf.math.log(x)
    d = tf.math.log(tf.constant(10, dtype=n.dtype))
    return n / d

def psnr(y_true, y_pred):
    """
    Compute Peak Signal to Noise Ratio (PSNR) between two images.
    
    Args:
    - y_true: Ground truth image.
    - y_pred: Predicted image.
    
    Returns:
    - PSNR value.
    """
    max_pixel = 1.0
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))

def ergas(img_fake, img_real, scale=4):
    """
    Compute ERGAS (Relative Dimensionless Global Error) for 2D or 3D images.
    
    Args:
    - img_fake: Generated or fake image.
    - img_real: Ground truth or real image.
    - scale: Spatial resolution scale factor (default is 4).
    
    Returns:
    - ERGAS value.
    """
    if not img_fake.shape == img_real.shape:
        raise ValueError('Input images must have the same dimensions.')
    
    img_fake_ = img_fake.astype(np.float64)
    img_real_ = img_real.astype(np.float64)
    
    if img_fake_.ndim == 2:
        mean_real = img_real_.mean()
        mse = np.mean((img_fake_ - img_real_)**2)
        return 100 / scale * np.sqrt(mse / (mean_real**2 + np.finfo(np.float64).eps))
    elif img_fake_.ndim == 3:
        means_real = img_real_.reshape(-1, img_real_.shape[2]).mean(axis=0)
        mses = ((img_fake_ - img_real_)**2).reshape(-1, img_fake_.shape[2]).mean(axis=0)
        return 100 / scale * np.sqrt((mses / (means_real**2 + np.finfo(np.float64).eps)).mean())
    else:
        raise ValueError('Wrong input image dimensions.')

def _ssim(img1, img2, dynamic_range=255):
    """
    Compute Structural Similarity Index (SSIM) for 2D image.
    
    Args:
    - img1: First input image.
    - img2: Second input image.
    - dynamic_range: Dynamic range of pixel values (default is 255 for uint8).
    
    Returns:
    - SSIM map.
    """
    C1 = (0.01 * dynamic_range) ** 2
    C2 = (0.03 * dynamic_range) ** 2

    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)  # kernel size 11
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1_, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2_, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1_ ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2_ ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1_ * img2_, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def ssim(img1, img2, dynamic_range=255):
    """
    Compute Structural Similarity Index (SSIM) for 2D or 3D images.
    
    Args:
    - img1: First input image.
    - img2: Second input image.
    - dynamic_range: Dynamic range of pixel values (default is 255 for uint8).
    
    Returns:
    - SSIM value.
    """
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    
    if img1.ndim == 2:
        return _ssim(img1, img2, dynamic_range)
    elif img1.ndim == 3:
        ssims = [_ssim(img1[..., i], img2[..., i], dynamic_range) for i in range(img1.shape[2])]
        return np.array(ssims).mean()
    else:
        raise ValueError('Wrong input image dimensions.')

def sam(img1, img2):
    """
    Compute Spectral Angle Mapper (SAM) for 3D image.
    
    Args:
    - img1: First input image.
    - img2: Second input image.
    
    Returns:
    - SAM value.
    """
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    
    assert img1.ndim == 3 and img1.shape[2] > 1, "image n_channels should be greater than 1"
    
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    
    inner_product = (img1_ * img2_).sum(axis=2)
    img1_spectral_norm = np.sqrt((img1_ ** 2).sum(axis=2))
    img2_spectral_norm = np.sqrt((img2_ ** 2).sum(axis=2))
    
    # numerical stability
    cos_theta = (inner_product / (img1_spectral_norm * img2_spectral_norm + np.finfo(np.float64).eps)).clip(min=0, max=1)
    
    return np.mean(np.arccos(cos_theta))
