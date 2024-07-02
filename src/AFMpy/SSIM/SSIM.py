import numpy as np
from skimage.metrics import structural_similarity as ssim

from AFMpy import Utilities

def mean_masked_SSIM(img1: np.ndarray,
                     img2: np.ndarray,
                     threshold_rel: float = 0.0,
                     **kwargs):
    
    '''
    Calculates the average masked SSIM between two images. The mask is defined by pixels above a certain threshold in either image.

    Args:
        img1 (np.ndarray):
            The base image for comparison.
        img2 (np.ndarray):
            The image to compare to the base image.
        threshold_rel (float):
            The relative threshold for the mask. Pixels above this threshold in either image will be included in the mask.
        **kwargs:
            Additional keyword arguments to pass to skimage.metrics.structural_similarity.
    Returns:
        float:
            The masked SSIM between the two images.
    '''
    
    # Calculate the data range
    data_range = np.max([img1,img2])

    threshold = threshold_rel * data_range

    # Create the mask    
    mask = np.logical_or(img1>threshold, img2>threshold)
    
    _, ssim_image = ssim(img1, img2, data_range = data_range, full = True, **kwargs)
    
    return np.mean(ssim_image[mask])