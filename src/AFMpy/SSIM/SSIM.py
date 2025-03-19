import numpy as np
from skimage.metrics import structural_similarity as ssim

from AFMpy import REC

__all__ = ['masked_SSIM', 'registered_SSIM']

def masked_SSIM(img1: np.ndarray,
                img2: np.ndarray,
                threshold_rel: float = 0.05,
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

def registered_SSIM(ref: np.ndarray,
                    mov: np.ndarray,
                    threshold_rel: float = 0.05,
                    method: str = 'rigid',
                    **kwargs):
    '''
    Calculates the average masked SSIM between two unaligned images.
    First, the images are aligned using the method specified in the method argument.
    Then, the masked SSIM is calculated using the masked_SSIM function.

    Args:
        ref (np.ndarray):
            The reference image.
        mov (np.ndarray):
            The image to compare to the reference image.
        threshold_rel (float):
            The relative threshold for the mask. Pixels above this threshold in either image will be included in the mask.
        method (str):
            The registration method to use. Options are 'rigid' and 'affine'.
        **kwargs:
            Additional keyword arguments to pass to skimage.metrics.structural_similarity.

    '''
    registered_im = REC.register_image(ref, mov, method = method)

    return masked_SSIM(ref, registered_im, threshold_rel = threshold_rel, **kwargs)