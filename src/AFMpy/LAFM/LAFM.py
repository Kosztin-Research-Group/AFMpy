import numpy as np
import logging
import cv2
from skimage.exposure import rescale_intensity
from skimage.feature import peak_local_max

from typing import Tuple

from AFMpy import Utilities

logger = logging.getLogger(__name__)

__all__ = ['LAFM2D']

### Functions for Grayscale (0-1) Color Mapping
_red_func = lambda z: -z**2+2*z
_green_func = lambda z: _red_func(z)*z
_blue_func = lambda z: (z*(np.sin(3*np.pi*(z+1/2))+1)/2)

### Functions for Grayscale (0-255) Color Mapping 
_N = 256
_R = lambda z: -z**2/_N+2*z
_G = lambda z: _R(z)*z/_N
_B = lambda z: z*(np.sin(0.037*(z+_N/2))+1)/2

def LAFM2D(stack: np.ndarray,
           target_resolution: Tuple[int, int],
           sigma: float, **peak_local_max_kwargs):
    """
    Generates an LAFM image with real space height pixel intensity from a stack of AFM images.

    Args:
        stack (numpy.ndarray):
            Stack of N aligned AFM images with shape (N, X, Y). Each image represents the pixel resolution of the AFM scan.
        resize_factor (int):
            The factor by which to expand the image. For example, if each AFM image is (64,64) and resize_factor is 3, 
            the images will be interpolated to (192,192) using bicubic interpolation.
        sigma (float):
            Gaussian broadening width for generating the peak probability distribution. This value is in terms of pixels of 
            the expanded image. For example, if the stack is expanded by a factor of 3 and the original resolution is 1.5 
            Angstroms per pixel, applying a Gaussian broadening of 3 pixels will result in a 1.5 Angstrom broadening.
        **peak_local_max_kwargs:
            Additional keyword arguments to be passed to skimage.feature.peak_local_max. Refer to the skimage documentation 
            for more details: https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.peak_local_max

    Returns:
        numpy.ndarray:
            LAFM image with shape (X*resize_factor, Y*resize_factor). The pixel intensity corresponds to the real space 
            height of the sample.
    """
    ### Normalizing the stack
    normalized_stack = Utilities.Math.norm(stack, (0,1))

    ### Creating an empty list to store the LAFM contributions from each image in the stack
    lafm_stack = []

    ### Looping over each image in the stack
    for image, norm_image in zip(stack, normalized_stack):

        ### Resizing the image and normalized image by the resize factor
        expanded_image = cv2.resize(image,
                                    target_resolution, 
                                    cv2.INTER_CUBIC)
        
        expanded_norm_image = cv2.resize(norm_image,
                                         target_resolution,
                                         cv2.INTER_CUBIC)
        
        ### Extracting local maxima from the normalized image
        peaks = peak_local_max(expanded_image, **peak_local_max_kwargs)

        ### Extracing local maxima to create peak height array
        maxima = np.zeros(expanded_norm_image.shape)
        for (x,y) in peaks:
            ## Scaled Maxima
            maxima[x,y] = expanded_norm_image[x,y]

        ### Applying Gaussian broadening to the peak height array
        blurred = cv2.GaussianBlur(maxima, ksize = [0,0], sigmaX = sigma, sigmaY = sigma) #* np.max(maxima)
        
        ### Normalizing the broadened peak height array and multiplying by the image to create peaking
        ### probability contribution.
        pG = rescale_intensity(blurred, out_range = (0, np.max(expanded_norm_image)))# *expanded_image

        ## Multiplying grayscale image by peaking probability.
        combined_im = expanded_image * pG

        ### Stacking the RGB channels to have the correct shapt of (X*resize_factor, Y*resize_factor, 3)
        ### Then appdnding the RGB LAFM contribution to the LAFM stack
        lafm_stack.append(combined_im)
    
    ### Averaging per pixel over all LAFM contributions from each image in the stack.
    ### Rescaling the intensity of the image to increase visibility, and outputting.
    return  rescale_intensity(np.mean(np.array(lafm_stack), axis = 0),
                              out_range = (0, np.max(np.mean(stack, axis = 0))))