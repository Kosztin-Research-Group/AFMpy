import numpy as np
import cv2
from typing import Tuple, Dict
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import rbf_kernel

from pystackreg import StackReg

from pyAFM.DL import Models, Losses
from pyAFM import Utilities

__all__ = ['center_image', 'center_stack', 'register_image', 'register_stack', 'spectral_cluster']

logger = Utilities.Logging.make_module_logger(__name__)

##################################
##### Registration Functions #####
##################################

# Define the registration methods
registration_methods: Dict[str, int] = {
    'rigid': StackReg.RIGID_BODY,
    'affine': StackReg.AFFINE
}

def _center_of_geometry(image: np.ndarray,
                        threshold: float = 0.0) -> Tuple[float, float]:
    '''
    Determines the center of geometry of an image.

    Args:
        image (np.ndarray):
            The input image to find the center of geometry of.
        threshold (float):
            The threshold to determine which pixels to include in the center of geometry calculation.
    Returns:
        Tuple[float, float]: The center of geometry of the image with subpixel accuracy.
                             (X coordinate, Y coordinate)
    '''
    # Get the indices of the image
    y, x = np.indices(image.shape)

    # Get the pixels above the threshold
    on_pixels = image > threshold
    total_on_pixels = on_pixels.sum()

    # If there are no pixels above the threshold, return the center of the image
    if total_on_pixels == 0:
        return image.shape[1] / 2, image.shape[0] / 2

    # Calculate the center of geometry
    x_cg = (x[on_pixels].sum()) / total_on_pixels
    y_cg = (y[on_pixels].sum()) / total_on_pixels

    return (x_cg, y_cg)

def _center_of_mass(image: np.ndarray,
                    threshold: float = 0.0) -> Tuple[float, float]:
    '''
    Determines the center of mass of an image.

    Args:
        image (np.ndarray):
            The input image to find the center of mass of.
        threshold (float):
            The threshold to determine which pixels to include in the center of mass calculation.
    Returns:
        Tuple[float, float]: The center of mass of the image with subpixel accuracy.
                             (X coordinate, Y coordinate)
    '''
    # Get the indices of the image
    y, x = np.indices(image.shape)

    # Filter the image based on the threshold
    filtered_image = np.where(image > threshold, image, 0)

    # Calculate the total mass of the image
    total_mass = filtered_image.sum()

    # If there are no pixels above the threshold, return the center of the image
    if total_mass == 0:
        return image.shape[1] / 2, image.shape[0] / 2
    
    # Calculate the center of mass
    x_cm = (x * filtered_image).sum() / total_mass
    y_cm = (y * filtered_image).sum() / total_mass

    return (x_cm, y_cm)

# Define the centering methods
centering_methods: Dict[str, callable] = {
    'cog': _center_of_geometry,
    'com': _center_of_mass
}

def _translate_image(image: np.ndarray,
                     translation: Tuple[float, float]) -> np.ndarray:
    '''
    Translates an image by the given translation.

    Args:
        image (np.ndarray):
            The input image to translate.
        translation (Tuple[float, float]):
            The translation to apply to the image.
    Returns:
        np.ndarray: The translated image.
    '''
    # Get the translation components
    x_translation, y_translation = translation

    # Create the translation matrix
    translation_matrix = np.array([[1, 0, x_translation],
                                    [0, 1, y_translation]], dtype = np.float32)

    # Translate the image
    translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]),
                                      flags = cv2.INTER_CUBIC, borderMode = cv2.BORDER_CONSTANT, borderValue = 0)

    return translated_image

def center_image(image: np.ndarray,
                 method: str = 'cog',
                 threshold: float = 0.0) -> np.ndarray:
    '''
    Centers an image using the given method.
    
    Args:
        image (np.ndarray):
            The input image to center.
        method (str):
            The method to use to center the image. Options are available in the centering_methods dictionary.
        threshold (float):
            The threshold to use for the centering method.
    Returns:
        np.ndarray:
            The centered image.
    '''
    # Check if the method is valid
    if method not in centering_methods.keys():
        logger.error(f'Invalid centering method: {method}. Valid options are {list(centering_methods.keys())}.')
        raise ValueError(f'Invalid centering method: {method}. Valid options are {list(centering_methods.keys())}.')
    
    center_method = centering_methods[method]

    # Get the center of the image
    x_center, y_center = center_method(image, threshold)

    # Calculate the translation
    dx = image.shape[1] / 2 - x_center
    dy = image.shape[0] / 2 - y_center

    # Translate the image
    centered_image = _translate_image(image, (dx, dy))

    return centered_image

def center_stack(stack: np.ndarray,
                 method: str = 'cog',
                 threshold: float = 0.0) -> np.ndarray:
    '''
    Centers a stack of images using a given method.

    Args:
        stack (np.ndarray):
            The stack of images to center.
        method (str):
            The method to use to center the stack. Options are available in the centering_methods dictionary.
        threshold (float):
            The threshold to use for the centering method.

    Returns:
        np.ndarray:
            The centered stack.
    '''
    # Check if the centering method is valid.
    if method not in centering_methods:
        logger.error(f'Invalid method: {method}. Valid methods are: {list(centering_methods.keys())}')
        raise ValueError(f'Invalid method: {method}. Valid methods are: {list(centering_methods.keys())}')

    # Create an empty stack to hold the centered images
    centered_stack = np.empty_like(stack)

    # Center each image in the stack
    for i, image in enumerate(stack):
        centered_stack[i] = center_image(image, method=method, threshold=threshold)

    return centered_stack

def register_image(ref: np.ndarray,
                   mov: np.ndarray,
                   method: str = 'rigid') -> np.ndarray:
    '''
    Registers a moving image to a reference image using a given method. 
    Also used as a helper function for register_stack.

    Args:
        ref (np.ndarray):
            The reference image.
        mov (np.ndarray):
            The moving image.
        method (str):
            The method to use to register the images. Options are available in the registration_methods dictionary.
    Returns:
        np.ndarray: The registered image.
    '''
    # Check if the method is valid
    if method not in registration_methods:
        logger.error(f'Invalid method: {method}. Valid methods are: {list(registration_methods.keys())}')
        raise ValueError(f'Invalid method: {method}. Valid methods are: {list(registration_methods.keys())}')

    # Get the registration method
    registration_method = registration_methods[method]
    # Create the stackreg object
    sr = StackReg(registration_method)
    # Register the images
    registered_image = sr.register_transform(ref, mov)

    # Some pixel values may be negative, or very close to zero. This can cause issues elsewhere,
    # so we set them to zero with a mask.
    mask = np.logical_not(np.isclose(registered_image, 0, atol=1e-5)) * (registered_image > 0)
    registered_image = registered_image * mask

    return registered_image

def register_stack(ref: np.ndarray,
                   stack: np.ndarray,
                   method: str = 'rigid') -> np.ndarray:
    '''
    Registers a stack of images to the first image in the stack using a given method.

    Args:
        ref (np.ndarray):
            The reference image.
        stack (np.ndarray):
            The stack of images to register. Each image in the stack acts as the moving image.
        method (str):
            The method to use to register the images. Options are available in the registration_methods dictionary.
    Returns:
        np.ndarray: The registered stack.
    '''
    # Check if the method is valid
    if method not in registration_methods:
        logger.error(f'Invalid method: {method}. Valid methods are: {list(registration_methods.keys())}')
        raise ValueError(f'Invalid method: {method}. Valid methods are: {list(registration_methods.keys())}')

    # Create an empty stack to hold the registered images
    registered_stack = np.empty_like(stack)

    # Register each image in the stack
    for i, image in enumerate(stack):
        registered_stack[i] = register_image(ref, image, method=method)

    return registered_stack

##############################
##### Clustering Methods #####
##############################

def spectral_cluster(affinity: np.ndarray,
                     n_clusters: int = 2,
                     **SC_kwargs) -> np.ndarray:
    '''
    Performs spectral clustering on the given affinity matrix.

    Args:
        affinity (np.ndarray):
            The affinity matrix to use for clustering.
        n_clusters (int):
            The number of clusters to create.
        **SC_kwargs:
            Keyword arguments to pass to the SpectralClustering object.
    Returns:
        np.ndarray: The cluster labels for each image.
    '''
    # Create the spectral clustering object
    sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', **SC_kwargs)

    # Perform the clustering
    cluster_labels = sc.fit_predict(affinity)

    return cluster_labels

def DSC(image_stack: np.ndarray,
        filter_shape: Tuple[int, int] = (3, 3),
        num_filters: int = 32,
        latent_dim: int = 1024,
        epochs: int = 10,
        batch_size: int = 32,
        n_clusters: int = 2,
        **SC_kwargs) -> np.ndarray:
    '''
    Applies Deep Spectral Clustering to a stack of input images.

    Args:
        image_stack (np.ndarray):
            A stack of images to cluster.
        filter_shape (Tuple[int, int]):
            The shape of the convolutional filters.
        num_filters (int):
            The number of filters to use in the convolutional layers.
        latent_dim (int):
            The dimension of the latent space.
        n_clusters (int):
            The number of clusters to create.
        **SC_kwargs:
            Keyword arguments to pass to the SpectralClustering object.
    Returns:
        np.ndarray: The cluster labels for each image.
    '''
    # Preprocess the input data for the autoencoder.
    input_data = Utilities.Math.norm(image_stack, range = (0,1))
    input_data = np.expand_dims(input_data, axis = -1)
    input_shape = input_data.shape[1:]

    # Create the autoencoder
    autoencoder, encoder, _ = Models.CAE(input_shape, filter_shape, num_filters, latent_dim)

    # Compile the autoencoder
    autoencoder.compile(optimizer = 'adam', loss = Losses.ssim_loss)

    # Fit the autoencoder
    autoencoder.fit(input_data, input_data, epochs = epochs, batch_size = batch_size, verbose = 0)

    # Get the latent vector representation
    latent_vectors = encoder.predict(input_data, verbose = 0)

    # Generate the affinity matrix from the latent vectors with RBF Kernel
    affinity_matrix = rbf_kernel(latent_vectors, gamma = 1)

    # Perform spectral clustering
    cluster_labels = spectral_cluster(affinity_matrix, n_clusters, **SC_kwargs)

    return cluster_labels

