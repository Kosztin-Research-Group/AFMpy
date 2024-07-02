import gc
import numpy as np
import cv2
import psutil
from typing import Tuple, Dict
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_samples
from sklearn.metrics.pairwise import euclidean_distances
from tensorflow.python.keras import backend as K

from pystackreg import StackReg

from AFMpy.DL import Models, Losses
from AFMpy import Utilities

__all__ = ['center_image', 'center_stack', 'register_image', 'register_stack', 'local_scaled_affinity', 'spectral_cluster',
           'DSC', 'REC', 'IREC']

logger = Utilities.Logging.make_module_logger(__name__)

################################
##### Registration Methods #####
################################

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

    # Return the center of geometry
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

    # Return the center of mass
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
            The (dx,dy) translation to apply to the image.
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

    # Return the translated image
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
    center_method = centering_methods[method]

    # Get the center of the image
    x_center, y_center = center_method(image, threshold)

    # Calculate the translation
    dx = image.shape[1] / 2 - x_center
    dy = image.shape[0] / 2 - y_center

    # Translate the image
    centered_image = _translate_image(image, (dx, dy))

    # Return the centered image
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

    # Return the centered stack
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

    # Return the registered image
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

    # Return the registered stack
    return registered_stack

##############################
##### Clustering Methods #####
##############################

def local_scaled_affinity(features: np.ndarray,
                          k: int = 7) -> np.ndarray:
    '''
    Calculates the locally scaled affinity matrix for a set of feature vectors.
    
    A_ij = exp(-1 / (sigma_i * sigma_j) * x_i - x_j)^2)

    See https://papers.nips.cc/paper_files/paper/2004/hash/40173ea48d9567f1f393b20c855bb40b-Abstract.html

    Args:
        features (np.ndarray):
            The feature vectors to calculate the affinity matrix for.
        k (int):
            The number of nearest neighbors to consider for scaling thhe exponential sigma values. 
            Default is 7 (as recommended in the paper).
    Returns:
        np.ndarray: The locally scaled affinity matrix.
    '''

    # Calculate the distance matrix between the feature vectors
    distance_matrix = euclidean_distances(features)

    # Calculate the sigma matrix
    sigma_temp = np.sort(distance_matrix)[:, k]
    sigma = np.outer(sigma_temp, sigma_temp)

    # Calculate the affinity matrix
    affinity_matrix = np.exp(-distance_matrix ** 2 / sigma)

    # Return the affinity matrix
    return affinity_matrix

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
        np.ndarray: The cluster labels for the stack of images.
    '''
    # Create the spectral clustering object
    sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', **SC_kwargs)

    # Perform the clustering
    cluster_labels = sc.fit_predict(affinity)

    # Return the cluster labels
    return cluster_labels

def DSC(image_stack: np.ndarray,
        n_clusters: int = 2,
        return_affinity: bool = False,
        **kwargs) -> np.ndarray:
    '''
    Applies Deep Spectral Clustering to a stack of input images.

    Args:
        image_stack (np.ndarray):
            A stack of images to cluster.
        n_clusters (int):
            The number of clusters to create.
        return_affinity (bool):
            Whether to return the affinity matrix in addition to the cluster labels.
        **kwargs:
            Keyword arguments to pass AFM.DL.CAE() and AFM.REC.spectral_cluster().
    Returns:
        np.ndarray: The cluster labels for the stack of images.
    '''
    # Pop the autoencoder-specific arguments from kwargs
    filter_shape = kwargs.pop('filter_shape', (3, 3))
    num_filters = kwargs.pop('num_filters', 64)
    latent_dim = kwargs.pop('latent_dim', 512)
    epochs = kwargs.pop('epochs', 25)
    batch_size = kwargs.pop('batch_size', 32)
    verbose = kwargs.pop('verbose', 1)

    # Remaining kwargs are for spectral_cluster
    SC_kwargs = kwargs

    # Preprocess the input data for the autoencoder.
    input_data = Utilities.Math.norm(image_stack, range = (0,1))
    input_data = np.expand_dims(input_data, axis = -1)
    input_shape = input_data.shape[1:]

    # Create the autoencoder
    autoencoder, encoder, _ = Models.CAE(input_shape, filter_shape, num_filters, latent_dim)

    # Compile the autoencoder
    autoencoder.compile(optimizer = 'adam', loss = Losses.ssim_loss)

    # Fit the autoencoder
    autoencoder.fit(input_data, input_data, epochs = epochs, batch_size = batch_size, verbose = verbose)

    # Get the latent vector representation
    latent_vectors = encoder.predict(input_data, verbose = verbose)

    # Generate the affinity matrix from the latent vectors with RBF Kernel
    affinity_matrix = local_scaled_affinity(latent_vectors)

    # Perform spectral clustering
    cluster_labels = spectral_cluster(affinity_matrix, n_clusters, **SC_kwargs)

    # Cleaning up some memory
    # Utilities.clear_variables_tensorflow(autoencoder, encoder, latent_vectors, input_data, input_shape, units = 'mb')

    # Return the cluster labels and affinity matrix if requested
    if return_affinity:
        return cluster_labels, affinity_matrix
    else:
        return cluster_labels

def REC(image_stack: np.ndarray,
        reference_index: int = 0,
        n_clusters: int = 2,
        return_registered_stack: bool = False,
        return_refined_references: bool = False,
        **kwargs) -> np.ndarray:
    '''
    Applies the Registration and Clutsering (REC) algorithm to a stack of images.
    
    Args:
        image_stack (np.ndarray):
            The stack of images to process.
        reference_index (int):
            The index of the reference image to use for registration.
        n_clusters (int):
            The number of clusters to create.
        return_registered_stack (bool):
            Whether to return the registered stack in addition to the cluster labels.
        return_refined_references (bool):
            Whether to return the Silhouette Score refined reference indexes in addition to the cluster labels.
        **kwargs:
            Keyword arguments. Passed to centering, registration, and DSC functions.

    Returns:
        list: A list containing the cluster labels, and optionally the registered stack and refined reference indexes.
    '''

    # Pop the method-specific arguments from kwargs
    # Handling kwargs for centering images
    center_method = kwargs.pop('center_method', 'cog')
    centering_threshold = kwargs.pop('centering_threshold', 0.0)

    # Handling kwargs for registering images
    registration_method = kwargs.pop('registration_method', 'rigid')

    # Remaining kwargs are passed to DSC
    DSC_kwargs = kwargs

    # Center the stack
    centered_stack = center_stack(image_stack, method=center_method, threshold=centering_threshold)

    # Register the stack
    registered_stack = register_stack(centered_stack[reference_index], centered_stack, method=registration_method)

    output = []

    # Perform Deep Spectral Clustering
    cluster_labels, affinity_matrix = DSC(registered_stack, n_clusters=n_clusters, return_affinity = True, **DSC_kwargs)
    # Put the cluster labels into a boolean array
    bicluster_labels = np.array([np.where(cluster_labels == i,True,False) for i in range(n_clusters)])
    
    # Append the cluster labels to the output
    output.append(cluster_labels)

    # Add the registered stack if requested
    if return_registered_stack:
        # Append the registered stack to the output
        output.append(registered_stack)

    # Add the refined references if requested
    if return_refined_references:
        # Calculate the silhouette scores per image.
        sil_samples = silhouette_samples(1 - affinity_matrix, cluster_labels, metric = 'precomputed')

        # Determine the optimal reference images for each cluster by maximized silhouette score
        refined_reference_indexes = np.argmax(sil_samples * bicluster_labels, axis = 1)

        output.append(refined_reference_indexes)

    # Return the cluster labels
    return output

def IREC(image_stack: np.ndarray,
         reference_index: int = 0,
         n_clusters: int = 2,
         max_iterations: int = 10,
         return_labels: bool = False,
         **kwargs):
    '''
    Applies Iterative Registration and Clustering (IREC) algorithm to a stack of images.
    
    Args:
        image_stack (np.ndarray):
            The stack of images to process.
        reference_index (int):
            The index of the initial reference image to use for registration.
        n_clusters (int):
            The number of refined clusters to create.
        max_iterations (int):
            The maximum number of iterations to perform.
        return_labels (bool):
            Whether to return the cluster labels in addition to the registered stack.
        **kwargs:
            Keyword arguments. Passed to centering, registration, and DSC functions.

    Returns:
        list: A list containing n registered clustered stacks of images.
    '''

    # Perform the initial REC to get refined indexes
    logger.info(f'Performing initial REC with reference index {reference_index}')
    _, old_reference_indexes = REC(image_stack, reference_index, n_clusters, return_refined_references = True, **kwargs)
    logger.info(f'Initial REC complete. Initial cluster reference indexes: {old_reference_indexes}')

    # Initialize the past indexes array and convergence flags
    past_indexes = [old_reference_indexes]
    converged = np.full(n_clusters, False)

    # Begin the iterative process
    for iteration in range(max_iterations):

        logger.info(f'Beginning iteration {iteration + 1}.')

        # Initialize the iteration indexes
        iteration_indexes = []

        # Loop over each rerference index
        for k, reference_index in enumerate(old_reference_indexes):
            
            # Check if the cluster has converged
            if converged[k]:

                # If the cluster has converged, skip it
                logger.info(f'Cluster {k} has converged. Skipping.')
                iteration_indexes.append(reference_index)       
                continue

            # Call REC with the reference
            logger.info(f'Performing REC with reference index {reference_index}')
            cluster_labels, new_reference_indexes = REC(image_stack, reference_index, n_clusters, return_refined_references = True, **kwargs)
            bicluster_labels = np.array([np.where(cluster_labels == i,True,False) for i in range(n_clusters)])

            # Determine which cluster the reference image belongs to
            which_cluster = bicluster_labels[:,reference_index]

            # Get the refined inddex from that cluster
            refined_index = new_reference_indexes[which_cluster][0]

            # Add the new reference index to the iteration indexes
            logger.info(f'Iteration: {iteration}. Cluster {k} refined index: {refined_index}')
            iteration_indexes.append(refined_index)

        # Check for matches in the past indexes
        logger.info(f'Iteration indexes: {iteration_indexes}')
        matches = np.any(np.array(iteration_indexes) == np.array(past_indexes), axis = 0)
        
        converged = np.logical_or(converged, matches)
        logger.info(f'Converged clusters: {converged}')

        # Update the past indexes and the references used for the previous iteration step
        past_indexes.append(iteration_indexes)
        old_reference_indexes = iteration_indexes

        # Check if all clusters have converged, break the loop
        if np.all(converged):
            logger.info('All clusters have converged. Exiting iterative process.')
            break

        if iteration == max_iterations - 1:
            logger.warning('Maximum iterations reached. Exiting iterative process.')

    # Clear some memory
    Utilities.clear_variables_tensorflow(old_reference_indexes, converged, iteration_indexes, cluster_labels, bicluster_labels, which_cluster, refined_index, matches, units = 'mb')
    
    # With the refined indexes, perform the final REC
    logger.info('Performing final REC with refined reference indexes.')
    registered_clustered_stacks = []
    registered_clustered_labels = []

    # Looping over each refined reference index
    for reference_index in past_indexes[-1]:
        logger.info(f'Performing REC with reference index {reference_index}')
        # Applying Registration and Clustering
        cluster_labels, registered_stack = REC(image_stack, reference_index, n_clusters, return_registered_stack = True, **kwargs)

        # Determining which cluster the reference image belongs to
        bicluster_labels = np.array([np.where(cluster_labels == i,True,False) for i in range(n_clusters)])
        which_cluster = bicluster_labels[:,reference_index]

        REC_labels = bicluster_labels[which_cluster][0]

        # Appending the cluster labels
        registered_clustered_stacks.append(registered_stack[REC_labels])
        registered_clustered_labels.append(REC_labels)

    # Prepare the output
    output = []
    # Add the registered clustered stacks to the output
    output.append(registered_clustered_stacks)

    # Add the cluster labels to the output if requested
    if return_labels:
        output.append(registered_clustered_labels)

    # Return the final cluster labels
    logger.info('Final IREC complete. Returning cluster labels.')
    return output
