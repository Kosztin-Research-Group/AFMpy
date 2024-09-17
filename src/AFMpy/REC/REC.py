from copy import deepcopy
import gc
import numpy as np
import cv2
import psutil
from typing import Tuple, Dict

from scipy.integrate import quad
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d

from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_samples
from sklearn.metrics.pairwise import euclidean_distances

from pystackreg import StackReg

from AFMpy.DL import Models, Losses
from AFMpy.LAFM import LAFM2D
from AFMpy import SSIM
from AFMpy import Utilities

__all__ = ['center_image', 'center_stack', 'register_image', 'register_stack', 'local_scaled_affinity',
           'local_scaled_distance','affinity_refinement', 'calculate_LFV', 'spectral_cluster', 'DSC', 
           'hierarchical_DSC', 'REC', 'IREC', 'validate_autoencoder_params', 'validate_registration_params']

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
            The method to use to register the images. Options are available in the registration_methods 
            dictionary.
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
            The method to use to register the images. Options are available in the registration_methods 
            dictionary.
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
                          k_neighbors: int = 7) -> np.ndarray:
    '''
    Calculates the locally scaled affinity matrix for a set of feature vectors.
    
    A_ij = exp(-1 / (sigma_i * sigma_j) * x_i - x_j)^2)

    See https://papers.nips.cc/paper_files/paper/2004/hash/40173ea48d9567f1f393b20c855bb40b-Abstract.html

    Args:
        features (np.ndarray):
            The feature vectors to calculate the affinity matrix for.
        k_neighbors (int):
            The number of nearest neighbors to consider for scaling thhe exponential sigma values. 
            Default is 7 (as recommended in the paper).
    Returns:
        np.ndarray: The locally scaled affinity matrix.
    '''

    # Calculate the euclidean distance matrix between the feature vectors
    distance_matrix = euclidean_distances(features)

    # Calculate the sigma matrix
    sigma_temp = np.sort(distance_matrix)[:, k_neighbors]
    sigma = np.outer(sigma_temp, sigma_temp)

    # Calculate the affinity matrix
    affinity_matrix = np.exp(-distance_matrix ** 2 / sigma)

    # Return the affinity matrix
    return affinity_matrix

def local_scaled_distance(features: np.ndarray,
                          k_neighbors: int = 7) -> np.ndarray:
    '''
    Calculates the locally scaled distance matrix for a set of feature vectors.

    D_ij = (x_i - x_j)^2 / (sigma_i * sigma_j)

    See https://papers.nips.cc/paper_files/paper/2004/hash/40173ea48d9567f1f393b20c855bb40b-Abstract.html

    Args:
        features (np.ndarray):
            The feature vectors to calculate the distance matrix for.
        k_neighbors (int):
            The number of nearest neighbors to consider for scaling the sigma values.
            Default is 7 (as recommended in the paper).
        
    Returns:
        np.ndarray: The locally scaled distance matrix.
    '''

    # Calculate the euclidean distance matrix between the feature vectors
    distance_matrix = euclidean_distances(features)

    # Calculate the sigma matrix
    sigma_temp = np.sort(distance_matrix)[:, k_neighbors]
    sigma = np.outer(sigma_temp, sigma_temp)

    # Return the locally scaled distance matrix
    return distance_matrix**2 / sigma

def affinity_refinement(features: np.ndarray,
                        cluster_labels: np.ndarray,
                        affinity_matrix: np.ndarray,
                        x_granularity: int = 100,
                        cutoff: float = 0.25) -> np.ndarray:
    '''
    Refines the cluster labels to only include the most similar features within each cluster.
    i.e. The cutoff% of features whose affinity is closest to the feature with maximized silhouette score per 
    cluster.

    Args:
        features (np.ndarray):
            The feature vectors to refine the cluster labels for.
        cluster_labels (np.ndarray):
            The cluster labels to refine.
        affinity_matrix (np.ndarray):
            The affinity matrix used to calculate the silhouette scores.
        x_granularity (int):
            The granularity of the x mesh for the CDF.
        cutoff (float):
            The cutoff value for the CDF to determine the optimal reference images.

    Returns:
        np.ndarray: The refined cluster labels.
    '''

    # Calculate the silhouette scores per image
    distances = local_scaled_distance(features)

    # Create the X mesh for the CDF
    x_vals = np.linspace(np.min(distances), np.max(distances), x_granularity)

    # Calculate the silhouette scores per image
    sil_samples = silhouette_samples(1 - affinity_matrix, cluster_labels, metric = 'precomputed')

    # Generate the Biclusters
    biclusters = np.array([np.where(cluster_labels == i, True, False) for i in range(np.max(cluster_labels) + 1)])

    # Determine the optimal reference images for each cluster by maximized silhouette score
    reference_indexes = np.argmax(sil_samples * biclusters, axis = 1)

    # Refine the cluster labels
    refined_labels = np.empty_like(biclusters)

    # Loop over each bicluster
    for index, (bicluster, reference_index) in enumerate(zip(biclusters, reference_indexes)):

        # Get the distances per cluster
        clustered_distances = distances[reference_index][bicluster]

        # Calculate the CDF and pick calculate the cutoff value
        kde = gaussian_kde(clustered_distances)
        cdf = np.array([quad(lambda t: kde(t)[0], -np.inf, x)[0] for x in x_vals])
        cdf_interp = interp1d(cdf, x_vals)
        cutoff_value = cdf_interp(cutoff)

        # Determine which labels are below the cutoff distanc
        refined_labels[index] = bicluster*(distances[reference_index] < cutoff_value)

    # Return the refined labels
    return refined_labels

# Default parameters for configuring the convolutional autoencoder.
default_autoencoder_params: Dict[str, any] = {
    'filter_shape': (3, 3),
    'num_filters': 64,
    'latent_dim': 256,
    'batch_size': 32,
    'verbose': True,
    'loss': Losses.combined_ssim_loss,
    'num_epochs': 25
}
def validate_autoencoder_params(autoencoder_params: Dict[str, any]) -> None:
    '''
    Validates the supplied autoencoder parameters. Raises an exception if any are invalid.

    Args:
        autoencoder_params (Dict[str, any]):
            The autoencoder parameters to validate.
    Returns:
        None
    '''
    # Check each autoencoder parameter against the default parameters
    for key, value in autoencoder_params.items():
        # Check if the key is a valid parameter
        if key not in default_autoencoder_params:
            logger.error(f'Invalid autoencoder parameter: {key}.')
            raise ValueError(f'Invalid autoencoder parameter: {key}.')

        # Check if the value is the correct type
        expected_type = type(default_autoencoder_params[key])
        if not isinstance(value, expected_type) and value is not None:
            logger.error(f'Invalid parameter type for {key}. Expected {expected_type}, got {type(value)}.')
            raise ValueError(f'Invalid parameter type for {key}. Expected {expected_type}, got {type(value)}.')
    
    # Log that all parameters are valid
    logger.info('All supplied autoencoder parameters are valid.')

def calculate_LFV(image_stack: np.ndarray,
                  autoencoder_params: Dict[str, any] = None) -> np.ndarray:
    '''
    Calculate the latent feature vectors for a stack of images.

    Args:
        image_stack (np.ndarray):
            The stack of images to calculate the latent feature vectors for.
        autoencoder_params (Dict[str, any]):
            The parameters to use for the autoencoder. If None, the default_autoencoder_params are used.
            See AFM.REC.default_autoencoder_params for the default parameters.
    Returns:
        np.ndarray: The latent feature vectors for the stack of images.
    '''
    # Get the autoencoder parameters
    params = deepcopy(default_autoencoder_params)
    if autoencoder_params is not None:
        params.update(autoencoder_params)

    # Validate the autoencoder parameters
    validate_autoencoder_params(params)

    # Get the autoencoder parameters
    filter_shape = params.get('filter_shape')
    num_filters = params.get('num_filters')
    latent_dim = params.get('latent_dim')
    batch_size = params.get('batch_size')
    verbose = params.get('verbose')
    loss = params.get('loss')
    num_epochs = params.get('num_epochs')

    # Preprocess the input data for the autoencoder.
    input_data = Utilities.Math.norm(image_stack, data_range = (0,1))
    input_data = np.expand_dims(input_data, axis = -1)
    input_shape = input_data.shape[1:]

    # Create the autoencoder
    autoencoder, encoder, _ = Models.CAE(input_shape, filter_shape, num_filters, latent_dim)

    # Compile the autoencoder
    autoencoder.compile(optimizer = 'adam', loss = loss)

    # Fit the autoencoder
    autoencoder.fit(input_data, input_data, epochs = num_epochs, batch_size = batch_size, verbose = verbose)

    # Get the latent vector representation
    latent_feature_vectors = encoder.predict(input_data, verbose = verbose)

    # Return the latent feature vectors
    return latent_feature_vectors

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
        k_neighbors: int = 7,
        autoencoder_params: Dict[str, any] = None,
        return_affinity: bool = False,
        **SC_kwargs) -> np.ndarray:
    '''
    Applies Deep Spectral Clustering to a stack of input images.

    Args:
        image_stack (np.ndarray):
            A stack of images to cluster.
        n_clusters (int):
            The number of clusters to create.
        return_latent_vectors (bool):
            Whether to return the latent vectors in addition to the cluster labels.
        return_affinity (bool):
            Whether to return the affinity matrix in addition to the cluster labels.
        **kwargs:
            Keyword arguments to pass to AFM.REC.spectral_cluster().
    Returns:
        np.ndarray: The cluster labels for the stack of images.
    '''
    # Calculate the latent feature vectors
    latent_vectors = calculate_LFV(image_stack, autoencoder_params)

    # Generate the affinity matrix from the latent vectors with locally scaled affinity
    affinity_matrix = local_scaled_affinity(latent_vectors, k_neighbors = k_neighbors)

    # Perform spectral clustering
    cluster_labels = spectral_cluster(affinity_matrix, n_clusters, **SC_kwargs)
    
    # Return the cluster labels and optionally the affinity matrix
    if return_affinity:
        return cluster_labels, affinity_matrix
    else:
        return cluster_labels

def hierarchical_DSC(image_stack: np.ndarray,
                     k_neighbors: int = 7,
                     max_iterations: int = 5,
                     lafm_target_resolution: Tuple[int, int] = None,
                     lafm_sigma: float = 3.0,
                     distinct_cluster_threshold: float = 0.5,
                     stability_threshold = 0.85,
                     min_cluster_size = 150,
                     autoencoder_params: Dict[str, any] = None,
                     return_traceback: bool = True) -> np.ndarray:
    '''
    Applies Hierarchical Deep Spectral Clustering to a stack of input images.

    Args:
        image_stack (np.ndarray):
            A stack of images to cluster.
        k_neighbors (int):
            The number of nearest neighbors to consider for scaling the sigma values of the Locally Scaled 
            Affinity Matrix.
        max_iterations (int):
            The maximum number of iterations to perform for the hierarchical clustering.
        lafm_target_resolution (Tuple[int, int]):
            The target resolution to use for the LAFM2D function. If None, 3x the resolution of the input images
            is used.
        lafm_sigma (float):
            The sigma value to use for the LAFM2D function. Default is 3.0 pixels.
        stability_threshold (float):
            The threshold for the stability of the clusters. If the SSIM between the LAFMs of the 2 and 3 clusters
            is above this threshold, the clusters are considered stable. Default is 0.9.
        min_cluster_ratio (float):
            The minimum ratio of images to cluster. If the number of unclustered images falls below this ratio, the
            algorithm will stop. Default is 0.15 for 15% of the total stack. 
        autoencoder_params (Dict[str, any]):
            The autoencoder parameters to use for the DSC function. If None, the default_autoencoder_params are 
            used.
            See AFM.REC.default_autoencoder_params for the default parameters.
    Returns:
        np.ndarray: The cluster labels for the stack of images.
    '''
    # Check to see if the target resolution is None. If it is, set it to 3x the resolution of the input images.
    if lafm_target_resolution is None:
        lafm_target_resolution = (image_stack.shape[1] * 3, image_stack.shape[2] * 3)

    # Get the stack size. This is useful for a few things, but will determine the minimum cluster size.
    stack_size = image_stack.shape[0]
    # Create an array of mutable indexes. This array will be used to keep track of which images have been
    # clustered and which have not between iterations.
    mutable_indexes = np.arange(stack_size)
    # Create a mutable copy of the image stack. Because each iteration acts on a subset of the stack, we need to 
    # be able to modify it.
    mutable_stack = image_stack.copy()

    # Calculate the latent feature vectors
    latent_vectors = calculate_LFV(image_stack, autoencoder_params)

    # Generate the affinity matrix from the latent vectors with locally scaled affinity
    affinity_matrix = local_scaled_affinity(latent_vectors, k_neighbors = k_neighbors)

    output_indexes = []
    traceback = {'lafms': [], 'ssims': []}
    # Perform hierarchical spectral clustering'
    for iteration in range(max_iterations):

        # If the maximum iterations is met, break the loop and put the remaining images in the output indexes.
        if iteration == max_iterations - 1:
            output_indexes.append(mutable_indexes)
            logger.info(f'Maximum iterations reached. Exiting at iteration {iteration}.')
            break

        # Perform 2-Spectral Clustering
        cluster_labels2 = spectral_cluster(affinity_matrix, 2)
        clusters2 = np.array([np.where(cluster_labels2 == i, True, False) for i in range(2)])
        # calculate the LAFM2D for each cluster.
        lafms2 = np.array([LAFM2D(mutable_stack[cluster],
                                  target_resolution = lafm_target_resolution, 
                                  sigma = lafm_sigma ) for cluster in clusters2])
        # Calculate the SSIM between the LAFMs
        ssim2 = SSIM.masked_SSIM(*lafms2, threshold_rel = 0.05)

        if ssim2 > distinct_cluster_threshold:
            output_indexes.append(mutable_indexes)
            logger.info(f'2-Clustering found highly similar clusters. Exiting at iteration {iteration}.')
            break

        # Perform the 3-Spectral Clustering
        cluster_labels3 = spectral_cluster(affinity_matrix, 3)
        clusters3 = np.array([np.where(cluster_labels3 == i, True, False) for i in range(3)])
        # Calculate the LAFM2D for each cluster
        lafms3 = np.array([LAFM2D(mutable_stack[cluster],
                                  target_resolution = lafm_target_resolution,
                                  sigma = lafm_sigma) for cluster in clusters3])
        # Calculate the SSIM between the LAFMs
        pairs = [(0,1), (0,2), (1,2)]
        ssims3 = np.array([SSIM.masked_SSIM(lafms3[pair[0]],
                                      lafms3[pair[1]],
                                      threshold_rel = 0.05) for pair in pairs])

        # Calculate the stability of each 2-cluster by the SSIM of the LAFMs
        stability_ssims = np.array([[SSIM.masked_SSIM(lafm2, 
                                                 lafm3, 
                                                 threshold_rel = 0.05) for lafm3 in lafms3] for lafm2 in lafms2])

        traceback['lafms'].append((lafms2, lafms3))
        traceback['ssims'].append((ssim2, ssims3, stability_ssims))
        
        # Determine which clusters are stable
        which_stable = np.logical_or(*(stability_ssims > stability_threshold))

        # If none of the clusters are stable, break the loop
        if not np.any(which_stable):
            output_indexes.append(mutable_indexes)
            logger.info(f'No stable clusters found. Exiting at iteration {iteration}.')
            break

        # If all the clusters are stable, break the loop
        if np.all(which_stable):
            for cluster in clusters2:
                output_indexes.append(mutable_indexes[cluster])
            logger.info(f'All clusters are stable. Exiting at iteration {iteration}.')
            break

        # Extract the stable clusters
        stable_clusters = clusters3[which_stable]

        # Append the stable clusters to the output indexes and create the compliment. The compliment are the
        # images remaining unstably clustered images.
        compliment = []

        # Remove any clusters that are too small
        too_small = np.sum(stable_clusters, axis = 1) < min_cluster_size
        stable_clusters = stable_clusters[np.logical_not(too_small)]
        
        if len(stable_clusters) == 0:
            output_indexes.append(mutable_indexes)
            logger.info(f'All stable clusters are too small. Exiting at iteration {iteration}.')
            break

        for stable_cluster in stable_clusters:
            stable_indexes = mutable_indexes[stable_cluster]
            output_indexes.append(stable_indexes)

            compliment.append(np.logical_not(stable_cluster))
        remaining_unclustered = np.all(compliment, axis = 0)
        
        # If the remaining unclustered images are less than the minimum cluster ratio, break the loop and put the
        # remaining images in the output indexes.
        if np.sum(remaining_unclustered) < min_cluster_size:
            output_indexes.append(mutable_indexes[remaining_unclustered])
            logger.info(f'Minimum cluster size reached. Exiting at iteration {iteration}.')
            break

        # Update the affinity matrix, mutable indexes, and mutable stack with the remaining unclustered images.
        affinity_matrix = affinity_matrix[remaining_unclustered][:, remaining_unclustered]
        mutable_indexes = mutable_indexes[remaining_unclustered]
        mutable_stack = image_stack[mutable_indexes]

    # Return the output indexes and traceback if requested.
    if return_traceback:
        return output_indexes, traceback
    else:
        return output_indexes

# Default parameters for the centering and registration functions
default_registration_params: Dict[str, any] = {
    'center_method': 'cog',
    'centering_threshold': 0.0,
    'registration_method': 'rigid'
}

def validate_registration_params(registration_params: Dict[str, any]) -> None:
    '''
    Validates the supplied registration parameters. Raises an exception if any are invalid.

    Args:
        registration_params (Dict[str, any]):
            The registration parameters to validate.
    Returns:
        None
    '''
    # Check each registration parameter against the default parameters
    for key, value in registration_params.items():
        # Check if the key is a valid parameter
        if key not in registration_params:
            logger.error(f'Invalid registration parameter: {key}.')
            raise ValueError(f'Invalid registration parameter: {key} is not a valid parameter.')

        # Check if the value is the correct type
        expected_type = type(registration_params[key])
        if not isinstance(value, expected_type) and value is not None:
            logger.error(f'Invalid parameter type for {key}. Expected {expected_type}, got {type(value)}.')
            raise ValueError(f'Invalid parameter type for {key}. Expected {expected_type}, got {type(value)}.')

    # Log that all parameters are valid
    logger.info('All supplied registration parameters are valid.')

def REC(image_stack: np.ndarray,
        reference_index: int = 0,
        n_clusters: int = 2,
        k_neighbors: int = 7,
        registration_params: Dict[str, any] = None,
        autoencoder_params: Dict[str, any] = None,
        return_registered_stack: bool = False,
        return_refined_references: bool = False,
        **SC_kwargs) -> np.ndarray:
    '''
    Applies the Registration and Clutsering (REC) algorithm to a stack of images.
    
    Args:
        image_stack (np.ndarray):
            The stack of images to process.
        reference_index (int):
            The index of the reference image to use for registration.
        n_clusters (int):
            The number of clusters to create.
        k_neighbors (int):
            The number of nearest neighbors to consider for scaling the sigma values of the Locally Scaled 
            Affinity Matrix.
        registration_params (Dict[str, any]):
            The registration parameters to use. If None, the default_registration_params are used.
            See AFM.REC.default_registration_params for the default parameters.
        autoencoder_params (Dict[str, any]):
            The autoencoder parameters to use for the DSC function. If None, the default_autoencoder_params are used.
            See AFM.REC.default_autoencoder_params for the default parameters.
        return_registered_stack (bool):
            Whether to return the registered stack in addition to the cluster labels.
        return_refined_references (bool):
            Whether to return the Silhouette Score refined reference indexes in addition to the cluster labels.
        **SC_kwargs:
            Spectral Clustering Keyword arguments to be passed to the DSC function.

    Returns:
        list: A list containing the cluster labels, and optionally the registered stack and refined reference indexes.
    '''
    # Get the registration parameters
    params = deepcopy(default_registration_params)
    if registration_params is not None:
        params.update(registration_params)
    
    # Validate the registration parameters
    validate_registration_params(params)

    # Get the registration parameters
    center_method = params.get('center_method')
    centering_threshold = params.get('centering_threshold')
    registration_method = params.get('registration_method')

    # Center the stack
    centered_stack = center_stack(image_stack, method=center_method, threshold=centering_threshold)

    # Register the stack
    registered_stack = register_stack(centered_stack[reference_index], centered_stack, method=registration_method)

    output = []

    # Perform Deep Spectral Clustering
    cluster_labels, affinity_matrix = DSC(registered_stack, n_clusters=n_clusters, k_neighbors=k_neighbors, autoencoder_params=autoencoder_params, return_affinity=True, **SC_kwargs)
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
         k_neighbors: int = 7,
         max_iterations: int = 10,
         registration_params: Dict[str, any] = None,
         autoencoder_params: Dict[str, any] = None,
         return_labels: bool = False,
         **SC_kwargs):
    '''
    Applies Iterative Registration and Clustering (IREC) algorithm to a stack of images.
    
    Args:
        image_stack (np.ndarray):
            The stack of images to process.
        reference_index (int):
            The index of the initial reference image to use for registration.
        n_clusters (int):
            The number of refined clusters to create.
        k_neighbors (int):
            The number of nearest neighbors to consider for scaling the sigma values of the Locally Scaled 
            Affinity Matrix.
        max_iterations (int):
            The maximum number of iterations to perform.
        registration_params (Dict[str, any]):
            The registration parameters to use. If None, the default_registration_params are used.
            See AFM.REC.default_registration_params for the default parameters.
        autoencoder_params (Dict[str, any]):
            The autoencoder parameters to use for the DSC function. If None, the default_autoencoder_params are used.
            See AFM.REC.default_autoencoder_params for the default parameters.
        return_labels (bool):
            Whether to return the cluster labels in addition to the registered stack.
        **SC_kwargs:
            Spectral Clustering Keyword arguments to be passed to the DSC function.

    Returns:
        list: A list containing n registered clustered stacks of images.
    '''

    # Perform the initial REC to get refined indexes
    logger.info(f'Performing initial REC with reference index {reference_index}')
    _, old_reference_indexes = REC(image_stack, reference_index, n_clusters, k_neighbors=k_neighbors,
                                   registration_params=registration_params, autoencoder_params=autoencoder_params,
                                   return_refined_references = True, **SC_kwargs)
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
            cluster_labels, new_reference_indexes = REC(image_stack, reference_index, n_clusters,
                                                        k_neighbors=k_neighbors,
                                                        registration_params=registration_params,
                                                        autoencoder_params=autoencoder_params,
                                                        return_refined_references = True, **SC_kwargs)
            
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
    Utilities.clear_variables_tensorflow(old_reference_indexes, converged, iteration_indexes, cluster_labels, 
                                         bicluster_labels, which_cluster, refined_index, matches, units = 'mb')
    
    # With the refined indexes, perform the final REC
    logger.info('Performing final REC with refined reference indexes.')
    registered_clustered_stacks = []
    registered_clustered_labels = []

    # Looping over each refined reference index
    for reference_index in past_indexes[-1]:
        logger.info(f'Performing REC with reference index {reference_index}')
        # Applying Registration and Clustering
        cluster_labels, registered_stack = REC(image_stack, reference_index, n_clusters,
                                                        k_neighbors=k_neighbors,
                                                        registration_params=registration_params,
                                                        autoencoder_params=autoencoder_params,
                                                        return_registered_stack = True, **SC_kwargs)

        # Determining which cluster the reference image belongs to
        bicluster_labels = np.array([np.where(cluster_labels == i,True,False) for i in range(n_clusters)])
        which_cluster = bicluster_labels[:,reference_index]

        REC_labels = bicluster_labels[which_cluster][0]

        # Appending the cluster labels
        registered_clustered_stacks.append(registered_stack[REC_labels])
        registered_clustered_labels.append(REC_labels)

    # Return the registered clustered stacks and optionally the cluster labels
    if return_labels:
        return registered_clustered_stacks, registered_clustered_labels
    else:
        return registered_clustered_stacks