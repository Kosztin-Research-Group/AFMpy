from copy import deepcopy
import logging
import numpy as np
import cv2
from typing import Dict, List, Tuple

from scipy.integrate import quad
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d

from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_samples
from sklearn.metrics.pairwise import euclidean_distances

from pystackreg import StackReg

from AFMpy.DL import Models, Losses
from AFMpy import SSIM, Stack, Utilities

__all__ = ['center_image', 'center_image_stack', 'register_image', 'register_image_stack', 'DSC', 'REC', 'IREC',
           'hierarchical_DSC']

logger = logging.getLogger(__name__)

##########################################
##### Centering/Registration Methods #####
##########################################
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

CENTERING_METHODS: Dict[str, callable] = {
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
    # Check if the centering method is valid.
    if method not in CENTERING_METHODS:
        logger.error(f'Invalid method: {method}. Valid methods are: {list(CENTERING_METHODS.keys())}')
        raise ValueError(f'Invalid method: {method}. Valid methods are: {list(CENTERING_METHODS.keys())}')
    
    # Dispatch the centering method
    center_method = CENTERING_METHODS[method]

    # Get the center of the image
    x_center, y_center = center_method(image, threshold)

    # Calculate the translation
    dx = image.shape[1] / 2 - x_center
    dy = image.shape[0] / 2 - y_center

    # Translate the image
    centered_image = _translate_image(image, (dx, dy))

    # Return the centered image
    return centered_image

def center_image_stack(stack: np.ndarray,
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
    # Create an empty stack to hold the centered images
    centered_stack = np.empty_like(stack)

    # Center each image in the stack
    for i, image in enumerate(stack):
        centered_stack[i] = center_image(image, method=method, threshold=threshold)

    # Return the centered stack
    return centered_stack

def register_image(ref: np.ndarray,
                   mov: np.ndarray,
                   sr: StackReg) -> np.ndarray:
    '''
    Registers a moving image to a reference image using a given method. 
    Also used as a helper function for register_stack.

    Args:
        ref (np.ndarray):
            The reference image.
        mov (np.ndarray):
            The moving image.
        sr (StackReg):
            The pystackreg StackReg object to use for registration.
    Returns:
        np.ndarray: The registered image.
    '''
    # Register the images
    registered_image = sr.register_transform(ref, mov)

    # Some pixel values may be negative, or very close to zero. This can cause issues elsewhere,
    # so we set them to zero with a mask.
    mask = np.logical_not(np.isclose(registered_image, 0, atol=1e-5)) * (registered_image > 0)
    registered_image = registered_image * mask

    # Return the registered image
    return registered_image

def register_image_stack(ref: np.ndarray,
                         image_stack: np.ndarray,
                         sr: StackReg) -> np.ndarray:
    '''
    Registers a stack of images to a reference image.

    Args:
        ref (np.ndarray):
            The reference image.
        stack (np.ndarray):
            The stack of images to register. Each image in the stack acts as the moving image.
        sr (StackReg):
            The StackReg object to use for registration.
    Returns:
        np.ndarray: The registered stack.
    '''
    # Create an empty stack to hold the registered images
    registered_stack = np.empty_like(image_stack)

    # Register each image in the stack
    for i, image in enumerate(image_stack):
        registered_stack[i] = register_image(ref, image, sr = sr)

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
        np.ndarray: The locally scale        autoencoder_params: Dict[str, any] = None, distance matrix.
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

def calculate_LFV(image_stack: np.ndarray,
                  cae: Models.ConvolutionalAutoencoder) -> np.ndarray:
    '''
    Calculate the latent feature vectors for a stack of images.

    Args:
        image_stack (np.ndarray):
            The stack of images to calculate the latent feature vectors for. 
            Must be shape (num_images, height, width, channels).
        cae (Models.ConvolutionalAutoencoder):
            The convolutional autoencoder to use for calculating the latent feature vectors.
    Returns:
        np.ndarray: The latent feature vectors for the stack of images.
    '''
    # Compile the autoencoder
    cae.compile(optimizer = 'adam', loss = Losses.combined_ssim_loss)

    # Fit the autoencoder
    cae.fit(image_stack)

    # Get the latent vector representation
    lfvs = cae.encode(image_stack)

    # Return the latent feature vectors
    return lfvs

def spectral_cluster(affinity: np.ndarray,
                     n_clusters: int = 2,
                     **kwargs) -> np.ndarray:
    '''
    Performs spectral clustering on the given affinity matrix.

    Args:
        affinity (np.ndarray):
            The affinity matrix to use for clustering.
        n_clusters (int):
            The number of clusters to create.
        **kwargs:
            Keyword arguments to pass to the SpectralClustering object.
    Returns:
        np.ndarray: The cluster labels for the stack of images.
    '''
    # Create the spectral clustering object
    sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', **kwargs)

    # Perform the clustering
    cluster_labels = sc.fit_predict(affinity)

    # Return the cluster labels
    return cluster_labels

def DSC(input_stack: Stack.Stack,
        cae: Models.ConvolutionalAutoencoder,
        n_clusters: int = 2,
        k_neighbors: int = 7,
        **kwargs) -> List[Stack.Stack]:
    '''
    Applies Deep Spectral Clustering to an input Stack.

    Args:
        stack (Stack.Stack):
            The AFMpy Stack object containing the images to cluster.
        cae (Models.ConvolutionalAutoencoder):
            The convolutional autoencoder to use for calculating the latent feature vectors.
        n_clusters (int):
            The number of clusters to create. Default is 2.
        k_neighbors (int):
            The number of nearest neighbors to consider for scaling the sigma values of the Locally Scaled 
            Affinity Matrix. Default is 7.
        **kwargs:
            Keyword arguments. Passed to spectral_cluster function.
    Returns:
        List[Stack.Stack]: A list of AFMpy Stack objects. Each stack is a conformational cluster of the input stack.
    '''
    # Extract the images from the stack
    images = input_stack.stack

    # Expand the dimensions of the images so they have the shape (num_images, height, width, channels)
    images = np.expand_dims(images, axis = -1)

    # Normalize the images.
    images = Utilities.norm(images)

    # Calculate the latent feature vectors
    latent_vectors = calculate_LFV(images, cae)

    # Generate the affinity matrix from the latent vectors with locally scaled affinity
    affinity_matrix = local_scaled_affinity(latent_vectors, k_neighbors = k_neighbors)

    # Perform spectral clustering
    cluster_labels = spectral_cluster(affinity_matrix, n_clusters, **kwargs)
    
    # Convert the cluster labels to a list of boolean arrays: one for each cluster
    bicluster_labels = np.array([np.where(cluster_labels == i, True, False) for i in range(n_clusters)])

    # Create stack objects for each cluster
    clustered_stacks = [Stack.Stack(input_stack.stack[bicluster],
                                    resolution = input_stack.resolution,
                                    indexes = input_stack.indexes[bicluster]) for bicluster in bicluster_labels]
    
    # Return the clustered stacks
    logger.debug(f'Returning {n_clusters} DSC clustered stacks.')
    return clustered_stacks

def REC(input_stack: Stack.Stack,
        cae: Models.ConvolutionalAutoencoder,
        sr: StackReg = StackReg(StackReg.RIGID_BODY),
        reference_index: int = 0,
        n_clusters: int = 2,
        k_neighbors: int = 7,
        **kwargs) -> List[Stack.Stack]:
    '''
    Applies Registration and Clustering to an input Stack.
    
    Args:
        input_stack (Stack.Stack):
            The AFMpy Stack object containing the images to cluster.
        cae (Models.ConvolutionalAutoencoder):
            The convolutional autoencoder to use for calculating the latent feature vectors.
        sr (StackReg):
            The StackReg object to use for registration. Default is StackReg(StackReg.RIGID_BODY).
        reference_index (int):
            The index of the reference image to use for registration. Default is 0.
        n_clusters (int):
            The number of clusters to create. Default is 2.
        k_neighbors (int):
            The number of nearest neighbors to consider for scaling the sigma values of the Locally Scaled 
            Affinity Matrix. Default is 7.
        **kwargs:
            Keyword arguments. Passed to spectral_cluster function.
    '''
    # Extract the images from the stack
    input_images = input_stack.stack
    # Center the stack of images.
    centered_images = center_image_stack(input_images, method = 'cog')
    # Register the image stack according to the reference index
    registered_images = register_image_stack(centered_images[reference_index], centered_images, sr)

    # Expand the dimensions of the images so they have the shape (num_images, height, width, channels)
    registered_images = np.expand_dims(registered_images, axis = -1)

    # Normalize the images.
    registered_images_norm = Utilities.norm(registered_images)

    # Calculate the latent feature vectors
    latent_vectors = calculate_LFV(registered_images_norm, cae)

    # Generate the affinity matrix from the latent vectors with locally scaled affinity
    affinity_matrix = local_scaled_affinity(latent_vectors, k_neighbors = k_neighbors)

    # Perform spectral clustering
    cluster_labels = spectral_cluster(affinity_matrix, n_clusters, **kwargs)

    # Convert the cluster labels to a list of boolean arrays: one for each cluster
    bicluster_labels = np.array([np.where(cluster_labels == i, True, False) for i in range(n_clusters)])

    # Create stack objects for each cluster
    clustered_stacks = [Stack.Stack(registered_images[bicluster],
                                    resolution = input_stack.resolution,
                                    indexes = input_stack.indexes[bicluster]) for bicluster in bicluster_labels]
    
    # Return the registered clusters
    logger.debug(f'Returning {n_clusters} REC clustered stacks.')
    return clustered_stacks

def IREC(input_stack: Stack.Stack,
         cae: Models.ConvolutionalAutoencoder,
         sr: StackReg = StackReg(StackReg.RIGID_BODY),
         reference_index: int = 0,
         n_clusters: int = 2,
         k_neighbors: int = 7,
         max_iterations: int = 10,
         **kwargs) -> List[Stack.Stack]:
    '''
    Applies Iterative Registration and Clustering to an input Stack.

    Args:
        input_stack (Stack.Stack):
            The AFMpy Stack object containing the images to cluster.
        cae (Models.ConvolutionalAutoencoder):
            The convolutional autoencoder to use for calculating the latent feature vectors.
        sr (StackReg):
            The StackReg object to use for registration. Default is StackReg(StackReg.RIGID_BODY).
        reference_index (int):
            The index of the reference image to use for registration. Default is 0.
        n_clusters (int):
            The number of clusters to create. Default is 2.
        k_neighbors (int):
            The number of nearest neighbors to consider for scaling the sigma values of the Locally Scaled 
            Affinity Matrix. Default is 7.
        max_iterations (int):
            The maximum number of iterations to perform for the iterative clustering. Default is 10.
        **kwargs:
            Keyword arguments. Passed to spectral_cluster function.
    '''
    # Inital Registration and Clustering
    # Extract the images from the stack
    input_images = input_stack.stack
    # Center the stack of images.
    centered_images = center_image_stack(input_images, method = 'cog')
    # Register the image stack according to the reference index
    registered_images = register_image_stack(centered_images[reference_index], centered_images, sr)
    # Expand the dimensions of the images so they have the shape (num_images, height, width, channels) and normalize
    registered_images_norm = Utilities.norm(np.expand_dims(registered_images, axis = -1))
    # Calculate the latent feature vectors
    latent_vectors = calculate_LFV(registered_images_norm, cae)
    # Generate the affinity matrix from the latent vectors with locally scaled affinity
    affinity_matrix = local_scaled_affinity(latent_vectors, k_neighbors = k_neighbors)
    # Perform spectral clustering
    cluster_labels = spectral_cluster(affinity_matrix, n_clusters, **kwargs)
    # Convert the cluster labels to a list of boolean arrays: one for each cluster
    bicluster_labels = np.array([np.where(cluster_labels == i, True, False) for i in range(n_clusters)])
    # Get the silhouette samples for each cluster.
    sil_samples = silhouette_samples(1 - affinity_matrix, cluster_labels, metric = 'precomputed')
    # Determine the optimal reference images for each cluster by maximized silhouette score
    old_reference_indexes = np.argmax(sil_samples * bicluster_labels, axis = 1)
    # Save the old reference indexes for traceback
    past_indexes = [old_reference_indexes]
    # Initialize the lst of converged clusters
    converged = np.full(n_clusters, False)

    logger.info('Initial REC complete.')
    logger.info(f'Initial Reference Indexes: {old_reference_indexes}')
    # Begin the iterative process
    for iteration in range(max_iterations):
        logger.info(f'Beginning iteration {iteration + 1} of IREC.')

        # Initialize the list of references for this iteration
        iteration_indexes = []

        # Register and cluster with respect to each reference index.
        for k, reference_index in enumerate(old_reference_indexes):
            
            # If the cluster has converged, skip it
            if converged[k]:
                logger.info(f'Cluster {k} has converged. Skipping.')
                iteration_indexes.append(reference_index)
                continue
            
            logger.debug(f'Applying REC to cluster {k} with reference index {reference_index}.')
            # Register the image stack according to the reference index
            registered_images = register_image_stack(centered_images[reference_index], centered_images, sr)
            # Expand the dimensions of the images so they have the shape (num_images, height, width, channels) and normalize
            registered_images_norm = Utilities.norm(np.expand_dims(registered_images, axis = -1))
            # Reset the state of the CAE because the weights are fit to the previous iteration.
            cae.rebuild()
            # Calculate the latent feature vectors
            latent_vectors = calculate_LFV(registered_images_norm, cae)
            # Generate the affinity matrix from the latent vectors with locally scaled affinity
            affinity_matrix = local_scaled_affinity(latent_vectors, k_neighbors = k_neighbors)
            # Perform spectral clustering
            cluster_labels = spectral_cluster(affinity_matrix, n_clusters, **kwargs)
            # Convert the cluster labels to a list of boolean arrays: one for each cluster
            bicluster_labels = np.array([np.where(cluster_labels == i, True, False) for i in range(n_clusters)])
            # Get the silhouette samples for each cluster.
            sil_samples = silhouette_samples(1 - affinity_matrix, cluster_labels, metric = 'precomputed')
            # Determine which cluster contains the reference index.
            which_cluster = bicluster_labels[:, reference_index]

            # Determine the optimal reference images for each cluster by maximized silhouette score
            refined_references = np.argmax(sil_samples * bicluster_labels, axis = 1)
            # Determine which refined reference belongs to this cluster
            new_reference_index = refined_references[which_cluster][0]

            # Add the new reference index to the iteration indexes
            iteration_indexes.append(new_reference_index)

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
            logger.info('All clusters have converged.')
            break

        # Check if the maximum iterations is met, break the loop
        if iteration == max_iterations - 1:
            logger.warning('Maximum iterations reached. Clusters may not be optimally converged.')
            break
        
    logger.info('Executing final registration and clustering.')
    # Register and cluster with respect to the converged indexes
    converged_stacks = []
    for reference_index in old_reference_indexes:
        # Register the image stack according to the reference index
        registered_images = register_image_stack(centered_images[reference_index], centered_images, sr)
        # Expand the dimensions of the images so they have the shape (num_images, height, width, channels) and normalize
        registered_images_norm = Utilities.norm(np.expand_dims(registered_images, axis = -1))
        # Reset the state of the CAE because the weights are fit to the previous iteration.
        cae.rebuild()
        # Calculate the latent feature vectors
        latent_vectors = calculate_LFV(registered_images_norm, cae)
        # Generate the affinity matrix from the latent vectors with locally scaled affinity
        affinity_matrix = local_scaled_affinity(latent_vectors, k_neighbors = k_neighbors)
        # Perform spectral clustering
        cluster_labels = spectral_cluster(affinity_matrix, n_clusters, **kwargs)
        # Convert the cluster labels to a list of boolean arrays: one for each cluster
        bicluster_labels = np.array([np.where(cluster_labels == i, True, False) for i in range(n_clusters)])
        # Determine which cluster the registration reference is in
        which_cluster = bicluster_labels[:, reference_index]

        # Create a stack object for the converged cluster
        converged_stack = Stack.Stack(registered_images[bicluster_labels[which_cluster][0]],
                                      resolution = input_stack.resolution,
                                      indexes = input_stack.indexes[bicluster_labels[which_cluster][0]])
        # Append the converged stack to the list of converged stacks
        converged_stacks.append(converged_stack)
    
    # Return the converged stacks
    logger.debug(f'Returning {n_clusters} IREC clustered stacks.')
    return converged_stacks

def hierarchical_DSC(input_stack: Stack.Stack,
                     cae: Models.ConvolutionalAutoencoder,
                     k_neighbors: int = 7,
                     max_iterations: int = 5,
                     lafm_target_resolution: Tuple[int, int] = None,
                     lafm_sigma: float = 3.0,
                     distinct_cluster_threshold: float = 0.5,
                     stability_threshold = 0.85,
                     min_cluster_size = 150,
                     **kwargs) -> List[Stack.Stack]:
    '''
    Applies Hierarchical Deep Spectral Clustering to a stack of images.

    Args:
        input_stack (Stack.Stack):
            The AFMpy Stack object containing the images to cluster.
        cae (Models.ConvolutionalAutoencoder):
            The convolutional autoencoder to use for calculating the latent feature vectors.
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
        min_cluster_size (int):
            The minimum size of a cluster to be considered valid. Default is 150.
        **kwargs:
            Keyword arguments. Passed to spectral_cluster function.
    Returns:
        List[Stack.Stack]: A list of AFMpy Stack objects. Each stack is a conformational cluster of the input stack.
    '''
    logger.debug(f'Beginning Hierarchical DSC on stack {id(input_stack)}')
    # Extract the images from the stack
    base_images = input_stack.stack
    images = np.copy(base_images)

    base_indexes = input_stack.indexes
    indexes = np.copy(base_indexes)

    # If the LAFM target resolution is not provided, set it to 3x the resolution of the input images
    if lafm_target_resolution is None:
        logger.warning('LAFM target resolution not provided. Setting to 3x the resolution of the input images.')
        lafm_target_resolution = tuple([int(res * 3) for res in input_stack.shape[1:3]])

    # Expand the dimensions of the images so they have the shape (num_images, height, width, channels)
    images = np.expand_dims(images, axis = -1)

    # Normalize the images.
    images = Utilities.norm(images)

    # Calculate the latent feature vectors
    latent_vectors = calculate_LFV(images, cae)

    # Generate the affinity matrix from the latent vectors with locally scaled affinity
    base_affinity_matrix = local_scaled_affinity(latent_vectors, k_neighbors = k_neighbors)
    affinity_matrix = np.copy(base_affinity_matrix)

    output_clusters = []
    # Perform hierarchical spectral clustering on the affinity matrix:
    for iteration in range(max_iterations):
        logger.info(f'Beginning iteration {iteration} of hierarchical DSC.')
        # Perform 2-Spectral Clustering
        logger.debug('Performing 2-Spectral Clustering.')
        cluster_labels_2 = spectral_cluster(affinity_matrix, 2, **kwargs)
        bicluster_labels_2 = np.array([np.where(cluster_labels_2 == i, True, False) for i in range(2)])
        clustered_stacks_2 = np.array([Stack.Stack(images[bicluster],
                                                   resolution = input_stack.resolution,
                                                   indexes = indexes[bicluster]) for bicluster in bicluster_labels_2])
        clustered_lafms_2 = np.array([clustered_stack.calc_LAFM_image(target_resolution = lafm_target_resolution,
                                                                      sigma = lafm_sigma) for clustered_stack in clustered_stacks_2])
        # Calculate the SSIM between the LAFMs
        ssim_2 = SSIM.masked_SSIM(*clustered_lafms_2, threshold_rel = 0.05)
        logger.info(f'SSIM between 2-Clustering LAFMs: {ssim_2:.2f}')

        if ssim_2 > distinct_cluster_threshold:
            # If the SSIM is above the threshold, concatenate the clusters and break the loop.
            logger.info(f'2-Clustering found highly similar clusters. Exiting at iteration {iteration}.')
            concat = clustered_stacks_2[0].concatenate(clustered_stacks_2[1])
            output_clusters.append(concat)
            break

        # Perform the 3-Spectral Clustering
        logger.debug('Performing 3-Spectral Clustering.')
        cluster_labels_3 = spectral_cluster(affinity_matrix, 3, **kwargs)
        bicluster_labels_3 = np.array([np.where(cluster_labels_3 == i, True, False) for i in range(3)])
        clustered_stacks_3 = np.array([Stack.Stack(images[bicluster],
                                                   resolution = input_stack.resolution,
                                                   indexes = indexes[bicluster]) for bicluster in bicluster_labels_3])
        clustered_lafms_3 = np.array([clustered_stack.calc_LAFM_image(target_resolution = lafm_target_resolution,
                                                                      sigma = lafm_sigma) for clustered_stack in clustered_stacks_3])

        # Calculate the SSIM between the LAFMs
        pairs = [(0,1), (0,2), (1,2)]
        ssim_3 = np.array([SSIM.masked_SSIM(clustered_lafms_3[pair[0]],
                                            clustered_lafms_3[pair[1]],
                                            threshold_rel = 0.05) for pair in pairs])
        logger.info(f'SSIM between 3-Clustering LAFMs: {np.round(ssim_3,2)}')
        
        # Calculate the stability of each 2-cluster
        stability_ssims = np.array([[SSIM.masked_SSIM(lafm2, lafm3, threshold_rel = 0.05) for lafm2 in clustered_lafms_2] for lafm3 in clustered_lafms_3])
        logger.info(f'SSIM between 2-Clustering LAFMs and 3-Clustering LAFMs: {np.ravel(np.round(stability_ssims, 2))}')

        # Determine which clusters are stable
        which_stable = np.logical_or(*(stability_ssims > stability_threshold))
        logger.info(f'Cluster stability: {which_stable}')

        # If none of the clusters are stable, break the loop
        if not np.any(which_stable):
            logger.info(f'3-Clustering found no stable clusters. Exiting at iteration {iteration}.')
            concat = clustered_stacks_2[0].concatenate(clustered_stacks_2[1])
            output_clusters.append(concat)
            break

        # If all the clusters are stable, break the loop
        if np.all(which_stable):
            logger.info(f'3-Clustering found all clusters stable. Exiting at iteration {iteration}.')
            for stack in clustered_stacks_2:
                output_clusters.append(stack)
            break

        # Get the stable cluster and append it to the output clusters
        stable_cluster = clustered_stacks_2[which_stable][0]
        output_clusters.append(stable_cluster)

        # Get the unstable clusters    
        unstable_cluster = clustered_stacks_2[~which_stable][0]

        # Get the indexes of the unstable cluster, and update the affinity matrix to only include only the unstable cluster
        unstable_indexes = unstable_cluster.indexes

        # If the unstable cluster is smaller than the minimum cluster size, break the loop and append the cluster to the output clusters
        if unstable_indexes.shape[0] < min_cluster_size:
            logger.info(f'Unstable cluster is smaller than minimum cluster size. Exiting at iteration {iteration}.')
            output_clusters.append(unstable_cluster)
            break

        # If the maximum iterations is met, break the loop and append the cluster to the output clusters.
        if iteration == max_iterations - 1:
            logger.warning(f'Maximum iterations reached. Appending final unstable cluster to output. Exiting at iteration {iteration}.')
            output_clusters.append(unstable_cluster)
            break
        
        # Update the images and affinity matrix to only include the unstable cluster
        logger.info(f'Updating images and affinity matrix to only include unstable cluster. {unstable_indexes.shape[0]} images remaining.')
        affinity_matrix = base_affinity_matrix[unstable_indexes][:, unstable_indexes]
        images = base_images[unstable_indexes]
        indexes = base_indexes[unstable_indexes]

    # Return the output clusters containing all stable clusters.
    return output_clusters
        
# def hierarchical_DSC(image_stack: np.ndarray,
#                      k_neighbors: int = 7,
#                      max_iterations: int = 5,
#                      lafm_target_resolution: Tuple[int, int] = None,
#                      lafm_sigma: float = 3.0,
#                      distinct_cluster_threshold: float = 0.5,
#                      stability_threshold = 0.85,
#                      min_cluster_size = 150,
#                      autoencoder_params: Dict[str, any] = None,
#                      return_traceback: bool = True) -> np.ndarray:
#     '''
#     Applies Hierarchical Deep Spectral Clustering to a stack of input images.

#     Args:
#         image_stack (np.ndarray):
#             A stack of images to cluster.
#         k_neighbors (int):
#             The number of nearest neighbors to consider for scaling the sigma values of the Locally Scaled 
#             Affinity Matrix.
#         max_iterations (int):
#             The maximum number of iterations to perform for the hierarchical clustering.
#         lafm_target_resolution (Tuple[int, int]):
#             The target resolution to use for the LAFM2D function. If None, 3x the resolution of the input images
#             is used.
#         lafm_sigma (float):
#             The sigma value to use for the LAFM2D function. Default is 3.0 pixels.
#         stability_threshold (float):
#             The threshold for the stability of the clusters. If the SSIM between the LAFMs of the 2 and 3 clusters
#             is above this threshold, the clusters are considered stable. Default is 0.9.
#         min_cluster_ratio (float):
#             The minimum ratio of images to cluster. If the number of unclustered images falls below this ratio, the
#             algorithm will stop. Default is 0.15 for 15% of the total stack. 
#         autoencoder_params (Dict[str, any]):
#             The autoencoder parameters to use for the DSC function. If None, the default_autoencoder_params are 
#             used.
#             See AFM.REC.default_autoencoder_params for the default parameters.
#     Returns:
#         np.ndarray: The cluster labels for the stack of images.
#     '''
#     # Check to see if the target resolution is None. If it is, set it to 3x the resolution of the input images.
#     if lafm_target_resolution is None:
#         lafm_target_resolution = (image_stack.shape[1] * 3, image_stack.shape[2] * 3)

#     # Get the stack size. This is useful for a few things, but will determine the minimum cluster size.
#     stack_size = image_stack.shape[0]
#     # Create an array of mutable indexes. This array will be used to keep track of which images have been
#     # clustered and which have not between iterations.
#     mutable_indexes = np.arange(stack_size)
#     # Create a mutable copy of the image stack. Because each iteration acts on a subset of the stack, we need to 
#     # be able to modify it.
#     mutable_stack = image_stack.copy()

#     # Calculate the latent feature vectors
#     latent_vectors = calculate_LFV(image_stack, autoencoder_params)

#     # Generate the affinity matrix from the latent vectors with locally scaled affinity
#     affinity_matrix = local_scaled_affinity(latent_vectors, k_neighbors = k_neighbors)

#     output_indexes = []
#     traceback = {'lafms': [], 'ssims': []}
#     # Perform hierarchical spectral clustering'
#     for iteration in range(max_iterations):

#         # If the maximum iterations is met, break the loop and put the remaining images in the output indexes.
#         if iteration == max_iterations - 1:
#             output_indexes.append(mutable_indexes)
#             logger.info(f'Maximum iterations reached. Exiting at iteration {iteration}.')
#             break

#         # Perform 2-Spectral Clustering
#         cluster_labels2 = spectral_cluster(affinity_matrix, 2)
#         clusters2 = np.array([np.where(cluster_labels2 == i, True, False) for i in range(2)])
#         # calculate the LAFM2D for each cluster.
#         lafms2 = np.array([LAFM2D(mutable_stack[cluster],
#                                   target_resolution = lafm_target_resolution, 
#                                   sigma = lafm_sigma ) for cluster in clusters2])
#         # Calculate the SSIM between the LAFMs
#         ssim2 = SSIM.masked_SSIM(*lafms2, threshold_rel = 0.05)

#         if ssim2 > distinct_cluster_threshold:
#             output_indexes.append(mutable_indexes)
#             logger.info(f'2-Clustering found highly similar clusters. Exiting at iteration {iteration}.')
#             break

#         # Perform the 3-Spectral Clustering
#         cluster_labels3 = spectral_cluster(affinity_matrix, 3)
#         clusters3 = np.array([np.where(cluster_labels3 == i, True, False) for i in range(3)])
#         # Calculate the LAFM2D for each cluster
#         lafms3 = np.array([LAFM2D(mutable_stack[cluster],
#                                   target_resolution = lafm_target_resolution,
#                                   sigma = lafm_sigma) for cluster in clusters3])
#         # Calculate the SSIM between the LAFMs
#         pairs = [(0,1), (0,2), (1,2)]
#         ssims3 = np.array([SSIM.masked_SSIM(lafms3[pair[0]],
#                                       lafms3[pair[1]],
#                                       threshold_rel = 0.05) for pair in pairs])

#         # Calculate the stability of each 2-cluster by the SSIM of the LAFMs
#         stability_ssims = np.array([[SSIM.masked_SSIM(lafm2, 
#                                                  lafm3, 
#                                                  threshold_rel = 0.05) for lafm3 in lafms3] for lafm2 in lafms2])

#         traceback['lafms'].append((lafms2, lafms3))
#         traceback['ssims'].append((ssim2, ssims3, stability_ssims))
        
#         # Determine which clusters are stable
#         which_stable = np.logical_or(*(stability_ssims > stability_threshold))

#         # If none of the clusters are stable, break the loop
#         if not np.any(which_stable):
#             output_indexes.append(mutable_indexes)
#             logger.info(f'No stable clusters found. Exiting at iteration {iteration}.')
#             break

#         # If all the clusters are stable, break the loop
#         if np.all(which_stable):
#             for cluster in clusters2:
#                 output_indexes.append(mutable_indexes[cluster])
#             logger.info(f'All clusters are stable. Exiting at iteration {iteration}.')
#             break

#         # Extract the stable clusters
#         stable_clusters = clusters3[which_stable]

#         # Append the stable clusters to the output indexes and create the compliment. The compliment are the
#         # images remaining unstably clustered images.
#         compliment = []

#         # Remove any clusters that are too small
#         # too_small = np.sum(stable_clusters, axis = 1) < min_cluster_size
#         # stable_clusters = stable_clusters[np.logical_not(too_small)]
        
#         # if len(stable_clusters) == 0:
#         #     output_indexes.append(mutable_indexes)
#         #     logger.info(f'All stable clusters are too small. Exiting at iteration {iteration}.')
#         #     break

#         for stable_cluster in stable_clusters:
#             stable_indexes = mutable_indexes[stable_cluster]
#             output_indexes.append(stable_indexes)

#             compliment.append(np.logical_not(stable_cluster))
#         remaining_unclustered = np.all(compliment, axis = 0)
        
#         # If the remaining unclustered images are less than the minimum cluster ratio, break the loop and put the
#         # remaining images in the output indexes.
#         if np.sum(remaining_unclustered) < min_cluster_size:
#             output_indexes.append(mutable_indexes[remaining_unclustered])
#             logger.info(f'Minimum cluster size reached. Exiting at iteration {iteration}.')
#             break

#         # Update the affinity matrix, mutable indexes, and mutable stack with the remaining unclustered images.
#         affinity_matrix = affinity_matrix[remaining_unclustered][:, remaining_unclustered]
#         mutable_indexes = mutable_indexes[remaining_unclustered]
#         mutable_stack = image_stack[mutable_indexes]

#     # Return the output indexes and traceback if requested.
#     if return_traceback:
#         return output_indexes, traceback
#     else:
#         return output_indexes