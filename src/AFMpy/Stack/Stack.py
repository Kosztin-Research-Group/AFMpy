import copy
import lzma
import pickle
import logging
import warnings
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from AFMpy import Utilities, LAFM

# Create a logger for this module.
logger = logging.getLogger(__name__)

__all__ = ['Stack']

class Stack():
    '''
    Class for AFM image stacks. Includes the image stack, the shape of stack, 

    Attributes:
        images (numpy.ndarray):
            The simulated AFM image stack.
        grid (numpy.ndarray):
            The grid of pixel coordinates used to generate the stack.
        shape (tuple):
            The shape of images in the image stack. (n_images, n_pixels_x, n_pixels_y)
        resolution (float):
            The resolution of images in the stack. Units are in Angstroms per pixel. The resolution must be square, i.e.
            the pixel width and height must be equal.
        **metadata (Any):
            The metadata associated with the image stack. Key-value pairs of metadata.
    '''

    def __init__(self,
                 images: np.ndarray,
                 resolution: float,
                 indexes = None,
                 **metadata: Any) -> None:
        '''
        Initialization method for SimAFM_Stack.
        
        Args:
            images (numpy.ndarray):
                The simulated AFM image stack. numpy.ndarray with shape (n_images, n_pixels_x, n_pixels_y).
            resolution (float):
                The resolution of images in the stack. Units are in Angstroms per pixel.
            indexes (numpy.ndarray):
                The indexes of the stack. numpy.ndarray with shape (n_images). Default is None.
                This should be left as none, unless creating a substack, in which the indexes should be inherited from the
                parent stack as a means of tracking the original order of the images.
            **metadata (Any):
                Key-value pairs of metadata to add to the metadata attribute.
        Returns:
            None
        '''
        # Set the images attribute.
        self._images = images

        # Set the shape attribute.
        self._shape = images.shape

        # Set the resolution attribute.
        self._resolution = resolution

        # Set the indexes attribute.
        if indexes is None:
            self._indexes = np.arange(self._shape[0])
        else:
            self._indexes = indexes

        # Set the metadata attributes.
        self._metadata = metadata

        # Set the processed image attributes to None.
        self._mean_image = None
        self._LAFM_image = None

    #############################
    ##### Save/Load Methods #####
    #############################

    def save_pickle(self,
                    pickle_filepath: str,
                    private_key_filepath: str = None,
                    sign_pickle: bool = True) -> None:
        '''
        Saves the SimAFM_Stack object to a pickle file. The pickle file is then digitally signed with a private key.

        Args:
            pickle_filepath (str):
                The path to save the pickle file.
            private_key_filepath (str):
                The path to your private key file.
            sign_pickle (bool):
                Whether or not to digitally sign the pickle file. Default is True. Signing the pickle is recommended if you
                intend to distribute the pickle file, so the recipient can verify the authenticity of the file.
        Returns:
            None
        '''
        # Save the pickle file.
        logger.debug(f'Writing SimAFM_Stack {id(self)} to {pickle_filepath}.')
        with open(pickle_filepath, 'wb') as file:
            pickle.dump(self, file)
        
        if sign_pickle:
            if not private_key_filepath:
                logger.error('Private key file path not provided. Cannot sign pickle file.')
                raise ValueError('Private key file path not provided. Cannot sign pickle file.')
            else:
                # Digitally sign the pickle file.
                logger.debug(f'Signing {pickle_filepath}.')
                Utilities.Signature.sign_file(pickle_filepath, private_key_filepath)
        else:
            logger.warning('Pickle signature disabled. This is not recommended if you intend to distribute the pickle file.')
            warnings.warn('Pickle signature disabled. This is not recommended if you intend to distribute the pickle file.')

        
    @classmethod
    def load_pickle(cls,
                    pickle_filepath: str,
                    public_key_filepath: str = None,
                    verify_pickle: bool = True) -> 'Stack':
        '''
        Loads a SimAFM_Stack object from a pickle file. The pickle file must be verified via a digital signature with a public key.

        Args:
            pickle_filepath (str):
                The path to the pickle file.
            public_key_filepath (str):
                The path to your public key file.
            verify_pickle (bool):
                Whether or not to verify the pickle file. Default is True. Pickle files may execute arbitrary code when
                loaded. Verifying the pickle file is recommended if you are loading a pickle file from an outside source.
                This does not ensure the pickle file is safe, but it does ensure the file has not been tampered with.
        Returns:
            SimAFM_Stack:
                The loaded SimAFM_Stack object.
        '''
        if verify_pickle:
            if not public_key_filepath:
                logger.error('Public key file path not provided. Cannot verify pickle file.')
                raise ValueError('Public key file path not provided. Cannot verify pickle file.')
            else:
                # Verify the digital signature of the pickle file.
                logger.debug(f'Verifying {pickle_filepath}.')
                Utilities.Signature.verify_file(pickle_filepath, public_key_filepath)
        else:
            logger.warning('Pickle verification bypassed by user request. This is not recommended if you are loading a pickle file from an outside source.')
            warnings.warn('Pickle verification bypassed by user request. This is not recommended if you are loading a pickle file from an outside source.')

        # Load the pickle file.
        logger.debug(f'Loading SimAFM_Stack from {pickle_filepath}.')
        with open(pickle_filepath, 'rb') as file:
            obj = pickle.load(file)
        
        logger.debug(f'Successfully loaded SimAFM_Stack {id(obj)} from {pickle_filepath}.')
        return obj
    
    def save_compressed_pickle(self,
                               pickle_filepath: str,
                               private_key_filepath: str = None,
                               sign_pickle: bool = True) -> None:
        '''
        Saves the SimAFM_Stack object to an LZMA compressed pickle file. The pickle file can be digitally signed with a private key.

        Args:
            pickle_filepath (str):
                The path to save the LZMA compressed pickle file.
            private_key_filepath (str):
                The path to your private key file.
            sign_pickle (bool):
                Whether or not to digitally sign the pickle file. Default is True. 
                Signing the pickle is recommended if you intend to distribute the pickle file.

        Returns:
            None
        '''
        # Save the LZMA compressed pickle file.
        logger.debug(f'Writing SimAFM_Stack {id(self)} to {pickle_filepath}.')
        with lzma.open(pickle_filepath, 'wb') as file:
            pickle.dump(self, file)

        if sign_pickle:
            if not private_key_filepath:
                logger.error('Private key file path not provided. Cannot sign pickle file.')
                raise ValueError('Private key file path not provided. Cannot sign pickle file.')
            else:
                # Digitally sign the pickle file.
                logger.debug(f'Signing {pickle_filepath}.')
                Utilities.Signature.sign_file(pickle_filepath, private_key_filepath)
        else:
            logger.warning('Pickle signature disabled. This is not recommended if you intend to distribute the pickle file.')
            warnings.warn('Pickle signature disabled. This is not recommended if you intend to distribute the pickle file.')

    @classmethod
    def load_compressed_pickle(cls,
                               pickle_filepath: str,
                               public_key_filepath: str = None,
                               verify_pickle: bool = True) -> 'Stack':
        '''
        Loads a SimAFM_Stack object from an LZMA compressed pickle file. The pickle file can be verified via a digital signature with a public key.

        Args:
            pickle_filepath (str):
                The path to the LZMA compressed pickle file.
            public_key_filepath (str):
                The path to the public key file.
            verify_pickle (bool):
                Whether or not to verify the pickle file. Default is True. 
                Pickle files may execute arbitrary code when loaded. Verifying the pickle file is recommended 
                if you are loading a pickle file from an outside source. 
                This does not ensure the pickle file is safe, but it does ensure the file has not been tampered with.

        Returns:
            SimAFM_Stack:
                The loaded SimAFM_Stack object.
        '''
        if verify_pickle:
            if not public_key_filepath:
                logger.error('Public key file path not provided. Cannot verify pickle file.')
                raise ValueError('Public key file path not provided. Cannot verify pickle file.')
            else:
                # Verify the digital signature of the pickle file.
                logger.debug(f'Verifying {pickle_filepath}.')
                Utilities.Signature.verify_file(pickle_filepath, public_key_filepath)
        else:
            logger.warning('Pickle verification bypassed by user request. This is not recommended if you are loading a pickle file from an outside source.')
            warnings.warn('Pickle verification bypassed by user request. This is not recommended if you are loading a pickle file from an outside source.')

        # Load the LZMA compressed pickle file.
        logger.debug(f'Loading SimAFM_Stack from {pickle_filepath}.')
        with lzma.open(pickle_filepath, 'rb') as file:
            obj = pickle.load(file)

        logger.debug(f'Successfully loaded SimAFM_Stack {id(obj)} from {pickle_filepath}.')
        return obj
    
    ###############################
    ##### Processing Methods ######
    ###############################
    
    def calc_mean_image(self,
                        force_recalc: bool = False,
                        **kwargs) -> np.ndarray:
        '''
        Calculates the mean image of the image stack.

        Args:
            force_recalc (bool):
                Whether or not to force recalculation of the mean image. Default is False.
            **kwargs:
                Additional keyword arguments to be passed to np.mean. Refer to the numpy documentation for more details:
        Returns:
            numpy.ndarray:
                The mean image of the stack.
        '''
        # Check if force_recalc is True and that recalculation is being forced.
        if force_recalc:
            logger.debug(f'Forcing recalculation of the mean image for SimAFM_Stack {id(self)}.')
        # Check if the mean image is already calculated.
        if force_recalc or self._mean_image is None:
            logger.debug(f'Calculating the mean image of for SimAFM_Stack {id(self)}.')
            self._mean_image = np.mean(self._images, axis = 0, **kwargs)
            return self._mean_image
        else:
            logger.warning(f'Mean image already calculated for SimAFM_Stack {id(self)}. Returning the mean image without recalculation. Set force_recalc to True to force recalculation.')
            return self._mean_image
        
    def calc_LAFM_image(self,
                        target_resolution: Tuple[int, int],
                        sigma: float,
                        force_recalc: bool = False,
                        **kwargs) -> np.ndarray:
        '''
        Calculates the LAFM image of the stack.

        Args:
            target_resolution (Tuple[int, int]):
                The target resolution of the LAFM image.
            sigma (float):
                Gaussian broadening width for generating the peak probability distribution. This value is in terms of pixels of 
                the expanded image. For example, if the stack is expanded by a factor of 3 and the original resolution is 1.5 
                Angstroms per pixel, applying a Gaussian broadening of 3 pixels will result in a 1.5 Angstrom broadening.
            force_recalc (bool):
                Whether or not to force recalculation of the LAFM image. Default is False.
            **kwargs:
                Additional keyword arguments to be passed to skimage.feature.peak_local_max. Refer to the skimage documentation 
                for more details: https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.peak_local_max
        Returns:
            numpy.ndarray:
                The LAFM image of the stack.
        '''
        # Check if force_recalc is True and that recalculation is being forced.
        if force_recalc:
            logger.debug(f'Forcing recalculation of the LAFM image for SimAFM_Stack {id(self)}.')
        # Check if the LAFM image is already calculated.
        if force_recalc or self._LAFM_image is None:
            logger.debug(f'Calculating the LAFM image of for SimAFM_Stack {id(self)}.')
            self._LAFM_image = LAFM.LAFM2D(self._images, target_resolution, sigma, **kwargs)
            return self._LAFM_image
        else:
            logger.warning(f'LAFM image already calculated for SimAFM_Stack {id(self)}. Returning the LAFM image without recalculation. Set force_recalc to True to force recalculation.')
            return self._LAFM_image

    
    ########################
    ##### Misc Methods #####
    ########################

    def copy(self) -> 'Stack':
        '''
        Returns a deep copy of the SimAFM_Stack object. This prevents the mutable attributes of the object from being modified.

        Returns:
            SimAFM_Stack:
                A deep copy of the SimAFM_Stack object.
        '''
        logger.debug(f'Creating a deep copy of SimAFM_Stack {id(self)} instance.')
        return copy.deepcopy(self)

    def shuffle(self) -> None:
        '''
        Shuffles the image stack in place.
        '''
        # Generate a random permutation of the indexes and shuffle the stack.
        logger.debug(f'Shuffling SimAFM_Stack {id(self)}.')
        self._indexes = np.random.permutation(self._indexes)
        self._images = self._images[self._indexes]

        # Set the shuffled metadata attribute to True.
        self.add_metadata(shuffled = True)

    def unshuffle(self) -> None:
        '''
        Unshuffles the image stack in place.
        '''
        # Sort the indexes and unshuffle the stack.
        logger.debug(f'Unshuffling SimAFM_Stack {id(self)}.')
        self._indexes = np.argsort(self._indexes)
        self._images = self._images[self._indexes]

        # Set the shuffled metadata attribute to False.
        self.add_metadata(shuffled = False)

    def concatenate(self,
                    other: 'Stack',
                    **metadata: Any) -> 'Stack':
        '''
        Concatenate another SimAFM_Stack object to the current SimAFM_Stack object.
        Automatically checks if the stacks are compatible, and updates the metadata with concatenated = True.
        Args:
            other (SimAFM_Stack):
                The SimAFM_Stack object to concatenate.
            **metadata (Any):
                Key-value pairs of metadata to add to the metadata attribute.
                Should be used to add metadata related to the concatenation event.
        Returns:
            The concatenated SimAFM_Stack object.
        '''
        # Check if the other object is a SimAFM_Stack object.
        if not isinstance(other, Stack):
            logger.error('The object to concatenate is not a SimAFM_Stack object.')
            raise TypeError('The object to concatenate is not a SimAFM_Stack object.')
        # Check if the other object has the same resolution.
        if self._resolution != other.resolution:
            logger.error('The resolution of the two SimAFM_Stack objects are not the same.')
            raise ValueError('The resolution of the two SimAFM_Stack objects are not the same.')
        
        # Concatenate the two stacks.
        logger.debug(f'Concatenating SimAFM_Stack {id(self)} and SimAFM_Stack {id(other)}.')
        self._images = np.concatenate((self._images, other.stack), axis = 0)
        self._indexes = np.concatenate((self._indexes, other.indexes), axis = 0)

        # Update the shape attribute.
        self._shape = self._images.shape

        # Add metadata to the metadata attribute.
        self.add_metadata(**metadata)
        self.add_metadata(concatenated = True)

        # Reseat the processed image attributes.
        self._mean_image = None
        self._LAFM_image = None
        return self

    ###########################
    ##### Plottin Methods #####
    ###########################

    def plot_image(self,
                   index: int,
                   ax: plt.Axes = None,
                   **kwargs) -> plt.Axes:
        '''
        Plot a single image from the stack.

        Args:
            index (int):
                The index of the image to plot.
            ax (matplotlib.axes.Axes):
                The matplotlib axes to plot on. If None, uses plt.gca().
            **kwargs:
                Additional keyword arguments to be passed to plt.imshow.
        Returns:
            ax (matplotlib.axes.Axes):
                The matplotlib axes with the plotted image.
        '''
        # If the axis is None, use plt.gca().
        ax = ax or plt.gca()

        # Show the image
        ax.imshow(self._images[index], **kwargs)
        return ax
    
    def plot_mean_image(self,
                        ax: plt.Axes = None,
                        **kwargs) -> plt.Axes:
        '''
        Plot the mean image of the stack.
        Args:
            ax (matplotlib.axes.Axes):
                The matplotlib axes to plot on. If None, uses plt.gca().
            **kwargs:
                Additional keyword arguments to be passed to plt.imshow.
        Returns:
            ax (matplotlib.axes.Axes):
                The matplotlib axes with the plotted mean image.
        '''
        # If the axis is None, use plt.gca().
        ax = ax or plt.gca()

        # If the mean image is not computed, raise an error.
        if self._mean_image is None:
            logger.error('Mean image has not been calculated. Call calc_mean_image() to calculate the mean image.')
            raise ValueError('Mean image has not been calculated. Call calc_mean_image() to calculate the mean image.')
        
        # Show the mean image
        ax.imshow(self._mean_image, **kwargs)
        return ax
    
    def plot_LAFM_image(self,
                        ax: plt.Axes = None,
                        **kwargs) -> plt.Axes:
        '''
        Plot the LAFM image of the stack.
        Args:
            ax (matplotlib.axes.Axes):
                The matplotlib axes to plot on. If None, uses plt.gca().
            **kwargs:
                Additional keyword arguments to be passed to plt.imshow.
        Returns:
            ax (matplotlib.axes.Axes):
                The matplotlib axes with the plotted LAFM image.
        '''
        # If the axis is None, use plt.gca().
        ax = ax or plt.gca()

        # If the LAFM image is not computed, raise an error.
        if self._LAFM_image is None:
            logger.error('LAFM image has not been calculated. Call calc_LAFM_image() to calculate the LAFM image.')
            raise ValueError('LAFM image has not been calculated. Call calc_LAFM_image() to calculate the LAFM image.')
        
        # Show the LAFM image
        ax.imshow(self._LAFM_image, **kwargs)
        return ax

    ######################
    ##### Properties #####
    ######################

    @property
    def images(self) -> np.ndarray:
        '''
        The simulated AFM image stack. numpy.ndarray with shape (n_images, n_pixels_x, n_pixels_y).
        '''
        return self._images
    
    @property
    def shape(self) -> tuple:
        '''
        The shape of images in the image stack. (n_images, n_pixels_x, n_pixels_y)
        '''
        return self._shape
    
    @property
    def indexes(self) -> np.ndarray:
        '''
        The indexes of the stack. numpy.ndarray with shape (n_images).
        '''
        return self._indexes

    @property
    def resolution(self) -> float:
        '''
        The resolution of the image stack. Units are in Angstroms per pixel.
        '''
        if self._resolution is None:
            logger.warning('Resolution is ambiguous. x_resolution and y_resolution are not equal.')
        return self._resolution
    
    ######################################
    ##### Processed Image Properties #####
    ######################################
    
    @property
    def mean_image(self) -> np.ndarray:
        '''
        The mean image of the stack.
        '''
        if self._mean_image is None:
            logger.error('Mean image has not been calculated. Call calc_mean_image() to calculate the mean image.')
            raise ValueError('Mean image has not been calculated. Call calc_mean_image() to calculate the mean image.')
        return self._mean_image
    
    @property
    def LAFM_image(self) -> np.ndarray:
        '''
        The LAFM image of the stack.
        '''
        if self._LAFM_image is None:
            logger.error('LAFM image has not been calculated. Call calc_LAFM_image() to calculate the LAFM image.')
            raise ValueError('LAFM image has not been calculated. Call calc_LAFM_image() to calculate the LAFM image.')
        return self._LAFM_image
    
    #############################
    ##### Metadata Handling #####
    #############################

    @property
    def metadata(self) -> Dict[str, Any]:
        '''
        The metadata associated with the image stack.
        '''
        return self._metadata
    
    def add_metadata(self, **metadata: Any) -> None:
        '''
        Adds metadata to the metadata attribute.

        Args:
            **metadata (Any):
                Key-value pairs of metadata to add to the metadata attribute.
        Returns:
            None
        '''
        logging.debug(f'Adding {metadata} to SimAFM_Stack {id(self)} metadata.')
        self._metadata.update(metadata)
    
    def remove_metadata(self, *keys: str) -> None:
        '''
        Removes metadata from the metadata attribute.

        Args:
            *keys (str):
                The keys of the metadata to remove.
        Returns:
            None
        '''
        for key in keys:
            logging.debug(f'Removing {key} from SimAFM_Stack {id(self) }metadata.')
            self._metadata.pop(key, None)

    def get_metadata(self, key: str) -> Any:
        '''
        Returns the value of a metadata attribute. If the key is not found, warns the user and returns None.

        Args:
            key (str):
                The key of the metadata attribute to return.
        Returns:
            Any:
                The value of the metadata attribute.
        '''
        logger.debug(f'Retrieving "{key}" from SimAFM_Stack {id(self)} metadata.')
        if key not in self._metadata.keys():
            logger.warning(f'Metadata "{key}" not found. Returning None.')

        return self._metadata.get(key, None)

    def display_metadata(self) -> None:
        '''
        Method for displaying all metadata pairs from the metadata attribute.
        '''
        # Determine if the environment is interactive (e.g. Jupyter Notebook).
        try:
            from IPython import get_ipython
        except ImportError:
            get_ipython = None

        # If the environment is interactive, use IPython to display the metadata. Otherwise, print the metadata.
        ip = get_ipython() if get_ipython else None
        if ip is not None:
            # Display the metadata as a pandas DataFrame.
            logger.debug(f'Inside Interactive enviroment. Displaying SimAFM_Stack {id(self)} metadata.')
            from IPython.display import display

            metadata_df = pd.DataFrame(self._metadata.items(), columns = ['Key', 'Value'])
            display(metadata_df.style.hide(axis="index"))
        else:
            # Print the metadata as nicely formatted text.
            logger.debug(f'Not in Interactive enviroment. Printing SimAFM_Stack {id(self)} metadata.')
            print('SimAFM_Stack Metadata:')
            for key, value in self._metadata.items():
                print(f'{key}: {value}')