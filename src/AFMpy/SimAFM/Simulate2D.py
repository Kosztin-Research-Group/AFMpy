import os
import hashlib
import copy
import pickle
import lzma
import multiprocessing as mp
from typing import Any, Dict, Optional
from abc import ABC, abstractmethod

import numpy as np
import MDAnalysis as MDA

from AFMpy import Utilities

logger = Utilities.Logging.make_module_logger(__name__)

__all__ = ['VDW_Dict', 'AA_VDW_Dict', 'CG_VDW_Dict', 'SimAFM_Stack', 'make_grid', 'make_radius_array', 'simulate_AFM2D', 'simulate_AFM2D_stack', 'simulate_AFM2D_stack_MP']

# Warning message for hash verification. This is used many times in the code and is stored here for easy access.
hash_warning = 'Beware: Compressed objects are not secure against erroneous or maliciously constructed data. Never unpack data recieved from an untrusted or unauthenticated source. See https://docs.python.org/3/library/pickle.html.'

class VDW_Dict(ABC, dict):
    '''
    Base class for Van der Waals radius dictionary.
    '''
    def __init__(self):
        super().__init__()
        self.update(self.init_dict())

    @abstractmethod
    def init_dict(self):
        pass

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'
    
    def __str__(self) -> str:
        return f'{self.__class__.__name__}'
    
class AA_VDW_Dict(VDW_Dict):
    '''
    VDW radius dictionary for All Atom (AA) CHARMM force field.
    '''
    def init_dict(self):
        return {'H': 0.2245, 'HC': .2245, 'HA': 1.32, 'HP': 1.3582, 'HB1': 1.32, 'HB2': 1.32,'HR1': 0.9, 
                'HR2': 0.7, 'HR3': 1.468, 'HS': 0.45, 'HE1': 1.25, 'HE2': 1.26, 'HA1': 1.34, 'HA2': 1.34, 
                'HA3': 1.34, 'C': 2.0, 'CA': 1.9924, 'CT': 2.275, 'CT1': 2.0, 'CT2': 2.010, 'CT2A': 2.010, 
                'CT3': 2.040, 'CPH1': 1.80, 'CPH2': 1.80, 'CPT': 1.86, 'CY': 1.99, 'CP1': 2.275, 
                'CP2': 2.175, 'CP3': 2.175, 'CC': 2.0, 'CD': 2.0, 'CS': 2.20, 'CE1': 2.09, 'CE2': 2.08, 
                'CAI': 1.99, 'C3': 2.275, 'N': 1.85, 'NR1': 1.85, 'NR2': 1.85, 'NR3': 1.85, 'NH1': 1.85, 
                'NH2': 1.85, 'NH3': 1.85, 'NC2': 1.85, 'NY': 1.85, 'NP': 1.85, 'O': 1.7, 'OB': 1.7,
                'OC': 1.7, 'OH1': 1.77, 'OS': 1.77, 'S': 2.0, 'SM': 1.975, 'SS': 2.2}
    
class CG_VDW_Dict(VDW_Dict):
    '''
    VDW radius dictionary for Coarse Grained (CG) Beads Martini force field.
    '''
    def init_dict(self):
        return {'B': 2.64, 'BB':2.64,'S':2.30, 'SC1':2.30, 'SC2':2.30, 'SC3':2.30, 'SC4':2.30, 'SCN':2.30,
                'SCP':2.30}
    
class SimAFM_Stack():
    '''
    Class for Simulated AFM image stack. Includes parameters used to generate the stack and methods for saving and loading stacks.

    Attributes:
        stack (numpy.ndarray):
            The simulated AFM image stack.
        grid (numpy.ndarray):
            The grid of pixel coordinates used to generate the stack.
        shape (tuple):
            The shape of images in the image stack. (n_images, n_pixels_x, n_pixels_y)
        boundaries (tuple):
            The boundaries of the image stack. Units are in Angstroms. ((x_min, x_max), (y_min, y_max))
            Additionally, x_boundaries and y_boundaries are available, in case they are needed separately.
        resolution (float):
            The resolution of images in the stack. Units are in Angstroms per pixel. If the resolution along the x axis is
            not equal to the resolution along the y axis, resolution will be set to None as it is ambiguous.
            Additionally, x_resolution and y_resolution are available, in case they are needed separately.
        metadata (Dict[str, Any]):
            The metadata associated with the image stack. Key-value pairs of metadata.
    '''

    def __init__(self,
                 stack: np.ndarray,
                 grid: np.ndarray,
                 **metadata: Any) -> None:
        '''
        Initialization method for SimAFM_Stack.
        
        Args:
            stack (numpy.ndarray):
                The simulated AFM image stack. numpy.ndarray with shape (n_images, n_pixels_x, n_pixels_y).
            grid (numpy.ndarray):
                The grid of pixel coordinates used to generate the stack. numpy.ndarray with shape (n_pixels_x, n_pixels_y, 2).
            **metadata (Any):
                Key-value pairs of metadata to add to the metadata attribute.
        Returns:
            None
        '''
        
        # Set the stack attribute.
        self._stack = stack

        # Set the shape attribute.
        self._shape = stack.shape

        # Set the grid attribute.
        self._grid = grid

        # Set the indexes attribute.
        self._indexes = np.arange(self._shape[0])

        # Set the boundaries attributes.
        self._x_boundaries = (grid[...,0].min(), grid[...,0].max())
        self._y_boundaries = (grid[...,1].min(), grid[...,1].max())
        self._boundaries = (self._x_boundaries, self._y_boundaries)

        # Calculate the resolution.
        self._x_resolution = (self._x_boundaries[1] - self._x_boundaries[0]) / self._shape[1]
        self._y_resolution = (self._y_boundaries[1] - self._y_boundaries[0]) / self._shape[2]
        
        # Check if the resolution along the x axis is equal to the resolution along the y axis. If not, warn the user.
        if self._x_resolution != self._y_resolution:
            logger.warning('Resolution along the x axis is not equal to the resolution along the y axis. This may cause images to appear stretched, the image resolution becomes ambiguous, and it will be set to None. This may cause unexpected behavior.')
            self._resolution = None
        else:
            self._resolution = self._x_resolution

        # Set the metadata attributes.
        self._metadata = metadata

    #############################
    ##### Save/Load Methods #####
    #############################

    @staticmethod
    def _hash_file(filepath: str) -> str:
        '''
        Calculates the sha256 hash of a file.

        Args:
            filepath (str):
                The path to the file to hash.
        Returns:
            str:
                The sha256 hash of the file.
        '''
        # Initialize the hash object.
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as file:
            # Read the file in chunks and update the hash object.
            for chunk in iter(lambda: file.read(4096), b''):
                sha256.update(chunk)
        # Return the hexdigest of the hash object.
        return sha256.hexdigest()
    
    def save_pickle(self,
                    filepath: str,
                    hash: bool = True) -> None:
        '''
        Saves the SimAFM_Stack object to a pickle file.

        Args:
            filepath (str):
                The path to save the pickle file.
            hash (bool):
                Whether or not to save the sha256 hash of the file. Default is True.
        Returns:
            None
        '''
        # Save the pickle file.
        with open(filepath, 'wb') as file:
            pickle.dump(self, file)
        
        # Save the hash of the file.
        if hash:
            hash_str = self._hash_file(filepath)
            with open(f'{filepath}.sha256', 'w') as file:
                file.write(hash_str)
        else:
            # Warn the user that hashing is disabled.
            logger.warning(f'Hashing disabled. {hash_warning}')
        
    @classmethod
    def load_pickle(cls,
                    filepath: str,
                    verify_hash: bool = True) -> 'SimAFM_Stack':
        '''
        Loads a SimAFM_Stack object from a pickle file. If verify_hash is True, the sha256 hash of the file will be checked 
        against the hash file with the same name as the pickle file and the extension .sha256. If the hashes do not match,
        a ValueError will be raised. If verify_hash is False, the hash will not be checked. Pickles are not
        robust against eroneous or maliciously constructed data. It is recommended that you never unpack data recieved
        from untrusted or unauthenticated sources. See https://docs.python.org/3/library/pickle.html.

        Args:
            filepath (str):
                The path to the pickle file.
            verify_hash (bool):
                Whether or not to verify the hash of the file. Default is True.
        Returns:
            SimAFM_Stack:
                The loaded SimAFM_Stack object.
        '''
        # Verify the hash of the file
        if verify_hash:
            hash_str = cls._hash_file(filepath)

            # Try to open the hash file. If it does not exist, raise a FileNotFoundError.
            try:
                with open(f'{filepath}.sha256', 'r') as file:
                    file_hash = file.read()
            except FileNotFoundError:
                logger.error(f'Hash file not found. Cannot verify the integrity of the pickle file. Verify the integrity of the file manually or disable hash verification. {hash_warning}')
                raise FileNotFoundError(f'Hash file not found. Cannot verify the integrity of the pickle file. Verify the integrity of the file manually or disable hash verification. {hash_warning}')
            
            # Check if the hash of the file matches the hash in the hash file. If not, raise a ValueError.
            if hash_str != file_hash:
                logger.error(f'Hash mismatch. {hash_warning}')
                raise ValueError(f'Hash mismatch. Cannot verify the integrity of the pickle file. Verify the integrity of the file manually or disable hash verification. {hash_warning}')
            
            # Load the pickle file.
            with open(filepath, 'rb') as file:
                obj = pickle.load(file)
        
        else:
            # Warn the user that hash verification is disabled.
            logger.warning(f'Hash verification disabled. {hash_warning}')
            
            # Load the pickle file.
            with open(filepath, 'rb') as file:
                obj = pickle.load(file)

        return obj
    
    def save_lzma(self,
                  filepath: str,
                  hash: bool = True) -> None:
        '''
        Saves the SimAFM_Stack object to a lzma compressed pickle file.

        Args:
            filepath (str):
                The path to save the lzma compressed pickle file.
            hash (bool):
                Whether or not to save the sha256 hash of the file. Default is True.
        Returns:
            None       
        '''
        # Save the lzma compressed pickle file.
        with lzma.open(filepath, 'wb') as file:
            pickle.dump(self, file)
        
        # Save the hash of the file.
        if hash:
            hash_str = self._hash_file(filepath)
            with open(f'{filepath}.sha256', 'w') as file:
                file.write(hash_str)
        else:
            # Warn the user that hashing is disabled.
            logger.warning(f'Hashing disabled. {hash_warning}')
    
    @classmethod
    def load_lzma(cls,
                  filepath: str,
                  verify_hash: bool = True) -> 'SimAFM_Stack':
        '''
        Loads a SimAFM_Stack object from a lzma compressed pickle file. If verify_hash is True, the sha256 hash of the file will be checked 
        against the hash file with the same name as the lzma file and the extension .sha256. If the hashes do not match,
        a ValueError will be raised. If verify_hash is False, the hash will not be checked. Pickle objects are not
        robust against eroneous or maliciously constructed data. It is recommended that you never unpack data recieved
        from untrusted or unauthenticated sources. See https://docs.python.org/3/library/pickle.html.

        Args:
            filepath (str):
                The path to the lzma compressed pickle file.
            verify_hash (bool):
                Whether or not to verify the hash of the file. Default is True.
        Returns:
            SimAFM_Stack:
                The loaded SimAFM_Stack object.
        '''
        # Verify the hash of the file
        if verify_hash:
            hash_str = cls._hash_file(filepath)

            # Try to open the hash file. If it does not exist, raise a FileNotFoundError.
            try:
                with open(f'{filepath}.sha256', 'r') as file:
                    file_hash = file.read()
            except FileNotFoundError:
                logger.error(f'Hash file not found. Cannot verify the integrity of the lzma file. Verify the integrity of the file manually or disable hash verification. {hash_warning}')
                raise FileNotFoundError(f'Hash file not found. Cannot verify the integrity of the lzma file. Verify the integrity of the file manually or disable hash verification. {hash_warning}')
            
            # Check if the hash of the file matches the hash in the hash file. If not, raise a ValueError.
            if hash_str != file_hash:
                logger.error(f'Hash mismatch. {hash_warning}')
                raise ValueError(f'Hash mismatch. Cannot verify the integrity of the lzma file. Verify the integrity of the file manually or disable hash verification. {hash_warning}')
            
            # Load the lzma file.
            with lzma.open(filepath, 'rb') as file:
                obj = pickle.load(file)
        
        else:
            # Warn the user that hash verification is disabled.
            logger.warning(f'Hash verification disabled. {hash_warning}')
            
            # Load the lzma file.
            with lzma.open(filepath, 'rb') as file:
                obj = pickle.load(file)

        return obj
    
    ########################
    ##### Misc Methods #####
    ########################

    def copy(self) -> 'SimAFM_Stack':
        '''
        Returns a deep copy of the SimAFM_Stack object. This prevents the mutable attributes of the object from being modified.

        Returns:
            SimAFM_Stack:
                A deep copy of the SimAFM_Stack object.
        '''

        return copy.deepcopy(self)

    def shuffle(self) -> None:
        '''
        Shuffles the stack in place.
        '''
        # Generate a random permutation of the indexes and shuffle the stack.
        self._indexes = np.random.permutation(self._indexes)
        self._stack = self._stack[self._indexes]

        # Set the shuffled metadata attribute to True.
        self.add_metadata(shuffled = True)

    def unshuffle(self) -> None:
        '''
        Unshuffles the stack in place.
        '''
        # Sort the indexes and unshuffle the stack.
        self._indexes = np.argsort(self._indexes)
        self._stack = self._stack[self._indexes]

        # Set the shuffled metadata attribute to False.
        self.add_metadata(shuffled = False)

    ######################
    ##### Properties #####
    ######################

    @property
    def stack(self) -> np.ndarray:
        '''
        The simulated AFM image stack. numpy.ndarray with shape (n_images, n_pixels_x, n_pixels_y).
        '''
        return self._stack
    
    @property
    def shape(self) -> tuple:
        '''
        The shape of images in the image stack. (n_images, n_pixels_x, n_pixels_y)
        '''
        return self._shape
    
    @property
    def grid(self) -> np.ndarray:
        '''
        The grid of pixel coordinates used to generate the stack. numpy.ndarray with shape (n_pixels_x, n_pixels_y, 2).
        '''
        return self._grid
    
    @property
    def indexes(self) -> np.ndarray:
        '''
        The indexes of the stack. numpy.ndarray with shape (n_images).
        '''
        return self._indexes

    @property
    def x_boundaries(self) -> tuple:
        '''
        The boundaries of the x axis of the image stack. Units are in Angstroms. (x_min, x_max)
        '''
        return self._x_boundaries
    
    @property
    def y_boundaries(self) -> tuple:
        '''
        The boundaries of the y axis of the image stack. Units are in Angstroms. (y_min, y_max)
        '''
        return self._y_boundaries

    @property
    def boundaries(self) -> tuple:
        '''
        The boundaries of the image stack. Units are in Angstroms. ((x_min, x_max), (y_min, y_max))
        '''
        return self._boundaries
    
    @property
    def x_resolution(self) -> float:
        '''
        The resolution of the image stack along the x axis. Units are in Angstroms per pixel.
        '''
        return self._x_resolution
    
    @property
    def y_resolution(self) -> float:
        '''
        The resolution of the image stack along the y axis. Units are in Angstroms per pixel.
        '''
        return self._y_resolution
    
    @property
    def resolution(self) -> float:
        '''
        The resolution of the image stack. Units are in Angstroms per pixel.
        '''
        if self._resolution is None:
            logger.warning('Resolution is ambiguous. x_resolution and y_resolution are not equal.')
        return self._resolution
    
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
            self._metadata.pop(key, None)

    def get_metadata(self, key: str) -> Optional[Any]:
        '''
        Returns the value of a metadata attribute. If the key is not found, warns the user and returns None.

        Args:
            key (str):
                The key of the metadata attribute to return.
        Returns:
            Any:
                The value of the metadata attribute.
        '''

        if key not in self._metadata.keys():
            logger.warning(f'Metadata "{key}" not found. Returning None.')

        return self._metadata.get(key, None)

    def print_metadata(self) -> None:
        '''
        Method for printing all metadata pairs from the metadata attribute.
        '''
        for key, value in self._metadata.items():
            print(f'{key}: {value}')

def make_grid(boundaries: tuple,
              shape: tuple) -> np.ndarray:
    '''
    Generates a 2D grid of pixel coordinates.

    Args:
        boundaries (tuple):
            The boundaries of the grid. Units are in Angstroms. ((x_min, x_max), (y_min, y_max))
        shape (tuple):
            The shape of the grid. (n_pixels_x, n_pixels_y)
    Returns:
        numpy.ndarray:
            2D numpy.ndarray of pixel coordinates. Shape is (n_pixels_y, n_pixels_x, 2).
            Note: because of the way meshgrid works in combination with matplotlib, the shape is (n_pixels_y, n_pixels_x, 2).
    '''    
    # Unpack the boundaries
    (x_min, x_max), (y_min, y_max) = boundaries
    # Unpack the shape
    n_pixels_x, n_pixels_y = shape

    # Check if the resolution along the x axis is equal to the resolution along the y axis. If not, warn the user.
    if (x_max - x_min) / n_pixels_x != (y_max - y_min) / n_pixels_y:
        logger.warning('Resolution along the x axis is not equal to the resolution along the y axis. This may cause images to appear stretched and could cause unexpected behavior.')

    # Generate the grid
    x = np.linspace(x_min, x_max, n_pixels_x)
    y = np.linspace(y_min, y_max, n_pixels_y)

    # Return the grid.
    return np.array(np.meshgrid(x, y)).T

def make_radius_array(atom_group: MDA.AtomGroup,
                      radius_dict: VDW_Dict) -> np.ndarray:
    '''
    Creates a 1D numpy.ndarray of Van der Waals radii for each atom in an atom group.

    Args:
        atom_group (AtomGroup):
            MDAnalysis AtomGroup object. The atom group to create the radius array for.
        radius_dict (VDW_Dict):
            The Van der Waals radius dictionary to use.
    Returns:
        numpy.ndarray:
            1D numpy.ndarray of Van der Waals radii for each atom in the atom group.
    '''    
    # Map the atom types to the radius dictionary and return numpy array.
    return np.array([*map(radius_dict.get, atom_group.types)])

def simulate_AFM2D(atom_coords: np.ndarray,
                   atom_radii: np.ndarray,
                   grid: np.ndarray,
                   tip_radius: float,
                   tip_theta: float) -> np.ndarray:
    ''' 
    Generates a simulated AFM image given the coordinates of the atoms, the Van der Waals radii of the atoms, and the tip parameters.
    Calculation is vectorized for speed, but may be memory intensive for large grids. 

    Args:
        atom_coords (numpy.ndarray):
            2D numpy.ndarray with shape (n_atoms, 3). The coordinates of the atoms. Units are in Angstroms.
        atom_radii (numpy.ndarray):
            1D numpy.ndarray with shape (n_atoms). The Van der Waals radii of the atoms. Units are in Angstroms.
        grid (numpy.ndarray):
            grid of pixel coordinates. Shape is (n_pixels_x, n_pixels_y, 2). Units are in Angstroms.
        tip_radius (float):
            The radius of the spherical portion of the tip. Units are in Angstroms.
        tip_theta (float):
            The angle of the tip cone. Units are in degrees. Converted to radians internally.
    Returns:
        numpy.ndarray:
            2D numpy.ndarray of the simulated AFM image. Shape is (n_pixels_x, n_pixels_y). Height units are in Angstroms.
    '''
    # Check if the number of atom coordinates matches the number of atom radii
    if atom_coords.shape[0] != atom_radii.shape[0]:
        logger.error('The number of atom coordinates does not match the number of atom radii.')
        raise ValueError('The number of atom coordinates does not match the number of atom radii.')
    
    # Convert the tip_theta to radians
    tip_theta = np.deg2rad(tip_theta)

    # Unpack the atom coordinates and grid coordinates. Adjusting shapes to prepare for vectorized calculations.
    Sx, Sy, Sz = atom_coords.T
    Px, Py = grid[..., 0], grid[..., 1]
    Px, Py = Px[..., np.newaxis], Py[..., np.newaxis]

    # Calculate the distance between the tip centers and the atom coordinates
    f = np.sqrt((Px - Sx)**2 + (Py - Sy)**2)
    
    # Calculate the condition for contact to occur on the cone
    condition = f >= (atom_radii + tip_radius) * np.cos(tip_theta)

    # Calculate the contact height for the cone and the sphere
    cone_contact = Sz + atom_radii * (1/np.sin(tip_theta)) - f * (1/np.tan(tip_theta)) + (tip_radius * (1/np.sin(tip_theta) - 1))

    sphere_contact = np.sqrt(np.abs((atom_radii + tip_radius)**2 - f**2)) + Sz - tip_radius
    contact_heights = np.max(np.where(condition, cone_contact, sphere_contact), axis = -1)

    # Return the contact heights above the surface
    return contact_heights * (contact_heights > 0)

def simulate_AFM2D_stack(universe: MDA.Universe,
                         atom_selection: str,
                         bg_selection: str,
                         grid: np.ndarray,
                         tip_radius: float,
                         tip_theta: float,
                         vdw_dict: VDW_Dict) -> np.ndarray:
    '''
    Generates a stack of simulated AFM images in serial.

    Args:
        universe (MDAnalysis.Universe):
            The MDAnalysis Universe object.
        atom_selection (str):
            The atom selection string.
        bg_selection (str):
            The background selection string.
        grid (numpy.ndarray):
            The grid of pixel coordinates. Shape is (n_pixels_x, n_pixels_y, 2).
        tip_radius (float):
            The radius of the spherical portion of the tip. Units are in Angstroms.
        tip_theta (float):
            The angle of the tip cone. Units are in degrees. Converted to radians internally.
        vdw_dict (VDW_Dict):
            The Van der Waals radius dictionary.
    Returns:
        numpy.ndarray:
            The stack of simulated AFM images. Shape is (n_images, n_pixels_x, n_pixels_y)
    '''
    stack = np.empty((universe.trajectory.n_frames, *grid.shape[:2]))
    # Loop through the trajectory and create an image for each frame
    for traj_index, _ in enumerate(universe.trajectory):
        
        # Set the background height
        background = universe.select_atoms(bg_selection).center_of_geometry()[-1]

        # Select the protein atoms above the background and subtract the background
        prot_atoms = universe.select_atoms(f'{atom_selection} and prop z > {background}')
        prot_positions = prot_atoms.positions
        prot_positions[:,-1] = prot_positions[:,-1] - background

        # Create the radius array
        radius_array = make_radius_array(prot_atoms, vdw_dict)

        # Generate the AFM image and place it in the stack
        stack[traj_index] = simulate_AFM2D(prot_positions, radius_array, grid, tip_radius, tip_theta)

    return stack

def _thread_simulate_AFM2D_stack(args_tuple: tuple) -> np.ndarray:
    '''
    Thread function for multiprocessing Simulated AFM image generation. This function is called by simulate_AFM2D_stack_MP.

    Args:
        args_tuple (tuple):
            Tuple of arguments to pass to the thread function. The tuple should contain the following elements:
                coords_array (numpy.ndarray):
                    1D numpy.ndarray with shape (n_frames). The coordinates of the atoms for each frame. Units are in Angstroms.
                radii_array (numpy.ndarray):
                    1D numpy.ndarray with shape (n_frames). The Van der Waals radii of the atoms for each frame. Units are in Angstroms.
                grid (numpy.ndarray):
                    2D numpy.ndarray of pixel coordinates. Shape is (n_pixels_x, n_pixels_y, 2). Units are in Angstroms.
                tip_radius (float):
                    The radius of the spherical portion of the tip. Units are in Angstroms.
                tip_theta (float):
                    The angle of the tip cone. Units are in degrees. Converted to radians internally.
    Returns:
        numpy.ndarray:
            Simulated AFM image stack. Shape is (n_frames, n_pixels_x, n_pixels_y).
    '''
    # Unpack the arguments
    coords_array, radii_array, grid, tip_radius, tip_theta = args_tuple

    # Initialize the empty output stack
    output_stack = np.empty((len(coords_array), *grid.shape[:2]))

    # Loop over the trajectory and calculate the AFM image for each frame
    for index, (coords, radii) in enumerate(zip(coords_array, radii_array)):

        # Generate the AFM image
        image = simulate_AFM2D(coords, radii, grid, tip_radius, tip_theta)

        # Insert the image into the output stack
        output_stack[index] = image

    # Return the output stack
    return output_stack

def simulate_AFM2D_stack_MP(universe: MDA.Universe,
                            atom_selection: str,
                            bg_selection: str,
                            grid: np.ndarray,
                            tip_radius: float,
                            tip_theta: float,
                            vdw_dict: VDW_Dict,
                            n_procs = None) -> np.ndarray:
    '''
    Generates a stack of simulated AFM images using multiprocessing.

    Note: the simulateAFM2D is vectorized and already uses numpy's parallelization under the hood. The original
    implementation was written very poorly and required manual parallelization, wheras the current implementation is 
    probably better. Parallelization may only provide a speedup if your machine has a very high core count. My personal
    testing has shown that using more then 3 or 4 processes performs slightly worse than running in serial. 
    Proceed with caution.

    Args:
        universe (MDAnalysis.Universe):
            The MDAnalysis Universe object.
        atom_selection (str):
            The atom selection string.
        bg_selection (str):
            The background selection string.
        grid (numpy.ndarray):
            The grid of pixel coordinates. Shape is (n_pixels_x, n_pixels_y, 2).
        tip_radius (float):
            The radius of the spherical portion of the tip. Units are in Angstroms.
        tip_theta (float):
            The angle of the tip cone. Units are in degrees. Converted to radians internally.
        vdw_dict (VDW_Dict):
            The Van der Waals radius dictionary.
        n_procs (int):
            The number of processes to use. Default is None. If None, the number of processes will be determined by os.sched_getaffinity(0).
    Returns:
        numpy.ndarray:
            The stack of simulated AFM images. Shape is (n_images, n_pixels_x, n_pixels_y)
    '''
    # Determine number of processes to create. If n_procs is None, use os.sched_getaffinity(0) to determine the
    # number of available CPUs. This often returns the number of threads, not the number of physical cores and
    # may degrade performance.
    if n_procs is None:
        n_procs = len(os.sched_getaffinity(0))
        logger.warning(f'Number of processes not specified. Using {n_procs} processes. This may degrade performance. It is recommended to specify a number of processes <= the number of physical cores on the machine.')
    
    # Initialize empty coordinate arrays and radius arrays
    traj_len = universe.trajectory.n_frames
    traj_coords = np.empty(traj_len, dtype = object)
    traj_radii = np.empty(traj_len, dtype = object)

    # Loop through the trajectory and extract the atom coordinates and radii
    for traj_index, _ in enumerate(universe.trajectory):
        
        # Set the background height
        background = universe.select_atoms(bg_selection).center_of_geometry()[-1]

        # Select the protein atoms above the background and subtract the background
        prot_atoms = universe.select_atoms(f'{atom_selection} and prop z > {background}')
        prot_positions = prot_atoms.positions
        prot_positions[:,-1] = prot_positions[:,-1] - background

        # Create the radius array
        radius_array = make_radius_array(prot_atoms, vdw_dict)

        # Store the coordinates and radii
        traj_coords[traj_index] = prot_positions
        traj_radii[traj_index] = radius_array
    
    # Split the trajectory into chunks for multiprocessing
    splits = np.array_split(np.arange(0, traj_len), n_procs)
    split_coords = [traj_coords[split] for split in splits]
    split_radii = [traj_radii[split] for split in splits]

    # Create the argument tuples to be passed to each _thread_simulate_AFM2D_stack call
    args = tuple(zip(split_coords,
                     split_radii,
                     [grid]*n_procs,
                     [tip_radius]*n_procs,
                     [tip_theta]*n_procs))
    
    # Create the multiprocessing pool and generate the stack
    with mp.Pool(processes = n_procs) as pool:
        stack = np.concatenate(pool.map(_thread_simulate_AFM2D_stack, args))

    return stack