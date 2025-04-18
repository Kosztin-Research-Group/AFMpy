import logging
from abc import ABC, abstractmethod

import MDAnalysis as MDA
import numpy as np

# Create a logger for this module.
logger = logging.getLogger(__name__)

__all__ = ['VDW_Dict', 'AA_VDW_Dict', 'CG_VDW_Dict', 'make_grid', 'make_radius_array']

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
    logger.debug(f'Generating grid with boundaries: {boundaries} and shape: {shape}')
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