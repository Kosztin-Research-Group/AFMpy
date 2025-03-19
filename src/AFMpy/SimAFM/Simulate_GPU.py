import logging
import numpy as np
import cupy as cp

# Import MDAnalysis
import MDAnalysis as MDA

# Import the VDW_Dict class and the make_radius_array function from the Simulate_Common module.
from .Simulate_Common import VDW_Dict, make_radius_array

# Create a logger for this module
logger = logging.getLogger(__name__)

__all__ = ['simulate_AFM2D', 'simulate_AFM2D_stack']

def simulate_AFM2D(atom_coords: np.ndarray,
                   atom_radii: np.ndarray,
                   grid: np.ndarray,
                   tip_radius: float,
                   tip_theta: float) -> np.ndarray:
    ''' 
    Generates a simulated AFM image given the coordinates of the atoms, the Van der Waals radii of the atoms, and the tip parameters.
    Calculation is vectorized for speed, and passed to the GPU with CuPy.

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
    # Convert inputs to CuPy arrays
    atom_coords = cp.asarray(atom_coords)
    atom_radii = cp.asarray(atom_radii)
    grid = cp.asarray(grid)
    
    # Check if the number of atom coordinates matches the number of atom radii
    if atom_coords.shape[0] != atom_radii.shape[0]:
        logger.error('The number of atom coordinates does not match the number of atom radii.')
        raise ValueError('The number of atom coordinates does not match the number of atom radii.')
    
    # Convert the tip_theta to radians
    tip_theta = cp.deg2rad(tip_theta)
    
    # Unpack the atom coordinates and grid coordinates
    Sx, Sy, Sz = atom_coords.T
    Px, Py = grid[..., 0], grid[..., 1]
    Px, Py = Px[..., cp.newaxis], Py[..., cp.newaxis]
    
    # Calculate the distance between the tip centers and the atom coordinates
    f = cp.sqrt((Px - Sx)**2 + (Py - Sy)**2)
    
    # Calculate the condition for contact to occur on the cone
    condition = f >= (atom_radii + tip_radius) * cp.cos(tip_theta)
    
    # Calculate the contact height for the cone and the sphere
    sin_theta = cp.sin(tip_theta)
    tan_theta = cp.tan(tip_theta)
    cone_contact = (Sz + atom_radii / sin_theta - f / tan_theta +
                    tip_radius * (1 / sin_theta - 1))
    sphere_contact = (cp.sqrt(cp.abs((atom_radii + tip_radius)**2 - f**2)) +
                      Sz - tip_radius)
    
    # Compute contact heights
    contact_heights = cp.max(cp.where(condition, cone_contact, sphere_contact), axis=-1)
    
    # Return the contact heights above the surface as a NumPy array
    return cp.asnumpy(contact_heights * (contact_heights > 0))

def simulate_AFM2D_stack(universe: MDA.Universe,
                         atom_selection: str,
                         memb_selection: str,
                         head_selection: str,
                         grid: np.ndarray,
                         tip_radius: float,
                         tip_theta: float,
                         vdw_dict: VDW_Dict) -> np.ndarray:
    '''
    Generates a stack of simulated AFM images in serial using GPU acceleration.

    Args:
        universe (MDAnalysis.Universe):
            The MDAnalysis Universe object.
        atom_selection (str):
            The atom selection string.
        memb_selection (str):
            The membrane selection string.
        head_selection (str):
            The headgroup selection string.
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
    # Logging that this calculation is being performed on GPU with CuPy.
    logger.info('Simulating AFM images on GPU with CuPy.')

    stack = np.empty((universe.trajectory.n_frames, *grid.shape[:2]))
    # Loop through the trajectory and create an image for each frame
    for traj_index, _ in enumerate(universe.trajectory):

        # Logging the current index of the trajectory
        logger.debug(f'Generating Frame {traj_index}')
        # Set the background height
        memb_center = universe.select_atoms(memb_selection).center_of_geometry()
        head_atoms = universe.select_atoms(f'{memb_selection} and {head_selection} and prop z > {memb_center[-1]}')
        background = head_atoms.center_of_geometry()[-1]

        # Select the protein atoms above the background and subtract the background
        prot_atoms = universe.select_atoms(f'{atom_selection} and prop z > {background}')
        prot_positions = prot_atoms.positions
        prot_positions[:,-1] = prot_positions[:,-1] - background

        # Create the radius array
        radius_array = make_radius_array(prot_atoms, vdw_dict)

        # Generate the AFM image and place it in the stack
        stack[traj_index] = simulate_AFM2D(prot_positions, radius_array, grid, tip_radius, tip_theta)

    logger.info('Successfully generated the AFM image stack.')
    return stack