import os
import logging
import multiprocessing as mp
import numpy as np

# Import MDAnalysis
import MDAnalysis as MDA

# Import the VDW_Dict class and the make_radius_array function from the Simulate_Common module.
from .Simulate_Common import VDW_Dict, make_radius_array

# Create a logger for this module
logger = logging.getLogger(__name__)

__all__ = ['simulate_AFM2D', 'simulate_AFM2D_stack', 'simulate_AFM2D_stack_MP']

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
                         memb_selection: str,
                         head_selection: str,
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
    # Logging that this calculation is being performed on CPU.
    logger.info('Simulating AFM images on single CPU with numpy.')

    stack = np.empty((universe.trajectory.n_frames, *grid.shape[:2]))
    # Loop through the trajectory and create an image for each frame
    for traj_index, _ in enumerate(universe.trajectory):

        # Logging the current index of the trajectory
        logger.debug(f'Simulating Frame {traj_index}.')
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
        
        logger.debug(f'Simulating Frame {index} on process {mp.current_process().name}')

        # Generate the AFM image
        image = simulate_AFM2D(coords, radii, grid, tip_radius, tip_theta)

        # Insert the image into the output stack
        output_stack[index] = image

    # Return the output stack
    return output_stack

def simulate_AFM2D_stack_MP(universe: MDA.Universe,
                            atom_selection: str,
                            memb_selection: str,
                            head_selection: str,
                            grid: np.ndarray,
                            tip_radius: float,
                            tip_theta: float,
                            vdw_dict: VDW_Dict,
                            n_procs = None) -> np.ndarray:
    '''
    Generates a stack of simulated AFM images using multiprocessing.

    Note: the simulateAFM2D is vectorized and already uses numpy's parallelization under the hood. The original
    implementation was written very poorly and required manual parallelization, wheras the current implementation is 
    notably better. Parallelization may only provide a speedup if your machine has a very high core count. Personal
    testing has shown that using more then 3 or 4 processes performs slightly worse than running in serial. 
    Proceed with caution.

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
    
    logger.info(f'Simulating AFM images on multiple CPU with numpy using {n_procs} processes.')

    # Initialize empty coordinate arrays and radius arrays
    traj_len = universe.trajectory.n_frames
    traj_coords = np.empty(traj_len, dtype = object)
    traj_radii = np.empty(traj_len, dtype = object)

    logger.debug('Looping through the trajectory and extracting atom coordinates and radii.')
    # Loop through the trajectory and extract the atom coordinates and radii
    for traj_index, _ in enumerate(universe.trajectory):
        
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

        # Store the coordinates and radii
        traj_coords[traj_index] = prot_positions
        traj_radii[traj_index] = radius_array
    
    logger.debug('Splitting the trajectory coords and radii into chunks for multiprocessing.')
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
    logger.debug('Creating multiprocessing pool.')
    with mp.Pool(processes = n_procs) as pool:
        stack = np.concatenate(pool.map(_thread_simulate_AFM2D_stack, args))

    return stack