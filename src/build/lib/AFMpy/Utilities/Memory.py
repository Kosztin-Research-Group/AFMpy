import gc, psutil
import numpy as np

from tensorflow.keras import backend as K

from AFMpy.Utilities.Logging import make_module_logger

logger = make_module_logger(__name__)

__all__ = ['get_memory_usage', 'clear_variables', 'clear_variables_tensorflow']

unit_mapping = {
    'b': 0,
    'kb': 1,
    'mb': 2,
    'gb': 3,
}

def get_memory_usage(units: str = 'mb'):
    '''
    Get the current memory usage of the Python process in MB.

    Args:
        units (str):
            The units to return the memory usage in. Options are 'b', 'kb', 'mb', and 'gb'.
            Default is 'mb'.

    Returns (float):
        The memory usage of the Python process in MB.
    '''
    process = psutil.Process()

    return process.memory_info().rss / (1024 ** unit_mapping[units.lower()])

def clear_variables(*variables,
                    units: str = 'mb'):
    '''
    Clears the memory of the given variables.

    Args:
        *variables (list):
            The variables to clear from memory.
        units (str):
            The units to record the memory usage in. Options are 'b', 'kb', 'mb', and 'gb'.
            Default is 'mb'.

    Returns (None):
        None
    '''
    old_memory = get_memory_usage(units = units)

    logger.info('Clearing variables from memory...')
    logger.debug(f'Old memory usage: {old_memory:.2f} {units.upper()}')

    for variable in variables:
        del variable
    
    gc.collect()

    new_memory = get_memory_usage(units = units)
    logger.debug(f'New memory usage: {new_memory:.2f} {units.upper()}')
    logger.debug(f'Memory freed: {old_memory - new_memory:.2f} {units.upper()}')


def clear_variables_tensorflow(*variables,
                               units: str = 'mb'):
    '''
    Clears the memory of the given variables and resets the TensorFlow session.

    Args:
        *variables (list):
            The TensorFlow variables to clear from memory.
        units (str):
            The units to record the memory usage in. Options are 'b', 'kb', 'mb', and 'gb'.
            Default is 'mb'.

    Returns (None):
        None
    '''
    old_memory = get_memory_usage(units = units)

    logger.info('Clearing variables from memory and resetting the keras session...')
    logger.debug(f'Old memory usage: {old_memory:.2f} {units.upper()}')

    for variable in variables:
        del variable

    K.clear_session()
    gc.collect()

    new_memory = get_memory_usage(units = units)
    logger.debug(f'New memory usage: {new_memory:.2f} {units.upper()}')
    logger.debug(f'Memory freed: {old_memory - new_memory:.2f} {units.upper()}')
