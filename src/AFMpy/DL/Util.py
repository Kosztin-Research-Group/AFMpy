import logging
import tensorflow as tf

logger = logging.getLogger(__name__)

__all__ = ['is_gpu_available', 'log_memory_usage']

def is_gpu_available() -> bool:
    '''
    Check if a GPU is available for TensorFlow.
    
    Args:
        None
    Returns:
        bool: True if a GPU is available, False otherwise.
    '''
    devices = tf.config.list_physical_devices('GPU')
    if len(devices) > 0:
        logger.info(f"GPUs available: {[device.name for device in devices]}")
        return True
    else:
        logger.info("No GPUs available.")
        return False
    
def log_memory_usage():
    '''
    Log the current memory usage of the current tensorflow session.
    
    Args:
        None
    Returns:
        None
    '''
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            dev_name = ':'.join(gpu.name.split(':')[1:])
            info = tf.config.experimental.get_memory_info(dev_name)
            current_mem = info["current"] / 1024**2
            peak_mem = info["peak"] / 1024**2
            logger.debug(f'Device {dev_name} - Current Memory: {current_mem:.0f} MiB, Peak Memory: {peak_mem:.0f} MiB')