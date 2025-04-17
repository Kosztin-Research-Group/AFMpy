import logging
import tensorflow as tf

logger = logging.getLogger(__name__)

__all__ = ['is_gpu_available']

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