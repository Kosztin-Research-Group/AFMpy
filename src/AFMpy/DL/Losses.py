import tensorflow as tf

__all__ = ['ssim_loss']

def ssim_loss(y_true: tf.Tensor,
              y_pred: tf.Tensor) -> tf.Tensor:
    '''
    Calculates the Structural Similarity Index Measure (SSIM) loss between two images.

    Args:
        y_true (tf.Tensor):
            The ground truth image.

        y_pred (tf.Tensor):
            The predicted image.
    
    Returns:
        tf.Tensor:
            The SSIM loss between the two images.
    '''
    ssim = tf.image.ssim(y_true, y_pred, max_val = 1.0)
    return 1 - ssim