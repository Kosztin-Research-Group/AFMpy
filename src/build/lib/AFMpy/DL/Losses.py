import tensorflow as tf
import tensorflow.keras.backend as K

__all__ = ['ssim_loss', 'masked_ssim_loss', 'combined_ssim_loss', 'normalized_combined_ssim_loss']

# def ssim_loss(y_true: tf.Tensor,
#               y_pred: tf.Tensor) -> tf.Tensor:
#     '''
#     Calculates the Structural Similarity Index Measure (SSIM) loss between two images.

#     Args:
#         y_true (tf.Tensor):
#             The ground truth image.

#         y_pred (tf.Tensor):
#             The predicted image.
    
#     Returns:
#         tf.Tensor:
#             The SSIM loss between the two images.
#     '''
#     ssim = tf.image.ssim(y_true, y_pred, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03, verify_tensor_rank=False)
    
#     return 1 - ssim

def ssim_loss(y_true: tf.Tensor,
              y_pred: tf.Tensor) -> tf.Tensor:
    '''
    Custom SSIM loss function without internal assertions.
    '''
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    y_true = K.cast(y_true, tf.float32)
    y_pred = K.cast(y_pred, tf.float32)

    mu_true = K.mean(y_true)
    mu_pred = K.mean(y_pred)
    sigma_true = K.var(y_true)
    sigma_pred = K.var(y_pred)
    sigma_true_pred = K.mean((y_true - mu_true) * (y_pred - mu_pred))

    ssim_index = ((2 * mu_true * mu_pred + c1) * (2 * sigma_true_pred + c2)) / ((mu_true ** 2 + mu_pred ** 2 + c1) * (sigma_true + sigma_pred + c2))
    return 1 - ssim_index

def masked_ssim_loss(y_true: tf.Tensor,
                     y_pred: tf.Tensor,
                     threshold_rel = 0.05,
                     window_size=11,
                     sigma=1.5) -> tf.Tensor:
    
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    y_true = K.cast(y_true, tf.float32)
    y_pred = K.cast(y_pred, tf.float32)

    # Create Gaussian window
    def gaussian_kernel(size, sigma):
        coords = tf.range(size, dtype=tf.float32) - size // 2
        g = tf.exp(-0.5 * tf.square(coords) / tf.square(sigma))
        g = g / tf.reduce_sum(g)
        g = tf.expand_dims(g, axis=0) * tf.expand_dims(g, axis=1)
        return g

    kernel = gaussian_kernel(window_size, sigma)
    kernel = tf.expand_dims(tf.expand_dims(kernel, axis=-1), axis=-1)

    # Compute local means and variances
    mu_true = tf.nn.depthwise_conv2d(y_true, kernel, strides=[1, 1, 1, 1], padding='SAME')
    mu_pred = tf.nn.depthwise_conv2d(y_pred, kernel, strides=[1, 1, 1, 1], padding='SAME')
    mu_true_sq = mu_true ** 2
    mu_pred_sq = mu_pred ** 2
    mu_true_pred = mu_true * mu_pred

    sigma_true_sq = tf.nn.depthwise_conv2d(y_true ** 2, kernel, strides=[1, 1, 1, 1], padding='SAME') - mu_true_sq
    sigma_pred_sq = tf.nn.depthwise_conv2d(y_pred ** 2, kernel, strides=[1, 1, 1, 1], padding='SAME') - mu_pred_sq
    sigma_true_pred = tf.nn.depthwise_conv2d(y_true * y_pred, kernel, strides=[1, 1, 1, 1], padding='SAME') - mu_true_pred

    # Compute SSIM map
    ssim_map = ((2 * mu_true_pred + c1) * (2 * sigma_true_pred + c2)) / ((mu_true_sq + mu_pred_sq + c1) * (sigma_true_sq + sigma_pred_sq + c2))

    # Calculate threshold
    max_y_true = tf.reduce_max(y_true)
    threshold = threshold_rel * max_y_true

    # Create mask based on threshold
    mask = tf.logical_or(y_true > threshold, y_pred > threshold)

    # Apply mask to SSIM map
    masked_ssim = tf.boolean_mask(ssim_map, mask)

    # Compute average SSIM value in the masked region
    average_ssim = tf.reduce_mean(masked_ssim)

    return 1 - average_ssim

def combined_ssim_loss(y_true: tf.Tensor,
                       y_pred: tf.Tensor,
                       threshold_rel: float = 0.05,
                       alpha: float = 0.5) -> tf.Tensor:
    '''
    Combined SSIM loss function that considers both SSIM and masked SSIM.
    '''
    # Original SSIM loss
    ssim_loss_value = ssim_loss(y_true, y_pred)
    
    # Masked SSIM loss
    masked_ssim_loss_value = masked_ssim_loss(y_true, y_pred, threshold_rel)
    
    # Combined loss
    combined_loss = alpha * ssim_loss_value + (1 - alpha) * masked_ssim_loss_value
    
    return combined_loss

def normalized_combined_ssim_loss(y_true: tf.Tensor,
                                  y_pred: tf.Tensor,
                                  threshold_rel: float = 0.05,
                                  alpha: float = 0.5,
                                  beta: float = 0.5) -> tf.Tensor:
    
    ssim_loss_value = ssim_loss(y_true, y_pred)
    masked_ssim_loss_value = masked_ssim_loss(y_true, y_pred, threshold_rel)

    mean_ssim_loss = tf.reduce_mean(ssim_loss_value)
    std_ssim_loss = tf.math.reduce_std(ssim_loss_value)

    mean_masked_ssim_loss = tf.reduce_mean(masked_ssim_loss_value)
    std_masked_ssim_loss = tf.math.reduce_std(masked_ssim_loss_value)

    normalized_ssim_loss = (ssim_loss_value - mean_ssim_loss) / (std_ssim_loss + 1e-6)
    normalized_masked_ssim_loss = (masked_ssim_loss_value - mean_masked_ssim_loss) / (std_masked_ssim_loss + 1e-6)

    combined_loss = alpha * normalized_ssim_loss + beta * normalized_masked_ssim_loss

    return combined_loss
