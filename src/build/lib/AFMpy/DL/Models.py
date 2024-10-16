from typing import Tuple
from keras import layers, models

from AFMpy import Utilities

__all__ = ['CAE']

logger = Utilities.Logging.make_module_logger(__name__)

def CAE(input_shape: tuple,
        filter_shape: tuple = (3, 3),
        num_filters: int = 32,
        latent_dim: int = 1024) -> Tuple[models.Model, models.Model, models.Model]:
    '''
    Creates a Convoluational Autoencoder (CAE), with associated encoder and decoder models with the given parameters.

    Args:
        input_shape (tuple):
            The shape of the input images. (width, height, channels)
            Note: The input shape must be square and be a multiple of 2^n where n is the number of maxpooling layers.
                  The current implementation uses 2 maxpooling layers, so the input shape must be a multiple of 4.
        filter_shape (tuple):
            The shape of the convolutional filters. (width, height)
        num_filters (int):
            The number of filters to use in the convolutional layers.
        latent_dim (int):
            The dimension of the latent space.
    Returns:
        Tuple[models.Model, models.Model, models.Model]:
            The autoencoder, encoder, and decoder models.
            Often only the autoencoder and encoder are needed.
    '''

    # Check to see if this requrement is created before generating the model.
    if (input_shape[0] != input_shape[1]) or (input_shape[0] % 4 != 0):
        logger.error('The input shape must be square and a multiple of 2^n the number of maxpooling layers.')
        raise ValueError('The input shape must be square and a multiple of 2 ^ the number of maxpooling layers.')

    # Create the encoder
    encoder_input = layers.Input(shape = input_shape, name = 'encoder_input')
    x = layers.Conv2D(num_filters, filter_shape, activation = 'relu', padding = 'same')(encoder_input)
    x = layers.MaxPooling2D((2, 2), padding = 'same')(x)
    x = layers.Conv2D(num_filters, filter_shape, activation = 'relu', padding = 'same')(x)
    x = layers.MaxPooling2D((2, 2), padding = 'same')(x)

    # Save the shape before flattening. For reshaping later.
    shape_before_flattening = x.shape[1:]

    # Flatten the input to the latent space
    x = layers.Flatten()(x)
    latent = layers.Dense(latent_dim,activation = 'relu', name = 'latent')(x)

    # Encapsulate the encoder
    encoder = models.Model(encoder_input, latent, name = 'encoder')

    # Create the decoder
    decoder_input = layers.Input(shape = (latent_dim,), name = 'decoder_input')
    x = layers.Dense(shape_before_flattening[0] * shape_before_flattening[1] * shape_before_flattening[2])(decoder_input)
    x = layers.Reshape(shape_before_flattening)(x)
    x = layers.Conv2DTranspose(num_filters, filter_shape, activation = 'relu', padding = 'same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(num_filters, filter_shape, activation = 'relu', padding = 'same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoder_output = layers.Conv2DTranspose(input_shape[2], filter_shape, activation = 'sigmoid', padding = 'same')(x)

    # Encapsulate the decoder
    decoder = models.Model(decoder_input, decoder_output, name = 'decoder')

    # Create the autoencoder
    autoencoder = models.Model(encoder_input, decoder(encoder(encoder_input)), name = 'autoencoder')

    # Return the models
    return autoencoder, encoder, decoder