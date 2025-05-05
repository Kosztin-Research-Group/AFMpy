from abc import ABC, abstractmethod
import logging
from typing import Tuple
import numpy as np
from keras import layers, models
import dataclasses

from AFMpy.DL import Config, Util

logger = logging.getLogger(__name__)

__all__ = ['ConvolutionalAutoencoder', 'DefaultCAE']

class ConvolutionalAutoencoder(ABC):
    '''
    Abstract base class for a Convolutional Autoencoder.
    Subclasses must implement `_build_models()` to create self._encoder, self._decoder, and self._autoencoder.
    '''
    def __init__(self,
                 compile_config: Config.CompileConfig | None = None,
                 fit_config: Config.FitConfig | None = None,
                 predict_config: Config.PredictConfig | None = None):
        
        # Initialize the model attributes
        self._encoder = None
        self._decoder = None
        self._autoencoder = None
        self._trained = False
        self._compiled = False
        
        # Initialize the compile, fit, and predict configurations
        self._compile_config = compile_config or Config.CompileConfig()
        self._fit_config = fit_config or Config.FitConfig()
        self._predict_config = predict_config or Config.PredictConfig()

        # Build the models
        self._build_models()
        
        # Cache the initial weights of the model
        self._cache_weights()

    @abstractmethod
    def _build_models(self):
        '''
        Build and set the encoder, decoder, and autoencoder models.

        Abstract method that must be implemented by subclasses.
        '''
        pass

    def _cache_weights(self):
        '''
        Caches the current state of the model weights and optimizer weights.

        This is useful for caching the initial state of the model to reset a trained model.
        '''
        self._init_weights = self.autoencoder.get_weights()


    def reset_weights(self) -> None:
        '''
        Reset the state of the Autoencoder to the initial weights.

        Args:
            None
        Returns:
            None
        '''
        if not hasattr(self, '_init_weights'):
            logger.error(f'Initial weights for ConvolutionalAutoencoder {id(self)} were never cached.')
            raise RuntimeError(f'Initial weights for ConvolutionalAutoencoder {id(self)} were never cached.')
        
        if not hasattr(self, '_init_optimizer_weights'):
            logger.error(f'Initial optimizer weights for ConvolutionalAutoencoder {id(self)} were never cached.')
            raise RuntimeError(f'Initial optimizer weights for ConvolutionalAutoencoder {id(self)} were never cached.')
        
        logger.debug(f'Resetting the weights for ConvolutionalAutoencoder {id(self)}.')

        # Reset the weights of the autoencoder
        self._autoencoder.set_weights(self._init_weights)

        # Reset the weights of the optimizer
        for var, val in zip(self._autoencoder.optimizer.variables,
                            self._init_optimizer_weights):
            var.assign(val)

        self._trained = False
        self._autoencoder.reset_metrics()
        Util.log_memory_usage()

    def compile(self, **override) -> None:
        '''
        Compile the autoencoder model. Must be called after _build_models().

        Args:
            **override:
                Additional keyword arguments to override the compile config.
        Returns:
            None
        '''
        if self._autoencoder is None:
            logger.error(f'The autoencoder model for ConvolutionalAutoencoder {id(self)} has not been created.')
            raise ValueError(f'The autoencoder model for ConvolutionalAutoencoder {id(self)} has not been created.')
        
        if self._compiled:
            logger.debug(f'ConvolutionalAutoencoder {id(self)} already compiled. Skipping...')
            return
        
        # Override the compile config with the provided arguments
        cfg = dataclasses.replace(self._compile_config, **override)
        
        logger.debug(f'Compiling the autoencoder model for ConvolutionalAutoencoder {id(self)}.')
        self._autoencoder.compile(optimizer = cfg.optimizer, loss = cfg.loss, **cfg.compile_kwargs)
        self._init_optimizer_weights = [var.numpy() for var in self._autoencoder.optimizer.variables]
        self._compiled = True
        Util.log_memory_usage()

    def fit(self, x, **override) -> None:
        '''
        Fit the autoencoder model. Must be called after compile().

        Args:
            x (numpy.ndarray):
                The input data to fit the model on.
            **override:
                Additional keyword arguments to override the fit config.
        Returns:
            None
        '''
        cfg = dataclasses.replace(self._fit_config, **override)

        # Check to see if the model has been trained before. If so, warn the user.
        if self._trained:
            logger.warning(f'Model {id(self)} has been trained previously. Resuming training.')
        logger.debug(f'Fitting the autoencoder model for ConvolutionalAutoencoder {id(self)}.')

        self._autoencoder.fit(x, x,
                              epochs = cfg.epochs,
                              batch_size = cfg.batch_size,
                              verbose = cfg.verbose,
                              callbacks = cfg.callbacks,
                              **cfg.fit_kwargs)

        # Set the trained flag to True
        self._trained = True

    def encode(self, x, **override) -> np.ndarray:
        '''
        Encode the input data.

        Args:
            x (numpy.ndarray):
                The input data to encode.
            **override:
                Additional keyword arguments to override the predict config.
        Returns:
            numpy.ndarray:
                The encoded data.
        '''
        cfg = dataclasses.replace(self._predict_config, **override)

        logger.debug(f'Encoding data with ConvolutionalAutoencoder {id(self)}.')
        return self._encoder.predict(x,
                                     batch_size = cfg.batch_size,
                                     verbose = cfg.verbose,
                                     **cfg.predict_kwargs)
    
    def decode(self, z, **override) -> np.ndarray:
        '''
        Decode the some encoded data.
        
        Args:
            z (numpy.ndarray):
                The encoded data to decode.
            **override:
                Additional keyword arguments to override the predict config.
        Returns:
            numpy.ndarray:
                The decoded data.
        '''
        cfg = dataclasses.replace(self._predict_config, **override)

        logger.debug(f'Decoding data with ConvolutionalAutoencoder {id(self)}.')
        return self._decoder.predict(z,
                                     batch_size = cfg.batch_size,
                                     verbose = cfg.verbose,
                                     **cfg.predict_kwargs)
    
    def reconstruct(self, x, **override) -> np.ndarray:
        '''
        Reconstruct the input data.

        Args:
            x (numpy.ndarray):
                The input data to reconstruct.
            **override:
                Additional keyword arguments to override the predict config.
        Returns:
            numpy.ndarray:
                The reconstructed data.
        '''
        cfg = dataclasses.replace(self._predict_config, **override)

        logger.debug(f'Reconstructing data with ConvolutionalAutoencoder {id(self)}.')
        return self._autoencoder.predict(x,
                                         batch_size = cfg.batch_size,
                                         verbose = cfg.verbose,
                                         **cfg.predict_kwargs)
    
    def save_weights(self, filepath: str, **kwargs) -> None:
        '''
        Save the weights of the autoencoder model. Weight file must end with the suffix .weights.h5 or an exception will be
        raised.

        Args:
            filepath (str):
                The filepath to save the weights to.
            **kwargs:
                Additional keyword arguments to pass to the save weights function.
        Returns:
            None
        '''
        logger.debug(f'Saving the weights of ConvolutionalAutoencoder {id(self)} to {filepath}.')
        self._autoencoder.save_weights(filepath, **kwargs)

    def load_weights(self, filepath: str, **kwargs) -> None:
        '''
        Load the weights of the autoencoder model. Sets the trained flag to True.

        Args:
            filepath (str):
                The filepath to load the weights from.
            **kwargs:
                Additional keyword arguments to pass to the load
        Returns:
            None
        '''
        logger.debug(f'Loading the weights from {filepath} to ConvolutionalAutoencoder {id(self)}.')
        self._autoencoder.load_weights(filepath, **kwargs)
        self._trained = True

    @property
    def encoder(self) -> models.Model:
        '''
        The encoder model. Available after calling _build_models().
        '''
        if self._encoder is None:
            logger.error(f'The encoder model for ConvolutionalAutoencoder {id(self)} has not been created.')
            raise ValueError(f'The encoder model for ConvolutionalAutoencoder {id(self)} has not been created.')
        return self._encoder
    
    @property
    def decoder(self) -> models.Model:
        '''
        The decoder model. Available after calling _build_models().
        '''
        if self._decoder is None:
            logger.error(f'The decoder model for ConvolutionalAutoencoder {id(self)} has not been created.')
            raise ValueError(f'The decoder model for ConvolutionalAutoencoder {id(self)} has not been created.')
        return self._decoder

    @property
    def autoencoder(self) -> models.Model:
        '''
        The autoencoder model. Available after calling _build_models().
        '''
        if self._autoencoder is None:
            logger.error(f'The autoencoder model for ConvolutionalAutoencoder {id(self)} has not been created.')
            raise ValueError(f'The autoencoder model for ConvolutionalAutoencoder {id(self)} has not been created.')
        return self._autoencoder
    
    @property
    def trained(self) -> bool:
        '''
        Whether the model has been trained or not.
        '''
        return self._trained
    
    @property
    def compiled(self) -> bool:
        '''
        Whether the model has been compiled or not.
        '''
        return self._compiled
    
class DefaultCAE(ConvolutionalAutoencoder):
    '''
    The default Convolutional Autoencoder model for the AFMpy project.
    '''
    def __init__(self,
                 input_shape: tuple,
                 filter_shape: tuple = (3, 3),
                 num_filters: int = 64,
                 latent_dim: int = 256,
                 **kwargs):
        '''
        Initialize the default Convolutional Autoencoder model.

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
            **kwargs:
                Additional keyword arguments to pass to the parent ConvolutionalAutoencoder parent class.
        '''
        self._input_shape = input_shape
        self._filter_shape = filter_shape
        self._num_filters = num_filters
        self._latent_dim = latent_dim

        super().__init__(**kwargs)

    def _build_models(self):
        # Check to see if this requrement is created before generating the model.
        if (self.input_shape[0] != self.input_shape[1]) or (self.input_shape[0] % 4 != 0):
            logger.error(f'The input shape {self.input_shape} must be square and a multiple of 2^n the number of maxpooling layers.')
            raise ValueError(f'The input shape {self.input_shape} must be square and a multiple of 2 ^ the number of maxpooling layers.')
        
        # Create the encoder
        logger.debug(f'Creating the encoder model for CAE {id(self)}')
        encoder_input = layers.Input(shape = self.input_shape, name = 'encoder_input')
        x = layers.Conv2D(self.num_filters, self.filter_shape, activation = 'relu', padding = 'same')(encoder_input)
        x = layers.MaxPooling2D((2, 2), padding = 'same')(x)
        x = layers.Conv2D(self.num_filters, self.filter_shape, activation = 'relu', padding = 'same')(x)
        x = layers.MaxPooling2D((2, 2), padding = 'same')(x)

        # Save the shape before flattening. For reshaping later.
        shape_before_flattening = x.shape[1:]

        # Flatten the input to the latent space
        x = layers.Flatten()(x)
        latent = layers.Dense(self.latent_dim,activation = 'relu', name = 'latent')(x)

        # Encapsulate the encoder
        encoder = models.Model(encoder_input, latent, name = 'encoder')

        # Create the decoder
        logger.debug(f'Creating the decoder model for CAE {id(self)}')
        decoder_input = layers.Input(shape = (self.latent_dim,), name = 'decoder_input')
        x = layers.Dense(shape_before_flattening[0] * shape_before_flattening[1] * shape_before_flattening[2])(decoder_input)
        x = layers.Reshape(shape_before_flattening)(x)
        x = layers.Conv2DTranspose(self.num_filters, self.filter_shape, activation = 'relu', padding = 'same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2DTranspose(self.num_filters, self.filter_shape, activation = 'relu', padding = 'same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        decoder_output = layers.Conv2DTranspose(self.input_shape[2], self.filter_shape, activation = 'sigmoid', padding = 'same')(x)

        # Encapsulate the decoder
        decoder = models.Model(decoder_input, decoder_output, name = 'decoder')

        # Create the autoencoder
        logger.debug(f'Combining the encoder and decoder models for CAE {id(self)}')
        autoencoder = models.Model(encoder_input, decoder(encoder(encoder_input)), name = 'autoencoder')
        
        # Set the models
        self._encoder = encoder
        self._decoder = decoder
        self._autoencoder = autoencoder
        self._cache_weights()
        
    @property
    def input_shape(self) -> Tuple:
        '''
        The input shape of the model.
        '''
        return self._input_shape
    
    @property
    def filter_shape(self) -> Tuple:
        '''
        The shape of the convolutional filters.
        '''
        return self._filter_shape
    
    @property
    def num_filters(self) -> int:
        '''
        The number of filters to use in the convolutional layers.
        '''
        return self._num_filters
    
    @property
    def latent_dim(self) -> int:
        '''
        The dimension of the latent space.
        '''
        return self._latent_dim