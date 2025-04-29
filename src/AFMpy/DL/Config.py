from dataclasses import dataclass, field
from typing import List, Any

import tensorflow as tf

from AFMpy.DL.Losses import combined_ssim_loss

__all__ = ['CompileConfig', 'FitConfig', 'PredictConfig']

@dataclass
class CompileConfig:
    '''
    Configuration for model compilation.
    '''
    optimizer: str | tf.keras.optimizers.Optimizer = 'adam'
    loss: str | tf.keras.losses.Loss = combined_ssim_loss
    compile_kwargs: dict[str, Any] = field(default_factory = dict)

@dataclass
class FitConfig:
    '''
    Configuration for model fitting.
    '''
    epochs: int = 25
    batch_size: int = 32
    verbose: int = 1
    callbacks: List[tf.keras.callbacks.Callback] = field(default_factory = list)
    fit_kwargs: dict[str, Any] = field(default_factory = dict)

@dataclass
class PredictConfig:
    '''
    Configuration for model prediction.
    '''
    batch_size: int = 32
    verbose: int = 1
    predict_kwargs: dict[str, Any] = field(default_factory = dict)