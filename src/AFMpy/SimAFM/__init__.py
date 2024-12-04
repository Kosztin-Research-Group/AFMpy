import importlib.util

# Import the common simulation tools.
from .Simulate_Common import *

# If cupy is installed, use the GPU version of the simulation tools. Otherwise, use the CPU version.
if importlib.util.find_spec('cupy') is not None:
    from .Simulate_GPU import *
else:
    from .Simulate_CPU import *