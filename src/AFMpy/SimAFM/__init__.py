from __future__ import annotations
import importlib.util as _iu

# Import the Simulate_Common module. This uses only NumPy and is always available regardless of the back-end.
from . import Simulate_Common

# Import the Simulate_CPU module. Again, this uses only NumPy and is always available regardless of the back-end.
from . import Simulate_CPU
cpu = Simulate_CPU

# Try to import the Simulate_GPU module. This uses CuPy and is only available if CuPy is installed and a GPU is available.
_gpu = None
_cupy_spec = _iu.find_spec("cupy")
if _cupy_spec is not None:
    try:
        import cupy as _cp
        from . import Simulate_GPU
        # Check if a GPU is available. If so, set _gpu to Simulate_GPU.
        if _cp.cuda.runtime.getDeviceCount() > 0:
            _gpu = Simulate_GPU
        else:
            _gpu = None
    # If CuPy is not installed, the exception will be caught and _gpu will be set to None.
    except Exception:
        _gpu = None

# If the GPU is available, set the default backend to Simulate_GPU. Otherwise, set it to Simulate_CPU.
_default = Simulate_GPU if _gpu is not None else Simulate_CPU

# Add the functions from the Simulate_Common module to the global namespace.
for _name in Simulate_Common.__all__:
    globals()[_name] = getattr(Simulate_Common, _name)

# Add the functions from the default module to the global namespace.
for _name in _default.__all__:
    globals()[_name] = getattr(_default, _name)

# Add the functions from the Simulate_CPU module to the global namespace with the suffix "_cpu".
for _name in Simulate_CPU.__all__:
    globals()[f"{_name}_cpu"] = getattr(Simulate_CPU, _name)

# IF the GPU is available, add the functions from the Simulate_GPU module to the global namespace with the suffix "_gpu".
if _gpu is not None:
    for _name in Simulate_GPU.__all__:
        globals()[f"{_name}_gpu"] = getattr(Simulate_GPU, _name)

# Finally, define the __all__ variable to include all the functions in the modules
__all__ = list(_default.__all__)                       # default symbols
__all__ += [f"{n}_cpu" for n in Simulate_CPU.__all__]          # explicit CPU
if _gpu is not None:
    __all__ += [f"{n}_gpu" for n in Simulate_GPU.__all__]      # explicit GPU

# Clean up the namespace from the temporary variables
del _name, _default, _iu, _cupy_spec, cpu, _gpu
try:
    del _cp
except NameError:
    pass