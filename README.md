# AFMpy

## Table of Contents
- [Installation](#installation)
    - [Automatic Installation](#automatic-installation)
    - [Manual CPU Installation](#manual-cpu-installation)
    - [Manual GPU Installation](#manual-gpu-installation)
- [Usage](#usage)
- [License](#license)
- [Contact](#contact)


## Installation
AFMpy is distributed with both a CPU and GPU version. Please see the following sections outlining the dependencies and installation instructions for each version. It is highly reccomended to use virtual environments to manage your installation. Furthermore, anaconda compatible ```environment.yml``` files are included to simplify creating the environments. Installing Miniconda is outlined [here](https://docs.anaconda.com/miniconda/install/).

### Automatic installation 
**(Linux/Mac/WSL Only)**

An installation bash script is included in the src/ directory called ```install.sh```. It will automatically configure a conda virtual environment and install AFMpy. You can run it in your terminal by navigating to ```src/``` and executing one of the following commands.

```bash
# To install the CPU only version
bash install.sh CPU

# To install the GPU version
bash install.sh GPU
```

Once the installation script is completed, you can activate the conda environment with
```bash
# If you installed the CPU version
conda activate AFMpy-CPU

# Or if you installed the GPU version
conda activate AFMpy-GPU
```

and you can confirm installation of AFMpy with the following:
```bash
conda list AFMpy
```

which should return something that looks like:
```bash
# packages in environment at /path/to/your/miniconda3/envs/your_environment_name:
#
# Name                    Version                   Build  Channel
afmpy                     x.y.z                    pypi_0    pypi
```

If the previous test succeeded, you are now ready to use AFMpy!

### Manual CPU Installation

**(Any Operating System)**

The CPU version of AFMpy is included for those without a CUDA enabled GPU. Performance is severely degraded when compared to the GPU version, and is not reccomended if the resources are available.

Create the conda environment by navigating to the `AFMpy/src/` directory and running one of the following:

```bash
# If you wish to use the default environment name
conda env create -f environment_CPU.yml

# If you want to use a custom environment name
conda env create -f environment_CPU.yml -n your_environment_name
```

Activate the conda environment by running:

```bash
# If using the default environment name
conda activate AFMpy-CPU

# Or, if you created a custom environment name
conda activate your_environment_name
```

Install AFMpy by running:
```bash
pip install .
```

Confirm installation of AFMpy with the following:
```bash
conda list AFMpy
```

Which should return something that looks like
```bash
# packages in environment at /path/to/your/miniconda3/envs/your_environment_name:
#
# Name                    Version                   Build  Channel
afmpy                     x.y.z                    pypi_0    pypi
```

If the previous test succeeded, you are now ready to use AFMpy!

### Manual GPU Installation

**(Any Operating System\*)**

The GPU version of AFMpy is included for those with a CUDA enabled GPU. Its performance eclipses the CPU version's by an order of magnitude and is highly reccomended if the resources are available.

*Note: Setting up GPU compatibility for Tensorflow is often quite difficult. The software requirements for tensorflow often vary by operating system and personal configuration. See the tensorflow installation documentation to see if your operating system is compatible [here](https://www.tensorflow.org/install/pip). Additionally, this version of AFMpy was developed using CUDA12. CUDA11 has not been tested. Those wishing to use CUDA11 will need to prepare their environment from scratch and edit the setup.py for `cupy-cuda11x` accordingly.

*****

Create the conda environment by navigating to the `AFMpy/src/` directory and running one of the following:
```bash
# If you wish to use the default environment name
conda env create -f environment_GPU.yml

# If you want to use a custom environment name
conda env create -f environment_GPU.yml -n your_environment_name
```

Activate the conda environment by running:
```bash
# If using the default environment name
conda activate AFMpy-GPU

# Or, if you created a custom environment name
conda activate your_environment_name
```

Install AFMpy by running:
```bash
pip install '.[gpu]'
```

Confirm installation of AFMpy with the following:
```bash
conda list AFMpy
```

Which should return something that looks like:
```bash
# packages in environment at /path/to/your/miniconda3/envs/your_environment_name:
#
# Name                    Version                   Build  Channel
afmpy                     x.y.z                    pypi_0    pypi
```

Confirm that tensorflow can access your GPU:
```bash
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Which should return a list of the available GPUs. It will look something like this:
```bash
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

If the previous command returned an empty list, your GPU is not being properly detected. Additional troubleshooting for preparing your python installation for GPU can be found [here](https://www.tensorflow.org/install/pip).

If the previous two tests succeeded, you are now ready to use AFMpy!

## Usage
You can load AFMpy into a python script or jupyter notebook with the following:
```python
import AFMpy as AFM
```

A set of interactive tutorials are included in the ```tutorials/``` folder. See the ```TUTORIAL_README.md``` for downloading the tutorial files and running the tutorials.

## License
AFMpy - Atomic Force Microscopy simulation and analysis in python.

Copyright (C) 2024  Creighton M. Lisowski

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.

## Contact
For questions, feedback, or contributions, feel free to reach out:

**Creighton M. Lisowski**

Email: CLisowski@missouri.edu