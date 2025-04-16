#!/usr/bin/env bash

# This script prepares the conda environments for the project.

# Exit immediately if a command exits with a non-zero status
set -e

# Check if conda is installed and available
# This will only work if you've run conda init beforehand
eval "$(conda shell.bash hook)"

# Check the flags to see if CPU or GPU is requested
if [ -z "$1" ]; then
    echo "Must Specify: $0 [CPU|GPU]"
    exit 1
fi

# If the user specifies CPU, create the CPU environment
if [ "$1" == "CPU" ]; then
    echo "Creating CPU environment from environment_CPU.yml"

    # Create the CPU environment
    conda env create -f environment_CPU.yml

    # Automatically detect the environment name
    ENV_NAME=$(awk '/name:/{print $2}' environment_CPU.yml)

    # Activate the CPU environment
    conda activate "$ENV_NAME"

    # Install the CPU version of the probject.
    pip install .

# If the user specifies GPU, create the GPU environment
# Note: This assumes you have a compatible GPU and the necessary drivers installed
elif [ "$1" == "GPU" ]; then
    echo "Creating GPU environment from environment_GPU.yml"

    # Create the GPU environment
    conda env create -f environment_GPU.yml

    # Automatically detect the esnvironment name
    ENV_NAME=$(awk '/name:/{print $2}' environment_GPU.yml)

    # Activate the GPU environment
    conda activate "$ENV_NAME"

    # After the GPU env has been created and activated, add CUDNN_PATH to the environment
    python_minor=$(python -c "import sys, pathlib, importlib.util as iu; \
        spec = iu.find_spec('nvidia.cudnn'); \
        path = pathlib.Path(spec.origin).parent; \
        print(path)")

    # Add the CUDNN_PATH to the environment
    conda env config vars set \
        CUDNN_PATH="$python_minor" \
        LD_LIBRARY_PATH="$python_minor/lib" \
        TF_ENABLE_ONEDNN_OPTS=0\
        TF_CPP_MIN_LOG_LEVEL=2

    # Install the GPU version of the probject.
    pip install '.[GPU]'

    # Deactivate the environment
    conda deactivate

# If the user specifies something else, print an error message
else
    echo "Invalid argument: $1"
    echo "Use 'CPU' or 'GPU'."
    exit 1
fi

echo "Conda environment $ENV_NAME created."
echo "Activate with: conda activate $ENV_NAME"