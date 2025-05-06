# AFMpy Tutorials

## Table of Contents
0. [General](#0-general)
1. [Simulated Atomic Force Microscopy](#1-simulated-atomic-force-microscopy)
2. [Localization Atomic Force Microscopy](#2-localization-atomic-force-microscopy)
3. [Deep Spectral Clustering](#3-deep-spectral-clustering)
4. [Hierarchical Deep Spectral Clustering](#4-hierarchical-deep-spectral-clustering)
5. [Registration and Clustering](#5-registraion-and-clustering)
6. [Iterative Registration and Clustering](#6-iterative-registration-and-clustering)

## 0. General

The following tutorials are included to give a working understanding of the AFMpy project and its various features. Each tutorial reqiures data files including molecular dynamics trajectories, simulated atomic force microscopy image stacks, and cryptographic keys for verifying the integrity of the distributed data files. The files can be acquired by visiting the Microsoft OneDrive link [here](https://mailmissouri-my.sharepoint.com/:f:/g/personal/cmldnf_umsystem_edu/EsuvntDMQxtOh7tFG5jBG6MBC6wPozJ3Yj0i-Dng5YOnlw?e=Ytp9bL).

Download Tutorial-Files.zip by clicking Download at the top of the page. 

Create a folder called ```common``` in the ```tutorials/``` folder, and place the contents of ```Tutorial-Files.zip``` inside it. Your directory structure should look like this:

```bash
tutorials
├── DSC
│   └── *DSC Files
├── HierarchicalDSC
│   └── *HierarchicalDSC Files
├── IREC
│   └── *IREC Files
├── LAFM
│   └── *LAFM Files
├── SimAFM
│   └── *SimAFM Files
├── TUTORIAL_README.md
└── common
    ├── MD
    │   └── *MD Files
    ├── keys
    │   └── *Key Files
    └── stacks
        └── *Stack Files
```

Once the tutorial files are in the right place, use the miniconda terminal to activate the anaconda virtual environment you created during installation:

```bash
# If using the default CPU environment name
conda activate AFMpy-CPU

# If using the default GPU environment name
conda activate AFMpy-GPU

# Or, if you created a custom environment name
conda activate your_environment_name
```

Navigate to the tutorials directory in your miniconda terminal.

Start a jupyter server with one of the following commands:
```bash
# If you want to use Jupyter Notebook (Very Stable)
jupyter notebook

# If you want to use Jupyter Lab (Extra Features)
jupyter lab
```

This will load jupyter in your default browser. Use the jupyter interface to find navigate to a tutorial of your choosing and open the ```.ipynb``` file. A logical tutorial order is as follows:

1. Simulated Atomic Force Microscopy (SimAFM)
2. Localization Atomic Force Microscopy (LAFM)
3. Deep Spectral Clustering (DSC)
4. Hierarchical Deep Spectral Clustering (HierarchicalDSC)
5. Iterative Registration and Clustering (IREC)

## 1. Simulated Atomic Force Microscopy

**New Topics Covered**:

- Importing the AFMpy modules
- Configuring AFMpy logging
- Loading the AFMpy Matplotlib configuration
- Creating a Simulated AFM (SimAFM) stack from a molecular dyanmics trajectory
- Common metadata to add to Stack objects for bookeeping
- Creating a AFMpy Stack object from SimAFM images
- Using built in plotting functions to visualize AFM images
- Generating private/public crypotgraphic key pairs to digitally sign Stack object pickle files
- Saving AFMpy Stack objects to a pickle file

## 2. Localization Atomic Force Microscopy

**New Topics Covered**:

- Loading AFMpy Stack pickle files into an object instance
- Verifying pickle file integrity with public keys files and signatures
- Viewing the metadata of Stack objects
- Generating mean and LAFM images from Stack objects
- Using built in plotting functions to visualize mean AFM and LAFM images

## 3. Deep Spectral Clustering

**New Topics Covered**:

- Check GPU availablility with AFMpy
- Generate an instance of the AFMpy ConvolutionalAutoencoder object
- Configure compilation, fitting, and prediction parameters with included dataclasses
- Loading and saving pretrained model weights
- Apply Deep Spectral Clustering (DSC)
- Calculate the masked structrual similarity index measure (SSIM) between images

## 4. Hierarchical Deep Spectral Clustering

**New Topics Covered**:

- Apply Hierarchical Deep Spectral Clustering

## 5. Iterative Registration and Clustering

**New Topics Covered**:

- Apply Iterative Registration and Clustering