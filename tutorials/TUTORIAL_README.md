# AFMpy Tutorials

## Table of Contents
- [General](#general)
1. [Simulated Atomic Force Microscopy](#1-simulated-atomic-force-microscopy)
2. [Localization Atomic Force Microscopy](#2-localization-atomic-force-microscopy)
3. [Deep Spectral Clustering](#3-deep-spectral-clustering)
4. [Hierarchical Deep Spectral Clustering](#4-hierarchical-deep-spectral-clustering)
5. [Registration and Clustering](#5-registraion-and-clustering)
6. [Iterative Registration and Clustering](#6-iterative-registration-and-clustering)

## General

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
4. Hierarchical Deep Spectral Clustering (HierarchicalDSC) *(WIP)*
5. Registration and Clustering (REC) *(WIP)*
6. Iterative Registration and Clustering (IREC) *(WIP)*

## 1. Simulated Atomic Force Microscopy

*Placeholder Text*

## 2. Localization Atomic Force Microscopy

*Placeholder Text*

## 3. Deep Spectral Clustering

*Placeholder Text*

## 4. Hierarchical Deep Spectral Clustering

*Placeholder Text*

## 5. Registration and Clustering

*Placeholder Text*

## 6. Iterative Registration and Clustering

*Placeholder Text*s