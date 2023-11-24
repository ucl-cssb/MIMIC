# gMLV project

This is the code for the gMLV project. The project is to use gLV models to predict the dynamics of microbial communities. The code is written in Python 3.7. The code is organized as follows:
- gMLV: contains the code for the gLV model
- data: contains the data for the project
- clustering: contains the code for clustering the data


## gMLV

The gLV model is in the file gMLV/gLV.py. The file gMLV/gLV_sim.py contains the code for simulating the gLV model. The file gMLV/gLV_fit.py contains the code for fitting the gLV model to data. The file gMLV/gLV_utils.py contains utility functions for the gLV model.

## data

The data is in the file data/data.csv. The file data/data_utils.py contains utility functions for the data.

## clustering

The file clustering/clustering.py contains the code for clustering the data. The file clustering/clustering_utils.py contains utility functions for clustering.

## Running the code

The code can be run from the command line. The code for the gLV model can be run with the following command:

`python3 run_gLV.py <output directory> <number to simulate> <number of species> <number of time points> <number of replicates> <number of time points to fit> <number of replicates to fit> <number of time points to predict> <number of replicates to predict>`

for example:
`python run_gLV.py "C:\\Users\\User\\Desktop\\test_gLV" 100 10 100 10 50 5 50 5`

The code for clustering can be run with the following command:

`python3 run_clustering.py <output directory> <number to simulate> <number of species> <number of time points> <number of replicates> <number of time points to fit> <number of replicates to fit> <number of time points to predict> <number of replicates to predict>`

for example:
`python run_clustering.py "C:\\Users\\User\\Desktop\\test_clustering" 100 10 100 10 50 5 50 5`

The code for generating the gLV simulations can be run with the following command:

`python3 run_gLV_sims.py <output directory> <number to simulate>`

for example:
`python run_gLV_sims.py "C:\\Users\\User\\Desktop\\test_gLV" 100`


## Notes

The method for generating the data is from this paper (it is in the code data/data_utils.py):

https://onlinelibrary.wiley.com/doi/full/10.1002/bies.201600188


## Installation





########## Installing conda environment ##########
```
conda create -n gMLV python=3.10
conda activate gMLV
conda install tensorflow, etc.

```

Notes: no need to install 'casadi' or ODE.RED