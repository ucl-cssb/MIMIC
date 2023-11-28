
# gMLV Project

## Overview

The gMLV project utilizes generalized Lotka-Volterra (gLV) models to predict the dynamics of microbial communities. This repository contains Python code developed for modeling, simulating, and analyzing these dynamics. Our approach is detailed in [this paper](https://onlinelibrary.wiley.com/doi/full/10.1002/bies.201600188).

## Structure

The repository is organized into three main directories:

- `gMLV`: Core code for the gLV model.
    - `gLV.py`: Implementation of the gLV model.
    - `gLV_sim.py`: Simulation of the gLV model.
    - `gLV_fit.py`: Fitting the gLV model to data.
    - `gLV_utils.py`: Utility functions for the gLV model.
- `data`: Data and related utilities.
    - `data.csv`: Dataset for the project.
    - `data_utils.py`: Utility functions for handling data.
- `clustering`: Clustering algorithms and utilities.
    - `clustering.py`: Clustering implementation.
    - `clustering_utils.py`: Utility functions for clustering.

## Installation

### Prerequisites

- Python 3.10
- Conda package manager

### Steps

1. Clone the repository:
   ```bash
   git clone [repository URL]
   ```
2. Create and activate a conda environment:
   ```bash
   conda create -n gMLV python=3.10
   conda activate gMLV
   ```
3. Install dependencies:
   ```bash
   conda install tensorflow [other dependencies]
   ```

Note: 'casadi' and ODE.RED are not required.

## Usage

### Running gLV Model Simulations

```bash
python3 run_gLV.py <output directory> <number to simulate> <number of species> <number of time points> <number of replicates> <number of time points to fit> <number of replicates to fit> <number of time points to predict> <number of replicates to predict>
```

Example:
```bash
python run_gLV.py "C:\Users\User\Desktop\test_gLV" 100 10 100 10 50 5 50 5
```

### Running Clustering

```bash
python3 run_clustering.py [parameters]
```

Example:
```bash
python run_clustering.py "C:\Users\User\Desktop\test_clustering" 100 10 100 10 50 5 50 5
```

### Generating gLV Simulations

```bash
python3 run_gLV_sims.py <output directory> <number to simulate>
```

Example:
```bash
python run_gLV_sims.py "C:\Users\User\Desktop\test_gLV" 100
```

## Contributing

We welcome contributions to the gMLV project. Please refer to our [Contribution Guidelines](CONTRIBUTING.md) for more information.

## License

This project is licensed under the [LICENSE](LICENSE.md).

## Acknowledgements

This project is based on methods proposed in [this paper](https://onlinelibrary.wiley.com/doi/full/10.1002/bies.201600188).

## Contact

For questions or feedback, please [contact us](mailto:contact@example.com).
