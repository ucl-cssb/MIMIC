=========
Examples
=========

This directory contains subfolders with Jupyter notebooks demonstrating different
models and methods in MIMIC:

- **`CRM/ <CRM/>`_**: Cross-Feeding Resource Model examples.
  
  - `Simulate Time Course Data <CRM/examples-sim-CRM.ipynb>`_: Simulate time course data from the CRM.
  - `Five Species Model <CRM/five_species_model.ipynb>`_: Modeling dynamics of five species.

- **`gLV/ <gLV/>`_**: Generalized Lotka-Volterra model examples.
  
  - `gLV Simulation <gLV/examples-sim-gLV.ipynb>`_: Simulating interactions using the gLV model.
  - `Parameter Estimation <gLV/gLV_parameter_estimation.ipynb>`_: Estimating parameters in the gLV model.

- **`gMLV/ <gMLV/>`_**: Generalized Metabolic Lotka-Volterra model examples.
  
  - `Five Species, Six Metabolites, Single Time Course <gMLV/examples-sim-gMLV.ipynb#five-species-six-metabolites-single-time-course>`_: 
    Simulate a single time course with five species and six metabolites.
  - `Five Species with Perturbation <gMLV/examples-sim-gMLV.ipynb#five-species-with-perturbation>`_: 
    Simulating perturbations in a five-species system.

- **`GP/ <GP/>`_**: Gaussian Processes regression and data-imputation examples.
  
  - `Data Imputation <GP/data_imputation.ipynb>`_: Using Gaussian Processes for data imputation.
  - `Gaussian Process Regression <GP/gp_regression.ipynb>`_: Performing regression with Gaussian Processes.

- **`MultiModel/Herold/ <MultiModel/Herold/>`_**: Real-life dataset exploration and inference using multiple models and workflows.
  
  - `Herold Analysis <MultiModel/Herold/herold_analysis.ipynb>`_: Comprehensive analysis using Herold's multi-model approach.
  - `Source Data Exploration <MultiModel/Herold/source_data_exploration.ipynb>`_: Exploring and preparing source data.

Each folder contains one or more Jupyter notebooks that guide you step-by-step
through setup, simulation, and analysis using the MIMIC package. Feel free to
explore each subfolder for model-specific instructions and examples.
