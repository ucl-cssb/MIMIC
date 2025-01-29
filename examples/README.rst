=========
Examples
=========

This directory contains subfolders with Jupyter notebooks (and related data files)
demonstrating different models and methods in MIMIC. Some of these examples use real-life
datasets published by Stein et al. and Herold et al.

- `CRM/ <CRM/>`_  

  - `examples-sim-CRM.ipynb <CRM/examples-sim-CRM.ipynb>`_: Simulate time-course data using the Cross-Feeding Resource Model.

- `gLV/ <gLV/>`_  

  - `examples-bayes-gLV.ipynb <gLV/examples-bayes-gLV.ipynb>`_: Bayesian inference for Generalized Lotka-Volterra.
  - `examples-lasso-gLV.ipynb <gLV/examples-lasso-gLV.ipynb>`_: Lasso-based inference for gLV.
  - `examples-ridge-gLV.ipynb <gLV/examples-ridge-gLV.ipynb>`_: Ridge-based inference for gLV.
  - `examples-Rutter-Dekker.ipynb <gLV/examples-Rutter-Dekker.ipynb>`_: Rutter-Dekker example.
  - `examples-sim-gLV.ipynb <gLV/examples-sim-gLV.ipynb>`_: Simulation of gLV dynamics.
  - `examples-Stein.ipynb <gLV/examples-Stein.ipynb>`_: Real-life dataset example (Stein et al.) applying gLV methods.
    *(Additional CSV files support these notebooks.)*

- `gMLV/ <gMLV/>`_

  **Generalized Metabolic Lotka-Volterra (gMLV)** is a variation of gLV that includes
  metabolite interactions alongside microbial abundances.

  - `examples-ridge-lasso-gMLV.ipynb <gMLV/examples-ridge-lasso-gMLV.ipynb>`_:
    Ridge/Lasso inference for gMLV.
  - `examples-sim-gMLV.ipynb <gMLV/examples-sim-gMLV.ipynb>`_: gMLV simulation examples.

- `GP/ <GP/>`_  

  - `examples-impute-GP.ipynb <GP/examples-impute-GP.ipynb>`_: Data imputation with Gaussian Processes.
  - `examples-impute-GP_Stein.ipynb <GP/examples-impute-GP_Stein.ipynb>`_: Extended GP-based imputation for Stein dataset.
    *(CSV files for input/output data are included here.)*

- `MultiModel/Herold/ <MultiModel/Herold/>`_

  These notebooks use real-life data from Herold et al. to demonstrate multi-model workflows.

  - `examples-Herold-sVAR.ipynb <MultiModel/Herold/examples-Herold-sVAR.ipynb>`_: sVAR approach on Herold dataset.
  - `examples-Herold-VAR.ipynb <MultiModel/Herold/examples-Herold-VAR.ipynb>`_: VAR approach on Herold dataset.
  - `examples_impute_data.ipynb <MultiModel/Herold/examples_impute_data.ipynb>`_: Data imputation for Herold multi-model workflows.
    *(`Source Data/` folder contains all raw files needed for these notebooks.)*

- `MVAR/ <MVAR/>`_  

  - `examples-infer-MVAR.ipynb <MVAR/examples-infer-MVAR.ipynb>`_: Inference with the Multivariate Autoregressive model.
  - `examples-sim-MVAR.ipynb <MVAR/examples-sim-MVAR.ipynb>`_: Simulation using MVAR.
    *(`parametersS.json` is included for these demos.)*

- `VAR/ <VAR/>`_  

  - `examples-bayes-VAR.ipynb <VAR/examples-bayes-VAR.ipynb>`_: Bayesian inference for Vector Autoregression.
  - `examples-sim-VAR.ipynb <VAR/examples-sim-VAR.ipynb>`_: Simulation examples for VAR.
    *(JSON files and CSV data support these notebooks.)*

- `run_gMLV_sims.py`: A script to run gMLV simulations from the command line.

Each subfolder includes one or more Jupyter notebooks that guide you step-by-step
through setup, simulation, and analysis with the MIMIC package. Feel free to explore
each subfolder for model-specific usage instructions, parameter files, and example data.
