# Physics-Informed Neural Network (PINN) examples

This directory collects several experiments exploring PINNs for the generalized Lotka–Volterra (gLV) model.  Development began with the simple example `model-v1.py` and gradually progressed to more sophisticated scripts and workflows.

## Chronology of the scripts

Below is an approximate order in which the different examples appeared in the repository together with the main additions introduced by each file.

### 1. `model-v1.py` and `models.py` (Nov 2024)
- **Origin**: first prototype contributed by Chris Barnes.
- **Purpose**: demonstrates a basic feed‑forward neural network approach to learn gLV dynamics from simulated data.  Parameters `mu` and `M` are part of the network (`dyn_model` in `models.py`).
- **Features**:
  - Generates synthetic data for three species.
  - Uses `dyn_model` and `loss_pinn` from `models.py` for training.
  - Produces diagnostic plots of the loss, fit, and parameter estimates.

### 2. `SBINN.py` → `gLV_inference/SBINN.py` (Nov 2024 → Jan 2025)
- **Initial commit**: `SBINN.py` (Nov 2024) implemented the *Sparse Bayesian INN* approach using DeepXDE.
- **Moved**: in Jan 2025 the script was moved to `gLV_inference/` and extended with plotting utilities.
- **Purpose**: infer `mu` and `M` for multi‑species gLV systems (up to six species).

### 3. `SBINNv2.py` (Jan 2025)
- **Adds**: a cleaned‑up version of the SBINN implementation.
- **Changes**:
  - Generates data via `sim_gLV`.
  - Trains a DeepXDE model and prints inferred parameters.

### 4. Perturbation experiments (Jan 2025)
Several variants explore gLV dynamics under external perturbations:
- **`SBINNwithPerturbations.py`** – trains with known perturbation functions; infers `mu`, `M` and perturbation coefficients `epsilon`.
- **`SBINNwithPerturbationsV2.py`** – similar but uses a step perturbation `u(t)` and more detailed visualisations of inferred parameters.
- **`SBINNwithPerturbationsV3.py`** – refines the TensorFlow implementation of the perturbation term and saves comparison plots for `epsilon`.

### 5. Results folders (Jan 2025)
- **`gLV_known_perturbation/`** – demonstration where the perturbation signal is known. Contains `sbinnglv_known_perturbations.py` and output plots.
- **`gLV_unkown_perturbation/`** – extends the previous script to also infer the unknown perturbation effects.
- **`gLV_inference/`** – holds `SBINN.py` outputs when inferring parameters for a six‑species system.

### 6. `Testing_correctness/` workflow (Feb 2025 → Jun 2025)
A collection of scripts to systematically test inference accuracy and hyper‑parameters.
Key components include:
- **`data_simulation.py` / `data_sim_usingMIMIC.py`** – generate synthetic datasets (multiple replicates) using either manual ODE code or the `sim_gLV` model.
- **`parameter_infer.py`** – main PINN inference routine with scaling of parameters, staged training, and optional L‑BFGS optimisation.
- **`original_parameter_infer.py`** – earlier reference version without stage splitting.
- **`hyperparameter_tuning_all.py`** – grid search over loss weights, regularisation and learning rate.
- **`adaptive_collation.py`** – experiments with adding new collocation points based on PDE residuals.
- **`function_constraints.py`** – explores softplus constraints on growth rates and negative self‑interaction.
- **`multi_stage_split.py`** and **`two_stage_split.py`** – compare different staged training schemes.
- **`compute_metrics.py`** – aggregates RMSE and relative error for inferred parameters.
- **`visualization_sim.py`** and **`visualization_reps.py`** – quick visual checks of the simulated replicates.
- **`ensembe_method.py`** – combines multiple inference runs into an ensemble estimate.

## Usage
These scripts are primarily exploratory.  Many produce plots or JSON summaries in their respective subfolders.  The `Testing_correctness/` suite can generate simulations, run parameter inference on each, and compute summary statistics.

