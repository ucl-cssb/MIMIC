import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OPENBLAS_MAIN_FREE"] = "1"

from mimic.utilities import *
from mimic.utilities.utilities import plot_CRM, plot_CRM_with_intervals

from mimic.model_infer.infer_CRM_bayes import *
from mimic.model_infer.infer_gompertz_bayes import *
from mimic.model_infer.infer_gompertz_bayes import plot_Gompertz_with_intervals
from mimic.model_infer import *
from mimic.model_simulate import *
from mimic.model_simulate.sim_CRM import *
from mimic.model_simulate.sim_Gompertz import *
from harmonic_mean import *

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import harmonic as hm

import arviz as az
import pymc as pm
import pytensor.tensor as at
import pickle
import cloudpickle

from scipy import stats
from scipy.stats import gaussian_kde
from scipy.integrate import odeint

import glob
import shutil

import sys
from datetime import datetime

print(f"\n{'='*70}", flush=True)
print(f"Script started at: {datetime.now()}", flush=True)
print(f"{'='*70}\n", flush=True)

###############################################
### Run Bayesian inference for the CRM batch model


###############################################
## Set paths

# Output folder 
output_folder = "Clare_EcN_Sent_EAB12glu_CL_CRM_1"  # Change this for different runs
# Create output directory
os.makedirs(output_folder, exist_ok=True)

# Data base directory
base_path = 'Clare_fitting_data2/'


###############################################
## Load data
EcN_Sent_EAB12glu = get_data(base_path + "co_EcN_Sent_EAB12glu_fitting.csv")

# write mean data to csv
num_species = EcN_Sent_EAB12glu.shape[1] - 1  
columns = ['time'] + [f'species_{i + 1}' for i in range(num_species)]

EcN_Sent_EAB12glu.columns = columns
EcN_Sent_EAB12glu.to_csv('EcN_Sent_EAB12glu.csv', index=False)


# Extract data and convert to numpy arrays
yobsdf = EcN_Sent_EAB12glu.iloc[:, [1, 2]]
yobs = yobsdf.to_numpy()
timesa = EcN_Sent_EAB12glu.iloc[:, 0]
times = timesa.to_numpy()



###############################################
## Set up model

num_species = 2
num_resources = 2 # number of carbon resources, here glucose and ethanolamine
num_secondary_resources = 1 # number of nitrogen resources, here ammonium


# Estimate reasonable resource initial conditions
y0_resources = np.array([1, 1]) 
y0_secondary_resources = np.array([1])  # Initial condition for ammonium




# Define prior parameters


prior_tau_mean = [0.7, 1.25]     
prior_tau_sigma = [0.2, 0.2]  
prior_w_mean = [0.9, 0.7, 0.5]       
prior_w_sigma = [0.15, 0.15, 0.15]
# prior_c_mean = [[1.2, 0.22], [0.65, 0.6]]  
# prior_c_sigma = [[0.3, 0.06], [0.15, 0.15]]
prior_c_mean = [[1.0, 0.22, 0.5], [0.95, 0.33, 0.5]]  
prior_c_sigma = [[0.15, 0.15, 0.15], [0.15, 0.15, 0.15]]
prior_m_mean = [0.2, 0.2] 
prior_m_sigma = [0.15, 0.15]
prior_alpha_mean = [[0.5, 0.75], [0.5, 0.5]]
prior_alpha_sigma = [[0.1, 0.1], [0.1, 0.1]]


# Sampling conditions
draws = 100
tune = 100
chains = 4
cores = 4


# Save model conditions to file
conditions_text = f"""Model Conditions and Priors
============================

Sampling Conditions:
- draws: {draws}
- tune: {tune}
- chains: {chains}
- cores: {cores}

Number of species: {num_species}
Number of resources: {num_resources}

Prior Parameters:
- tau: mean = {globals().get('prior_tau_mean', 'na')}, sigma = {globals().get('prior_tau_sigma', 'na')}
- w: mean = {globals().get('prior_w_mean', 'na')}, sigma = {globals().get('prior_w_sigma', 'na')}
- c: mean = {globals().get('prior_c_mean', 'na')}, sigma = {globals().get('prior_c_sigma', 'na')}
- m: mean = {globals().get('prior_m_mean', 'na')}, sigma = {globals().get('prior_m_sigma', 'na')}
- alpha: mean = {globals().get('prior_alpha_mean', 'na')}, sigma = {globals().get('prior_alpha_sigma', 'na')}

Notes:

first try

"""

with open(os.path.join(output_folder, 'model_conditions.txt'), 'w') as f:
    f.write(conditions_text)
print(f"Saved model conditions to {output_folder}/model_conditions.txt")


###############################################
## Run inference

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting CRM batch model...", flush=True)


inference = inferCRMbayes()

# adjust set_parameters to include either fixed parameters, as in tau=tau, or priors 
#   to infer them, as in prior_tau_mean=prior_tau_mean, prior_tau_sigma=prior_tau_sigma

inference.set_parameters(times=times, yobs=yobs, num_species=num_species, num_resources=num_resources,
                         num_secondary_resources=num_secondary_resources,
                         prior_tau_mean=prior_tau_mean, prior_tau_sigma=prior_tau_sigma,
                         prior_w_mean=prior_w_mean, prior_w_sigma=prior_w_sigma,
                         prior_m_mean=prior_m_mean, prior_m_sigma=prior_m_sigma,
                         prior_c_mean=prior_c_mean, prior_c_sigma=prior_c_sigma,
                         prior_alpha_mean=prior_alpha_mean, prior_alpha_sigma=prior_alpha_sigma,
                         draws=draws, tune=tune, chains=chains, cores=cores)

idata, idata_prior = inference.run_inference_CoLim_CRM()



# Save posterior samples to file
az.to_netcdf(idata, os.path.join(output_folder, 'model_posterior.nc'))

# Save prior samples to file
az.to_netcdf(idata_prior, os.path.join(output_folder, 'model_prior.nc'))

# To plot summary statistics of the posterior distributions
summary = az.summary(idata, var_names=["tau_hat", "w_hat","c_hat", "m_hat", "alpha_hat", "sigma", "y0_species"])
print("Summary Statistics:")
print(summary[["mean", "sd", "r_hat"]])

# Also save to text file
summary[["mean", "sd", "r_hat"]].to_csv(os.path.join(output_folder, 'summary_statistics.txt'), sep='\t')
print("Saved summary statistics to summary_statistics.txt")

print(f"[{datetime.now().strftime('%H:%M:%S')}] CoLim-CRM complete", flush=True)





###############################################
## Plot posterior distributions

# idata = az.from_netcdf('examples/CRM/Clare_EcN_Sent_EAB12glu_test_norm3/model_posterior.nc')

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting plotting ...", flush=True)

inference.plot_posterior(idata) # saves to wd as default

# Move all the generated posterior plot files to output folder
posterior_files = glob.glob("plot-posterior-*.pdf")
for file in posterior_files:
    shutil.move(file, os.path.join(output_folder, file))

print(f"Moved {len(posterior_files)} posterior plots to output folder")

az.plot_trace(idata, var_names=["tau_hat", "w_hat","c_hat", "m_hat", "alpha_hat", "sigma"])
#az.plot_trace(idata, var_names=["c_hat", "sigma"])
plt.savefig(os.path.join(output_folder, 'posterior-trace.jpg'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved posterior trace plot")


###############################################
## Create table of credible regions for species and resources for each parameter

# idata = az.from_netcdf('/home/cclare/projects/CRM/MIMIC/examples/CRM/Clare_EcN_Sent_EAB12glu_batch_norm/model_posterior.nc')

# idata_prior = az.from_netcdf('/home/cclare/projects/CRM/MIMIC/examples/CRM/Clare_EcN_Sent_EAB12glu_batch_norm/model_prior.nc')

# Extract posterior samples 
tau_posterior_samples = idata.posterior["tau_hat"].values
w_posterior_samples = idata.posterior["w_hat"].values
c_posterior_samples = idata.posterior["c_hat"].values 
m_posterior_samples = idata.posterior["m_hat"].values
alpha_posterior_samples = idata.posterior["alpha_hat"].values
y0_species_posterior = idata.posterior["y0_species"].values
sigma_posterior = idata.posterior["sigma"].values


# Create quantiles table for all parameters
quantiles_dict = {}

# List of all possible parameters to check
param_names = ["tau_hat", "w_hat", "c_hat", "m_hat", "alpha_hat", "sigma", "y0_species"]
# param_names = ["c_hat", "sigma"]

for param_name in param_names:
    if param_name in idata.posterior.data_vars:
        param_samples = idata.posterior[param_name].values
        param_shape = param_samples.shape[2:]  # Remove chain and draw dimensions
        
        if len(param_shape) == 0:  # Scalar parameter
            samples = param_samples.flatten()
            quantiles_dict[param_name] = {
                '0.025': np.percentile(samples, 2.5),
                '0.5': np.percentile(samples, 50),
                '0.975': np.percentile(samples, 97.5)
            }
        elif len(param_shape) == 1:  # 1D array
            for i in range(param_shape[0]):
                param_key = f'{param_name}[{i}]'
                samples = param_samples[:, :, i].flatten()
                quantiles_dict[param_key] = {
                    '0.025': np.percentile(samples, 2.5),
                    '0.5': np.percentile(samples, 50),
                    '0.975': np.percentile(samples, 97.5)
                }
        elif len(param_shape) == 2:  # 2D matrix
            for i in range(param_shape[0]):
                for j in range(param_shape[1]):
                    param_key = f'{param_name}[{i}, {j}]'
                    samples = param_samples[:, :, i, j].flatten()
                    quantiles_dict[param_key] = {
                        '0.025': np.percentile(samples, 2.5),
                        '0.5': np.percentile(samples, 50),
                        '0.975': np.percentile(samples, 97.5)
                    }

# Create DataFrame and save to text file
quantiles_df = pd.DataFrame(quantiles_dict).T
quantiles_df.to_csv(os.path.join(output_folder, 'parameter_credible_regions.txt'), sep='\t', float_format='%.4f')
print("Saved parameter quantiles to parameter_credible_regions.txt")
print(quantiles_df)


## Simulate model with observation noise for species (resources not observed)



n_samples = 2000
all_species_trajectories_with_noise = []
all_resource_trajectories = []
all_secondary_resource_trajectories = []

for i in range(n_samples):
    chain_idx = np.random.randint(0, c_posterior_samples.shape[0])
    draw_idx = np.random.randint(0, c_posterior_samples.shape[1])

    tau_sample = tau_posterior_samples[chain_idx, draw_idx]
    w_sample = w_posterior_samples[chain_idx, draw_idx]
    c_sample = c_posterior_samples[chain_idx, draw_idx]
    m_sample = m_posterior_samples[chain_idx, draw_idx]
    alpha_sample = alpha_posterior_samples[chain_idx, draw_idx]
    y0_species_sample = y0_species_posterior[chain_idx, draw_idx]
    sigma_sample = float(sigma_posterior[chain_idx, draw_idx])  
    
    sample_predictor = sim_CRM()
    sample_predictor.set_parameters(num_species=num_species,
                                   num_resources=num_resources,
                                   num_secondary_resources=num_secondary_resources,
                                   tau=tau_sample,
                                   w=w_sample,
                                   c=c_sample,
                                   m=m_sample,
                                   alpha=alpha_sample)

    
    y0_species_flat = np.squeeze(y0_species_sample).flatten()
    init_conditions = np.concatenate([y0_species_flat, y0_resources, y0_secondary_resources])
    
    sample_species, sample_resources, sample_secondary_resources = sample_predictor.simulate_CoLim_CRM(times, init_conditions)
    sample_species = np.squeeze(sample_species)
    sample_resources = np.squeeze(sample_resources)
    sample_secondary_resources = np.squeeze(sample_secondary_resources)
    
    # likelihood was: Y = pm.Lognormal("Y", mu=at.log(crm_curves), sigma=sigma, observed=...)
    species_with_noise = np.random.lognormal(mean=np.log(sample_species), sigma=sigma_sample)
    
    all_species_trajectories_with_noise.append(species_with_noise)
    all_resource_trajectories.append(sample_resources)
    all_secondary_resource_trajectories.append(sample_secondary_resources)

# Convert to arrays
all_species_trajectories_with_noise = np.array(all_species_trajectories_with_noise)  
all_resource_trajectories = np.array(all_resource_trajectories)  
all_secondary_resource_trajectories = np.array(all_secondary_resource_trajectories)

# Calculate percentiles for species - with observation noise
species_lower = np.percentile(all_species_trajectories_with_noise, 2.5, axis=0)
species_median = np.median(all_species_trajectories_with_noise, axis=0)
species_upper = np.percentile(all_species_trajectories_with_noise, 97.5, axis=0)

# Calculate percentiles for resources - no observation noise
resource_lower = np.percentile(all_resource_trajectories, 2.5, axis=0)
resource_median = np.median(all_resource_trajectories, axis=0)
resource_upper = np.percentile(all_resource_trajectories, 97.5, axis=0)

# Calculate percentiles for secondary resources - no observation noise
secondary_resource_lower = np.percentile(all_secondary_resource_trajectories, 2.5, axis=0)
secondary_resource_median = np.median(all_secondary_resource_trajectories, axis=0)
secondary_resource_upper = np.percentile(all_secondary_resource_trajectories, 97.5, axis=0)

# Plot
plot_CRM_with_intervals(species_median, resource_median,
                       species_lower, species_upper,
                       resource_lower, resource_upper,
                       times,
                       secondary_resource_median,
                       secondary_resource_lower, secondary_resource_upper,
                       'EcN_Sent_EAB12glu.csv')


plt.savefig(os.path.join(output_folder, 'CRM_with_confidence_intervals.jpg'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved CRM confidence intervals plot")
