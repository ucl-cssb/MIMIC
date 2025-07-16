from mimic.utilities import *
from mimic.utilities.utilities import plot_CRM, plot_CRM_with_intervals

from mimic.model_infer.infer_CRM_bayes import *
from mimic.model_infer import *
from mimic.model_simulate import *
from mimic.model_simulate.sim_CRM import *

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for background running
import matplotlib.pyplot as plt

import arviz as az
import pymc as pm
import pytensor.tensor as at
import pickle
import cloudpickle
import os

from scipy import stats
from scipy.integrate import odeint

import glob
import shutil

# Set the working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


## Load the parameters
with open("params-s2-r2.pkl", "rb") as f:
    params = pickle.load(f)
tau = params["tau"]
m = params["m"]
r = params["r"]
w = params["w"]
K = params["K"]
c = params["c"]

## read in the data
# When given both species and resource data

data = pd.read_csv("data-s2-r2.csv")

times = data.iloc[:, 0].values
yobs = data.iloc[:, 1:6].values

# When given only species data for the same system as above

# data = pd.read_csv("data-s2-infer-r2.csv")

# times = data.iloc[:, 0].values
# yobs = data.iloc[:, 1:3].values


# Output folder specification
output_folder = "s2_r2_inferC_1prior2"  # Change this for different runs
# Create output directory
os.makedirs(output_folder, exist_ok=True)


## Define the number of species and resources, and fixed parameters if necessary

num_species = 2
num_resources = 2

# fixed parameters
tau = params["tau"]
# c = params["c"]
m = params["m"]
r = params["r"]
w = params["w"]
K = params["K"]

# Define priors as necessary

# prior_tau_mean = 0.7
# prior_tau_sigma = 0.2

# prior_w_mean = 0.55
# prior_w_sigma = 0.2

prior_c_mean = [[0.2, 0.1], [0.1, 0.2]]  
prior_c_sigma = [[0.1, 0.1], [0.1, 0.1]]

# prior_m_mean = 0.25
# prior_m_sigma = 0.1

# prior_r_mean = 0.4
# prior_r_sigma = 0.1

# prior_K_mean = 5.5
# prior_K_sigma = 0.5


# Sampling conditions
draws = 50
tune = 50
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
- r: mean = {globals().get('prior_r_mean', 'na')}, sigma = {globals().get('prior_r_sigma', 'na')}
- K: mean = {globals().get('prior_K_mean', 'na')}, sigma = {globals().get('prior_K_sigma', 'na')}
"""

with open(os.path.join(output_folder, 'model_conditions.txt'), 'w') as f:
    f.write(conditions_text)
print(f"Saved model conditions to {output_folder}/model_conditions.txt")

# Run inference

inference = inferCRMbayes()

# adjust set_parameters to include either fixed parameters, as in tau=tau, or priors 
#   to infer them, as in prior_tau_mean=prior_tau_mean, prior_tau_sigma=prior_tau_sigma

inference.set_parameters(times=times, yobs=yobs, num_species=num_species, num_resources=num_resources,
                         tau=tau, w=w, m=m, r=r, K=K,
                         prior_c_mean=prior_c_mean, prior_c_sigma=prior_c_sigma,
                         draws=draws, tune=tune, chains=chains, cores=cores)

idata = inference.run_inference()

# To plot posterior distributions
inference.plot_posterior(idata, true_params=params) # saves to wd as default

# Move all the generated posterior plot files to output folder
posterior_files = glob.glob("plot-posterior-*.pdf")
for file in posterior_files:
    shutil.move(file, os.path.join(output_folder, file))

print(f"Moved {len(posterior_files)} posterior plots to output folder")


# To plot summary statistics of the posterior distributions, delete as appropriate
#summary = az.summary(idata, var_names=["tau_hat", "w_hat","c_hat", "m_hat", "r_hat", "K_hat", "sigma"])
summary = az.summary(idata, var_names=["c_hat", "sigma"])
print("Summary Statistics:")
print(summary[["mean", "sd", "r_hat"]])

# Also save to text file
summary[["mean", "sd", "r_hat"]].to_csv(os.path.join(output_folder, 'summary_statistics.txt'), sep='\t')
print("Saved summary statistics to summary_statistics.txt")

# Save posterior samples to file
az.to_netcdf(idata, os.path.join(output_folder, 'model_posterior.nc'))


#az.plot_trace(idata, var_names=["tau_hat", "w_hat","c_hat", "m_hat", "r_hat", "K_hat", "sigma"])
az.plot_trace(idata, var_names=["c_hat", "sigma"])
plt.savefig(os.path.join(output_folder, 'posterior-trace.jpg'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved posterior trace plot")


## Plot the CRM using median values from the posterior samples

init_species = 10 * np.ones(num_species+num_resources) 

# inferred parameters - adjust as appropriate
# tau_h = np.median(idata.posterior["tau_hat"].values, axis=(0,1))
# w_h = np.median(idata.posterior["w_hat"].values, axis=(0,1))
c_h = np.median(idata.posterior["c_hat"].values, axis=(0,1))
# m_h = np.median(idata.posterior["m_hat"].values, axis=(0,1))
# r_h = np.median(idata.posterior["r_hat"].values, axis=(0,1))
# K_h = np.median(idata.posterior["K_hat"].values, axis=(0,1))


# Individual parameter comparisons with specific filenames
#params_to_compare = [('tau', tau, tau_h), ('w', w, w_h), ('c', c, c_h), ('m', m, m_h), ('r', r, r_h), ('K', K, K_h)]
params_to_compare = [('c', c, c_h)]



for param_name, true_val, pred_val in params_to_compare:
    compare_params(**{param_name: (true_val, pred_val)})
    plt.savefig(os.path.join(output_folder, f'parameter_comparison_{param_name}.jpg'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {param_name} parameter comparison plot")

predictor = sim_CRM()

predictor.set_parameters(num_species = num_species,
                         num_resources = num_resources,
                         tau = tau,
                         w = w,
                         c = c_h,
                         m = m,
                         r = r,
                         K = K)

#predictor.print_parameters()

observed_species, observed_resources = predictor.simulate(times, init_species)
observed_data = np.hstack((observed_species, observed_resources))
 
# plot predicted species and resouce dynamics against observed data

plot_CRM(observed_species, observed_resources, times, 'data-s2-r2.csv')
plt.savefig(os.path.join(output_folder, 'CRM_prediction.jpg'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved CRM prediction plot")


## Plot CRM  with confidence intervals

# Get posterior samples for c_hat 
# tau_posterior_samples = idata.posterior["tau_hat"].values
# w_posterior_samples = idata.posterior["w_hat"].values
c_posterior_samples = idata.posterior["c_hat"].values 
# m_posterior_samples = idata.posterior["m_hat"].values
# r_posterior_samples = idata.posterior["r_hat"].values
# K_posterior_samples = idata.posterior["K_hat"].values


# Create quantiles table for all parameters
quantiles_dict = {}

# List of all possible parameters to check
# param_names = ["tau_hat", "w_hat", "c_hat", "m_hat", "r_hat", "K_hat", "sigma"]
param_names = ["c_hat", "sigma"]

for param_name in param_names:
    if param_name in idata.posterior.data_vars:
        param_samples = idata.posterior[param_name].values
        param_shape = param_samples.shape[2:]  # Remove chain and draw dimensions
        
        if len(param_shape) == 0:  # Scalar parameter
            samples = param_samples.flatten()
            quantiles_dict[param_name] = {
                '0.25': np.percentile(samples, 25),
                '0.5': np.percentile(samples, 50),
                '0.75': np.percentile(samples, 75)
            }
        elif len(param_shape) == 1:  # 1D array
            for i in range(param_shape[0]):
                param_key = f'{param_name}[{i}]'
                samples = param_samples[:, :, i].flatten()
                quantiles_dict[param_key] = {
                    '0.25': np.percentile(samples, 25),
                    '0.5': np.percentile(samples, 50),
                    '0.75': np.percentile(samples, 75)
                }
        elif len(param_shape) == 2:  # 2D matrix
            for i in range(param_shape[0]):
                for j in range(param_shape[1]):
                    param_key = f'{param_name}[{i}, {j}]'
                    samples = param_samples[:, :, i, j].flatten()
                    quantiles_dict[param_key] = {
                        '0.25': np.percentile(samples, 25),
                        '0.5': np.percentile(samples, 50),
                        '0.75': np.percentile(samples, 75)
                    }

# Create DataFrame and save to text file
quantiles_df = pd.DataFrame(quantiles_dict).T
quantiles_df.to_csv(os.path.join(output_folder, 'parameter_quantiles.txt'), sep='\t', float_format='%.4f')
print("Saved parameter quantiles to parameter_quantiles.txt")
print(quantiles_df)



lower_percentile = 2.5
upper_percentile = 97.5

n_samples = 50 # adjust as necessary
random_indices = np.random.choice(c_posterior_samples.shape[1], size=n_samples, replace=False)

# Store simulation results
all_species_trajectories = []
all_resource_trajectories = []

# Run simulations with different posterior samples
for i in range(n_samples):
    chain_idx = np.random.randint(0, c_posterior_samples.shape[0])
    draw_idx = np.random.randint(0, c_posterior_samples.shape[1])

    # tau_sample = tau_posterior_samples[chain_idx, draw_idx]
    # w_sample = w_posterior_samples[chain_idx, draw_idx]
    c_sample = c_posterior_samples[chain_idx, draw_idx]
    # m_sample = m_posterior_samples[chain_idx, draw_idx]
    # r_sample = r_posterior_samples[chain_idx, draw_idx]
    # K_sample = K_posterior_samples[chain_idx, draw_idx]
    
    sample_predictor = sim_CRM()
    sample_predictor.set_parameters(num_species=num_species,
                                   num_resources=num_resources,
                                   tau=tau,
                                   w=w,
                                   c=c_sample,
                                   m=m,
                                   r=r,
                                   K=K)

    sample_species, sample_resources = sample_predictor.simulate(times, init_species)
    
    # Store results
    all_species_trajectories.append(sample_species)
    all_resource_trajectories.append(sample_resources)


# Convert to numpy arrays
all_species_trajectories = np.array(all_species_trajectories)  
all_resource_trajectories = np.array(all_resource_trajectories)  

# Calculate percentiles across samples for each time point and species/resource
species_lower = np.percentile(all_species_trajectories, lower_percentile, axis=0)
species_median = np.median(all_species_trajectories, axis=0)
species_upper = np.percentile(all_species_trajectories, upper_percentile, axis=0)
resource_lower = np.percentile(all_resource_trajectories, lower_percentile, axis=0)
resource_median = np.median(all_resource_trajectories, axis=0)
resource_upper = np.percentile(all_resource_trajectories, upper_percentile, axis=0)

# plot the CRM with confidence intervals
plot_CRM_with_intervals(species_median, resource_median,
                       species_lower, species_upper,
                       resource_lower, resource_upper,
                       times, 'data-s2-r2.csv')
plt.savefig(os.path.join(output_folder, 'CRM_with_confidence_intervals.jpg'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved CRM confidence intervals plot")