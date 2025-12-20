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
output_folder = "Clare_EcN_Sent_EAB12glu_batch_norm12"  # Change this for different runs
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
num_resources = 2


# Estimate reasonable resource initial conditions
y0_resources = np.array([1, 1]) 




# Define prior parameters


prior_tau_mean = [0.7, 1.25]     
prior_tau_sigma = [0.2, 0.2]  
prior_w_mean = [0.9, 0.7]       
prior_w_sigma = [0.15, 0.15]
# prior_c_mean = [[1.2, 0.22], [0.65, 0.6]]  
# prior_c_sigma = [[0.3, 0.06], [0.15, 0.15]]
prior_c_mean = [[1.0, 0.22], [0.95, 0.33]]  
prior_c_sigma = [[0.15, 0.15], [0.15, 0.15]]
prior_m_mean = [0.2, 0.2] 
prior_m_sigma = [0.15, 0.15]



# Sampling conditions
draws = 250
tune = 250
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

Notes:

works well, y0 upper = 0.15, but posteriors are pretty shifted from the full, maybe need to adjust priors for batch dynamics, will try y0_reources = 10 incase
low resources are slowing the dynamics down

decreased growth rate prior but mortality still 0.2
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
                         prior_tau_mean=prior_tau_mean, prior_tau_sigma=prior_tau_sigma,
                         prior_w_mean=prior_w_mean, prior_w_sigma=prior_w_sigma,
                         prior_m_mean=prior_m_mean, prior_m_sigma=prior_m_sigma,
                         prior_c_mean=prior_c_mean, prior_c_sigma=prior_c_sigma,
                         draws=draws, tune=tune, chains=chains, cores=cores)

idata, idata_prior = inference.run_inference_batch()



# Save posterior samples to file
az.to_netcdf(idata, os.path.join(output_folder, 'model_posterior.nc'))

# Save prior samples to file
az.to_netcdf(idata_prior, os.path.join(output_folder, 'model_prior.nc'))

# To plot summary statistics of the posterior distributions
summary = az.summary(idata, var_names=["tau_hat", "w_hat","c_hat", "m_hat", "sigma", "y0_species"])
print("Summary Statistics:")
print(summary[["mean", "sd", "r_hat"]])

# Also save to text file
summary[["mean", "sd", "r_hat"]].to_csv(os.path.join(output_folder, 'summary_statistics.txt'), sep='\t')
print("Saved summary statistics to summary_statistics.txt")

print(f"[{datetime.now().strftime('%H:%M:%S')}] CRM batch complete", flush=True)



###############################################
# Run SMC sampling to estimate marginal likelihood

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting SMC sampling...", flush=True)

with inference.model:
    idata_smc = pm.sample_smc(
        draws=2000,
        chains=4,
        cores=4,
        progressbar=True
    )

# Extract marginal likelihood
log_ml_array = idata_smc.sample_stats['log_marginal_likelihood'].values

print(f"Shape of log_ml_array: {log_ml_array.shape}")

# The array contains lists - extract the last value from each list
log_ml_list = []
for chain_data in log_ml_array.flat:  # Flatten to iterate over all elements
    if isinstance(chain_data, list):
        # Get the last non-NaN value from this chain
        valid_values = [x for x in chain_data if not np.isnan(x)]
        if valid_values:
            log_ml_list.append(valid_values[-1])  # Take the last valid value
    else:
        # If it's already a number, just append it
        if not np.isnan(chain_data):
            log_ml_list.append(chain_data)

# Convert to numpy array
log_ml = np.array(log_ml_list)

print(f"\nCRM Fbatch - Log ML per chain: {log_ml}")

if len(log_ml) > 0:
    print(f"CRM batch - Log ML: {np.mean(log_ml):.4f} ± {np.std(log_ml):.4f}")
    
    # Save
    np.save(os.path.join(output_folder, 'log_marginal_likelihood.npy'), log_ml)
    
    with open(os.path.join(output_folder, 'smc_results.txt'), 'w') as f:
        f.write(f"Log Marginal Likelihood (SMC)\n")
        f.write(f"="*40 + "\n")
        for i, val in enumerate(log_ml):
            f.write(f"Chain {i}: {val:.4f}\n")
        f.write(f"\nMean: {np.mean(log_ml):.4f}\n")
        f.write(f"Std: {np.std(log_ml):.4f}\n")
    
    print(f"Saved to {output_folder}")
else:
    print("ERROR: No valid log ML values found!")

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] SMC sampling complete", flush=True)



###############################################
## Plot posterior distributions



print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting plotting ...", flush=True)

inference.plot_posterior(idata) # saves to wd as default

# Move all the generated posterior plot files to output folder
posterior_files = glob.glob("plot-posterior-*.pdf")
for file in posterior_files:
    shutil.move(file, os.path.join(output_folder, file))

print(f"Moved {len(posterior_files)} posterior plots to output folder")

az.plot_trace(idata, var_names=["tau_hat", "w_hat","c_hat", "m_hat", "sigma", "y0_species"])
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
y0_species_posterior = idata.posterior["y0_species"].values
sigma_posterior = idata.posterior["sigma"].values


# Create quantiles table for all parameters
quantiles_dict = {}

# List of all possible parameters to check
param_names = ["tau_hat", "w_hat", "c_hat", "m_hat", "r_hat", "K_hat", "sigma", "y0_species"]
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

for i in range(n_samples):
    chain_idx = np.random.randint(0, c_posterior_samples.shape[0])
    draw_idx = np.random.randint(0, c_posterior_samples.shape[1])

    tau_sample = tau_posterior_samples[chain_idx, draw_idx]
    w_sample = w_posterior_samples[chain_idx, draw_idx]
    c_sample = c_posterior_samples[chain_idx, draw_idx]
    m_sample = m_posterior_samples[chain_idx, draw_idx]
    y0_species_sample = y0_species_posterior[chain_idx, draw_idx]
    sigma_sample = float(sigma_posterior[chain_idx, draw_idx])  
    
    sample_predictor = sim_CRM()
    sample_predictor.set_parameters(num_species=num_species,
                                   num_resources=num_resources,
                                   tau=tau_sample,
                                   w=w_sample,
                                   c=c_sample,
                                   m=m_sample)

    
    y0_species_flat = np.squeeze(y0_species_sample).flatten()
    init_conditions = np.concatenate([y0_species_flat, y0_resources])
    
    sample_species, sample_resources = sample_predictor.simulate_batch(times, init_conditions)
    sample_species = np.squeeze(sample_species)
    sample_resources = np.squeeze(sample_resources)
    
    # likelihood was: Y = pm.Lognormal("Y", mu=at.log(crm_curves), sigma=sigma, observed=...)
    species_with_noise = np.random.lognormal(mean=np.log(sample_species), sigma=sigma_sample)
    
    all_species_trajectories_with_noise.append(species_with_noise)
    all_resource_trajectories.append(sample_resources)

# Convert to arrays
all_species_trajectories_with_noise = np.array(all_species_trajectories_with_noise)  
all_resource_trajectories = np.array(all_resource_trajectories)  

# Calculate percentiles for species - with observation noise
species_lower = np.percentile(all_species_trajectories_with_noise, 2.5, axis=0)
species_median = np.median(all_species_trajectories_with_noise, axis=0)
species_upper = np.percentile(all_species_trajectories_with_noise, 97.5, axis=0)

# Calculate percentiles for resources - no observation noise
resource_lower = np.percentile(all_resource_trajectories, 2.5, axis=0)
resource_median = np.median(all_resource_trajectories, axis=0)
resource_upper = np.percentile(all_resource_trajectories, 97.5, axis=0)

# Plot
plot_CRM_with_intervals(species_median, resource_median,
                       species_lower, species_upper,
                       resource_lower, resource_upper,
                       times, 'EcN_Sent_EAB12glu.csv')


plt.savefig(os.path.join(output_folder, 'CRM_with_confidence_intervals.jpg'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved CRM confidence intervals plot")




###############################################
## Corner plot of prior and posterior distributions of each parameter


prior_means = {
    'tau_hat': prior_tau_mean,
    'w_hat': prior_w_mean,
    'c_hat': prior_c_mean,
    'm_hat': prior_m_mean,
}

prior_sigmas = {
    'tau_hat': prior_tau_sigma,
    'w_hat': prior_w_sigma,
    'c_hat': prior_c_sigma,
    'm_hat': prior_m_sigma,
}

param_vars = ["tau_hat", "w_hat", "c_hat", "m_hat"]
available_params = [param for param in param_vars if param in idata.posterior.data_vars]

# Extract individual parameter elements correctly
samples_dict = {}
for var_name in available_params:
    samples = idata.posterior[var_name].values  
    
    # Get parameter shape (remove chains and draws dimensions)
    param_shape = samples.shape[2:]
    
    if len(param_shape) == 0:  # Scalar parameter
        samples_dict[var_name] = samples.flatten()
    elif len(param_shape) == 1:  # 1D array
        for i in range(param_shape[0]):
            key = f"{var_name}[{i}]"
            samples_dict[key] = samples[:, :, i].flatten()
    elif len(param_shape) == 2:  # 2D array 
        for i in range(param_shape[0]):
            for j in range(param_shape[1]):
                key = f"{var_name}[{i},{j}]"
                samples_dict[key] = samples[:, :, i, j].flatten()

print(f"\nTotal extracted parameters: {len(samples_dict)}")
print(f"Parameter names: {list(samples_dict.keys())}")

# Convert to numpy array for corner plot
parameter_names = list(samples_dict.keys())
posterior_array = np.column_stack([samples_dict[name] for name in parameter_names])

inference = inferCRMbayes()
inference.plot_corner_topright(parameter_names, posterior_array, prior_means, prior_sigmas)

plt.savefig(os.path.join(output_folder, 'corner_plot_parameters_cwm_top.jpg'), dpi=300, bbox_inches='tight')
plt.close()



###############################################
## Calculate KL divergence between prior and posterior for each parameter


param_list = ['tau_hat', 'w_hat', 'c_hat', 'm_hat', 'sigma', "y0_species"]

all_results = {}

print("KL DIVERGENCE RESULTS (Posterior || Prior)")

for param_name in param_list:
    results = compute_kl_for_parameter(param_name, idata, idata_prior)
    
    if results is not None:
        all_results[param_name] = results
        
        print(f"\n{param_name}:")
        print(f"  Shape: {results['shape']}")
        
        if results['is_scalar']:
            kl = results['kl_values'][0]
            print(f"  KL divergence: {kl:.4f} nats ({kl/np.log(2):.4f} bits)")
        else:
            print(f"  Number of components: {len(results['kl_values'])}")
            print(f"  KL range: [{min(results['kl_values']):.4f}, {max(results['kl_values']):.4f}] nats")
            print(f"  KL mean: {np.mean(results['kl_values']):.4f} nats")
            print(f"  KL median: {np.median(results['kl_values']):.4f} nats")
            
            # Show individual components
            print(f"\n  Individual components:")
            for idx, kl in zip(results['indices'], results['kl_values']):
                info = "Low" if kl < 0.5 else ("Moderate" if kl < 2 else "High")
                print(f"    {param_name}{idx}: {kl:.4f} nats ({info} info gain)")


## Save KL divergence results to a text file

# Flatten all results into a DataFrame for easy analysis
summary_data = []

for param_name, results in all_results.items():
    for idx, kl in zip(results['indices'], results['kl_values']):
        summary_data.append({
            'Parameter': param_name,
            'Index': idx,
            'Full_name': f"{param_name}{idx}" if idx != 'scalar' else param_name,
            'KL_nats': kl,
            'KL_bits': kl / np.log(2),
            'Info_gain': 'Low' if kl < 0.5 else ('Moderate' if kl < 2 else 'High')
        })

df_summary = pd.DataFrame(summary_data)
df_summary = df_summary.sort_values('KL_nats', ascending=False)


# Save to CSV
df_summary.to_csv(os.path.join(output_folder, 'kl_divergence_summary.csv'), index=False)
print("\nFull results saved to 'kl_divergence_summary.csv'")


# Plot KL divergence results

fig, ax = plt.subplots(figsize=(14, 6))

# Group data by parameter
x_labels = []
x_positions = []
current_x = 0

for param_name, results in all_results.items():
    n_components = len(results['kl_values'])
    
    # Plot each component 
    for i, (idx, kl) in enumerate(zip(results['indices'], results['kl_values'])):
        ax.bar(current_x + i, kl, alpha=0.7, color='steelblue')
        
        # Label
        label = f"{param_name}{idx}" if idx != 'scalar' else param_name
        x_labels.append(label)
        x_positions.append(current_x + i)
    
    # Add separator between component groups so easier to read
    current_x += n_components + 0.5

ax.set_xlabel('Parameter', fontsize=12)
ax.set_ylabel('KL Divergence (nats)', fontsize=12)
ax.set_title('Information Gain by Individual Parameter Component', fontsize=14, fontweight='bold')
ax.set_xticks(x_positions)
ax.set_xticklabels(x_labels, rotation=90, ha='right', fontsize=9)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax.axhline(y=2.0, color='gray', linestyle='--', alpha=0.5)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'kl_comparison_by_parameter.pdf'), dpi=300, bbox_inches='tight')
plt.show()



###############################################
## Plot each parameter component with interval, and calculate and plot R*


fig, ax = plt.subplots(figsize=(15/2.54, 4/2.54))


# Extract parameters we want to plot
plot_params = ['tau_hat', 'w_hat', 'c_hat', 'm_hat']

# Define colors for each parameter group
param_colors = {
    'tau_hat': '#023E8AB3',  
    'w_hat': '#0077B6B3',    
    'c_hat': '#0096C7B3',   
    'm_hat': '#00B4D8B3'    
}

x_labels = []
x_positions = []
current_x = 0

for param_name in plot_params:
    # Get all components for this parameter from quantiles_df
    param_rows = [idx for idx in quantiles_df.index if idx.startswith(param_name)]
    
    for i, param_key in enumerate(param_rows):
        median_val = quantiles_df.loc[param_key, '0.5']
        lower_val = quantiles_df.loc[param_key, '0.025']
        upper_val = quantiles_df.loc[param_key, '0.975']
        
        # Plot bar for median with parameter-specific color
        ax.bar(current_x + i, median_val, alpha=1, 
               color=param_colors[param_name], edgecolor='black', linewidth=0.5)
        
        # Add error bars for credible interval
        error_lower = median_val - lower_val
        error_upper = upper_val - median_val
        ax.errorbar(current_x + i, median_val, 
                   yerr=[[error_lower], [error_upper]], 
                   fmt='none', color='black', capsize=3, capthick=0.5, linewidth=0.5)
        
        # Reformat labels
        # Convert "tau_hat[0]" to "tau_1"
        clean_label = param_key.replace('_hat', '').replace('[', '_').replace(', ', '').replace(']', '')
        # Convert 0-indexed to 1-indexed
        import re
        def increment_indices(match):
            return str(int(match.group(1)) + 1)
        clean_label = re.sub(r'_(\d)', increment_indices, clean_label)
        
        x_labels.append(clean_label)
        x_positions.append(current_x + i)
    
    # Add separator between parameter groups
    current_x += len(param_rows) + 0.5

# ax.set_xlabel('Parameter Component', fontsize=6)
# ax.set_ylabel('Parameter Value', fontsize=6)
# ax.set_title('Posterior Median and 95% Credible Intervals by Parameter Component', 
#              fontsize=14, fontweight='bold')
# ax.set_xticks(x_positions)
# ax.set_xticklabels(x_labels, rotation=90, ha='right', fontsize=6)

# To remove tick labels for formating plots
ax.set_xticklabels([])
ax.set_yticklabels([])

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'posterior_medians_credible_intervals.pdf'), 
            dpi=300, bbox_inches='tight')
plt.close()
print("Saved posterior medians and credible intervals plot")




# Extract median values from quantiles_df
m_0 = quantiles_df.loc['m_hat[0]', '0.5']
m_1 = quantiles_df.loc['m_hat[1]', '0.5']

c_00 = quantiles_df.loc['c_hat[0, 0]', '0.5']
c_01 = quantiles_df.loc['c_hat[0, 1]', '0.5']
c_10 = quantiles_df.loc['c_hat[1, 0]', '0.5']
c_11 = quantiles_df.loc['c_hat[1, 1]', '0.5']

w_0 = quantiles_df.loc['w_hat[0]', '0.5']
w_1 = quantiles_df.loc['w_hat[1]', '0.5']

# Calculate R* = mi / (cij * wj)
# Species 0 = EcN, Species 1 = Sent
# Resource 0 = glucose, Resource 1 = EA

R_star_EcN_glucose = m_0 / (c_00 * w_0)
R_star_EcN_EA = m_0 / (c_01 * w_1)
R_star_Sent_glucose = m_1 / (c_10 * w_0)
R_star_Sent_EA = m_1 / (c_11 * w_1)

# Create bar plot with E. coli (red) and Salmonella (cyan) colours
fig, ax = plt.subplots(figsize=(5.5/2.54, 4/2.54))

species_resource = ['rstar_EcN_EA', 'rstar_EcN_glu', 'rstar_Sent_EA', 'rstar_Sent_glu']
R_star_values = [R_star_EcN_EA, R_star_EcN_glucose, R_star_Sent_EA, R_star_Sent_glucose]

# E. coli Nissle (red/dark red) and Salmonella enterica (cyan/turquoise)
colors = ['#E05554', '#E05554', '#24B5CA', '#24B5CA']
ax.bar(species_resource, R_star_values, color=colors, edgecolor='black', linewidth=0.5)

# ax.set_xlabel('Species-Resource Pair', fontsize=6)
# ax.set_ylabel('R* (Resource Threshold)', fontsize=6)
# ax.set_title('Resource Threshold (R*) for Population Maintenance\nUsing Median Posterior Values', 
#              fontsize=14, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# To remove tick labels for formating plots
ax.set_xticklabels([])
ax.set_yticklabels([])

# Add value labels on bars
for i, (label, value) in enumerate(zip(species_resource, R_star_values)):
    ax.text(i, value + max(R_star_values)*0.02, f'{value:.4f}', 
            ha='center', va='bottom', fontsize=6, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'R_star_values.pdf'), 
            dpi=300, bbox_inches='tight')
plt.close()
print("Saved R* values plot")

# Save R* values to text file
r_star_df = pd.DataFrame({
    'Species_Resource': ['EcN_EA', 'EcN_glucose', 'Sent_EA', 'Sent_glucose'],
    'R_star': [R_star_EcN_EA, R_star_EcN_glucose, R_star_Sent_EA, R_star_Sent_glucose],
    'Species': ['EcN', 'EcN', 'Sent', 'Sent'],
    'Resource': ['EA', 'glucose', 'EA', 'glucose']
})
r_star_df.to_csv(os.path.join(output_folder, 'R_star_values.txt'), 
                 sep='\t', index=False, float_format='%.6f')
print("Saved R* values to R_star_values.txt")
print("\nR* Values:")
print(r_star_df)







print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Plotting complete", flush=True)



print(f"\n{'='*70}", flush=True)
print(f"All complete at: {datetime.now()}", flush=True)
print(f"{'='*70}\n", flush=True)
