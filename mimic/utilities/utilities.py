import json
import os

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cols = ["red", "green", "blue", "royalblue", "orange", "black"]


def plot_gLV(yobs, timepoints):
    # fig, axs = plt.subplots(1, 2, layout='constrained')
    fig, axs = plt.subplots(1, 1)
    for species_idx in range(yobs.shape[1]):
        axs.plot(timepoints, yobs[:, species_idx], color=cols[species_idx])
    axs.set_xlabel('time')
    axs.set_ylabel('[species]')


def plot_CRM(observed_species, observed_resources, timepoints, csv_file=None):
    # Create a single axis
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    cols = plt.cm.tab10.colors

    total_entities = observed_species.shape[1] + observed_resources.shape[1]

    # Plot each species
    for species_idx in range(observed_species.shape[1]):
        label = f'Species {species_idx + 1}'
        ax.plot(timepoints,
                observed_species[:,
                                 species_idx],
                color=cols[species_idx],
                label=label)

    # Plot each resource - using distinct colors that continue from where
    # species left off
    for resource_idx in range(observed_resources.shape[1]):
        # Use a different color index for resources (continuing from where
        # species left off)
        color_idx = observed_species.shape[1] + resource_idx

        color_idx = color_idx % len(cols)

        label = f'Resource {resource_idx + 1}'
        ax.plot(timepoints,
                observed_resources[:,
                                   resource_idx],
                linestyle='--',
                color=cols[color_idx],
                label=label)

    # If CSV file is provided, overlay the observed data
    if csv_file:
        import pandas as pd
        data = pd.read_csv(csv_file)
        # Extract time and data columns
        time_col = data.columns[0]  # Assuming first column is time

        # Plot observed species data with markers
        num_species = observed_species.shape[1]
        for i in range(num_species):
            species_col = f'species_{i+1}'
            if species_col in data.columns:
                ax.scatter(data[time_col],
                           data[species_col],
                           marker='o',
                           color=cols[i % len(cols)],
                           s=10,
                           alpha=0.7,
                           label=f'Observed {species_col}')

        # Plot observed resource data with different markers and consistent
        # colors with simulated resources
        num_resources = observed_resources.shape[1]
        for i in range(num_resources):
            resource_col = f'resource_{i+1}'
            if resource_col in data.columns:
                color_idx = num_species + i
                color_idx = color_idx % len(cols)
                ax.scatter(data[time_col], data[resource_col],
                           marker='s', color=cols[color_idx], s=10, alpha=0.7,
                           label=f'Observed {resource_col}')

    # Set axis labels
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Concentration', fontsize=12)
    ax.set_title('CRM Growth Curves: Simulated vs Observed', fontsize=14)

    # Add a legend to label both species and resources
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout to make room for the legend
    plt.tight_layout()

    # Show the plot
    plt.show()

    return fig, ax

def plot_CRM_with_intervals(observed_species, observed_resources, species_lower, species_upper, 
                           resource_lower, resource_upper, times, filename=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot median trajectories
    for i in range(observed_species.shape[1]):
        ax.plot(times, observed_species[:, i], label=f'Species {i+1}', linewidth=2)
    
    for i in range(observed_resources.shape[1]):
        ax.plot(times, observed_resources[:, i], label=f'Resource {i+1}', linewidth=2, linestyle='--')
    
    # Add confidence ribbons 
    for i in range(observed_species.shape[1]):
        ax.fill_between(times, species_lower[:, i], species_upper[:, i], 
                       alpha=0.2, color=plt.cm.tab10(i))
    
    for i in range(observed_resources.shape[1]):
        ax.fill_between(times, resource_lower[:, i], resource_upper[:, i], 
                       alpha=0.2, color=plt.cm.tab10(i + observed_species.shape[1]))
    
    if filename:
        true_data = pd.read_csv(filename)
        true_times = true_data['time'].values
        
        for i in range(observed_species.shape[1]):
            col_name = f'species_{i+1}'
            if col_name in true_data.columns:
                ax.scatter(true_times, true_data[col_name], 
                          marker='o', s=30, color=plt.cm.tab10(i), label=f'True {col_name}')
        
        for i in range(observed_resources.shape[1]):
            col_name = f'resource_{i+1}'
            if col_name in true_data.columns:
                ax.scatter(true_times, true_data[col_name], 
                          marker='s', s=30, color=plt.cm.tab10(i + observed_species.shape[1]), 
                          label=f'True {col_name}')
    
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Concentration', fontsize=14)
    ax.set_title('Consumer-Resource Model Dynamics with 95% Credible Intervals', fontsize=16)
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if filename:
        plt.savefig(f"{filename.split('.')[0]}_with_intervals.png", dpi=300)
    plt.show()


def plot_gMLV(yobs, sobs, timepoints):
    # fig, axs = plt.subplots(1, 2, layout='constrained')
    fig, axs = plt.subplots(1, 2)
    for species_idx in range(yobs.shape[1]):
        axs[0].plot(timepoints, yobs[:, species_idx], color=cols[species_idx])
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('[species]')
    if sobs.shape[1] > 0:
        for metabolite_idx in range(sobs.shape[1]):
            axs[1].plot(timepoints, sobs[:, metabolite_idx],
                        color=cols[metabolite_idx])
        axs[1].set_xlabel('time')
        axs[1].set_ylabel('[metabolite]')


def plot_fit_gMLV(yobs, yobs_h, sobs, sobs_h, timepoints):
    # plot the fit
    # fig, axs = plt.subplots(1, 2, layout='constrained')
    fig, axs = plt.subplots(1, 2)

    for species_idx in range(yobs.shape[1]):
        axs[0].plot(timepoints, yobs[:, species_idx], color=cols[species_idx])
        axs[0].plot(timepoints, yobs_h[:, species_idx],
                    '--', color=cols[species_idx])
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('[species]')

    for metabolite_idx in range(sobs.shape[1]):
        axs[1].plot(timepoints, sobs[:, metabolite_idx],
                    color=cols[metabolite_idx])
        axs[1].plot(timepoints, sobs_h[:, metabolite_idx],
                    '--', color=cols[metabolite_idx])
    axs[1].set_xlabel('time')
    axs[1].set_ylabel('[metabolite]')


def plot_fit_gLV(yobs, yobs_h, timepoints):
    # plot the fit
    # fig, axs = plt.subplots(1, 2, layout='constrained')
    fig, axs = plt.subplots(1, 1)

    for species_idx in range(yobs.shape[1]):
        axs.plot(timepoints, yobs[:, species_idx], color=cols[species_idx])
        axs.plot(timepoints, yobs_h[:, species_idx],
                 '--', color=cols[species_idx])
    axs.set_xlabel('time')
    axs.set_ylabel('[species]')


# def compare_params(mu=None, M=None, alpha=None, e=None):
#     # each argument is a tuple of true and predicted values (mu, mu_hat)
#     if mu is not None:
#         print("mu_hat/mu:")
#         print(np.array(mu[1]))
#         print(np.array(mu[0]))

#         fig, ax = plt.subplots()
#         ax.stem(np.arange(0, len(mu[0]), dtype="int32"),
#                 np.array(mu[1]), markerfmt="D", label='mu_hat', linefmt='C0-')
#         ax.stem(np.arange(0, len(mu[0]), dtype="int32"),
#                 np.array(mu[0]), markerfmt="X", label='mu', linefmt='C1-')
#         ax.set_xlabel('i')
#         ax.set_ylabel('mu[i]')
#         ax.legend()

#     if M is not None:
#         print("\nM_hat/M:")
#         print(np.round(np.array(M[1]), decimals=2))
#         print("\n", np.array(M[0]))

#         fig, ax = plt.subplots()
#         ax.stem(
#             np.arange(
#                 0,
#                 M[0].shape[0] ** 2),
#             np.array(
#                 M[1]).flatten(),
#             markerfmt="D",
#             label='M_hat',
#             linefmt='C0-')
#         ax.stem(
#             np.arange(
#                 0,
#                 M[0].shape[0] ** 2),
#             np.array(
#                 M[0]).flatten(),
#             markerfmt="X",
#             label='M',
#             linefmt='C1-')
#         ax.set_ylabel('M[i,j]')
#         ax.legend()

#     if alpha is not None:
#         print("\na_hat/a:")
#         print(np.round(np.array(alpha[1]), decimals=2))
#         print("\n", np.array(alpha[0]))

#         fig, ax = plt.subplots()
#         ax.stem(
#             np.arange(
#                 0,
#                 alpha[0].shape[0] *
#                 alpha[0].shape[1]),
#             np.array(
#                 alpha[1]).flatten(),
#             markerfmt="D",
#             label='a_hat',
#             linefmt='C0-')
#         ax.stem(
#             np.arange(
#                 0,
#                 alpha[0].shape[0] *
#                 alpha[0].shape[1]),
#             np.array(
#                 alpha[0]).flatten(),
#             markerfmt="X",
#             label='a',
#             linefmt='C1-')
#         ax.set_ylabel('a[i,j]')
#         ax.legend()

#     if e is not None:
#         print("\ne_hat/e:")
#         print(np.round(np.array(e[1]), decimals=2))
#         print("\n", np.array(e[0]))

#         fig, ax = plt.subplots()
#         ax.stem(np.arange(0, e[0].shape[0]), np.array(
#             e[1]).flatten(), markerfmt="D", label='e_hat', linefmt='C0-')
#         ax.stem(np.arange(0, e[0].shape[0]), np.array(
#             e[0]).flatten(), markerfmt="X", label='e', linefmt='C1-')
#         ax.set_ylabel('e[i]')
#         ax.legend()


def compare_params(**kwargs):
    """
    Compare inferred and observed parameters with any parameter name.

    Parameters:
    ----------
    **kwargs : Each argument should be a tuple of (true_value, predicted_value)
               where true_value and predicted_value are numpy arrays

    """
    import numpy as np
    import matplotlib.pyplot as plt

    for param_name, param_values in kwargs.items():
        true_val, pred_val = param_values

        # Print comparison
        print(f"\n{param_name}_hat/{param_name}:")
        print(np.round(np.array(pred_val), decimals=2))
        print("\n", np.array(true_val))

        # Create figure
        fig, ax = plt.subplots()

        # Handle different shapes of parameters
        true_array = np.array(true_val)
        pred_array = np.array(pred_val)

        # Determine x-axis size based on parameter shape
        if true_array.ndim > 1:
            x_size = true_array.size
            true_flat = true_array.flatten()
            pred_flat = pred_array.flatten()
        else:
            x_size = len(true_array)
            true_flat = true_array
            pred_flat = pred_array

        # Create stem plots
        ax.stem(
            np.arange(0, x_size, dtype="int32"),
            pred_flat,
            markerfmt="D",
            label=f'{param_name}_hat',
            linefmt='C0-'
        )
        ax.stem(
            np.arange(0, x_size, dtype="int32"),
            true_flat,
            markerfmt="X",
            label=f'{param_name}',
            linefmt='C1-'
        )

        # Set labels and legend
        ax.set_xlabel('index')
        ax.set_ylabel(
            f'{param_name}[i]' if true_array.ndim == 1 else f'{param_name}[i,j]')
        ax.legend()

        # Show plot
        plt.tight_layout()
        plt.show()


def set_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)


def read_parameters(json_file):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(current_dir, json_file)
    with open(file_path, 'r') as f:
        parameters = json.load(f)
    return parameters
