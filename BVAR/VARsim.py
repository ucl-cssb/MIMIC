import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import read_parameters
import os


class VARSimulator:
    def __init__(self, n_obs=100, coefficients=None, initial_values=None, noise_stddev=1.0, output='show'):
        if coefficients is None:
            coefficients = [[0.8, -0.2], [0.3, 0.5]]
        if initial_values is None:
            initial_values = [[1], [2]]
        self.n_obs = n_obs
        self.coefficients = np.array(coefficients)
        self.initial_values = np.array(initial_values)
        self.noise_stddev = noise_stddev
        self.output = output
        self.data = None
        self.dataM = None

    def print_parameters(self):
        for attr, value in vars(self).items():
            print(f"{attr}: {value}")

    # Save the data to a file, if the data is not None
    def save_data(self, filename):
        if self.data is not None:
            # Check if the filename ends with .csv, if not, append it
            if not filename.endswith('.csv'):
                filename += '.csv'

            # Save the data as a CSV file in the directory containing the script
            np.savetxt(filename, self.data, delimiter=",")
        else:
            print("No data to save.")

    def generate_var1_data(self):
        """
        Generate simulated data from a VAR(1) process.

        Parameters:
        - n_obs (int): Number of observations to generate.
        - coefficients (numpy.ndarray): Coefficient matrix for the VAR(1) process.
                                        It should be a square matrix of shape (dim, dim).
        - initial_values (numpy.ndarray): Initial values for the process, a vector of shape (dim,).
        - noise_stddev (float): Standard deviation of the normally distributed noise term.

        Returns:
        - data (numpy.ndarray): Simulated data matrix of shape (n_obs, dim).
        """
        dim = len(self.initial_values)
        data = np.zeros((self.n_obs, dim))
        data[0, :] = self.initial_values[:, 0]

        for t in range(1, self.n_obs):
            # VAR(1) process: X_t = A * X_{t-1} + noise
            noise = np.random.normal(scale=self.noise_stddev, size=dim)
            data[t, :] = np.dot(self.coefficients, data[t - 1, :]) + noise

        if self.output != 'none':
            make_plot_overlay(data, None, self.output)

        self.data = data  # the generated data
        return data

    def generate_mvar1_data(self, coefficientsM, initial_valuesM):
        """
        Generates synthetic data for a multivariate autoregressive (MVAR) process of order 1.

        Parameters:
        n_obs (int): The number of observations to generate.
        coefficients (numpy.ndarray): The coefficients of the VAR process for X. It should be a 2D array of shape (nX, nX).
        coefficientsM (numpy.ndarray): The coefficients of the process for S. It should be a 2D array of shape (nS, nX).
        initial_values (numpy.ndarray): The initial values of the X process. It should be a 2D array of shape (nX, 1).
        initial_valuesM (numpy.ndarray): The initial values of the S process. It should be a 2D array of shape (nS, 1).
        noise_stddev (float, optional): The standard deviation of the Gaussian noise. Defaults to 1.

        Returns:
        tuple: A tuple containing two numpy.ndarrays. The first array is the generated data for the X process, and the second array is the generated data for the S process. Both arrays have shape (n_obs, nX) and (n_obs, nS) respectively.
        """
        nX = len(self.initial_values)
        data = np.zeros((self.n_obs, nX))
        data[0, :] = self.initial_values[:, 0]

        coefficientsM = np.array(coefficientsM)
        initial_valuesM = np.array(initial_valuesM)
        nS = len(initial_valuesM)
        dataM = np.zeros((self.n_obs, nS))
        dataM[0, :] = initial_valuesM[:, 0]

        for t in range(1, self.n_obs):
            # VAR(1) process: X_t = A * X_{t-1} + noise
            noise = np.random.normal(scale=self.noise_stddev, size=nX)
            data[t, :] = np.dot(self.coefficients, data[t - 1, :]) + noise

        for t in range(1, self.n_obs):
            # process: S_t = B * X_{t-1} + noise
            noise = np.random.normal(scale=self.noise_stddev, size=(nS))

            Xt = data[t - 1, :].reshape((nX, 1))
            # print( "mult:", (coefficientsM @ Xt).shape )
            product = coefficientsM @ Xt
            dataM[t, :] = product[:, 0] + noise

        if self.output != 'none':
            make_plot_overlay(data, dataM, self.output)

        self.data, self.dataM = data, dataM  # the generated data
        return data, dataM

    def run(self, command, coefficientsM=None, initial_valuesM=None):
        if command == "VARsim":
            self.generate_var1_data()
        elif command == "MVARsim":
            if coefficientsM is None or initial_valuesM is None:
                raise ValueError(
                    "coefficientsM and initial_valuesM must be provided for MVARsim")
            self.generate_mvar1_data(coefficientsM, initial_valuesM)
        else:
            raise ValueError("Invalid command. Must be 'VARsim' or 'MVARsim'")


def make_plot_overlay(dataX, dataS=None, output='show'):
    """
    Creates an overlay plot of the given data.

    Parameters:
    dataX (numpy.ndarray): A 2D array containing the data for the X process. Each column represents a different variable, and each row represents a different time point.
    dataS (numpy.ndarray): A 2D array containing the data for the S process. Each column represents a different variable, and each row represents a different time point.

    Returns:
    None

    This function generates two overlay plots: one for the X process and one for the S process. Each variable in the process is plotted with a different color. The plots are saved in a PDF file named "plot-data-XS-overlay.pdf".
    """
    # Number of variables in dataX
    nX = dataX.shape[1]

    # Create a figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(
        10, 4)) if dataS is not None else plt.subplots(1, 1, figsize=(10, 4))
    axs = np.atleast_1d(axs)  # Ensure axs is always a list

    # Plot each variable in dataX on the first subplot
    for i in range(nX):
        axs[0].plot(dataX[:, i], label=f"X{str(i)}")
    axs[0].set_title("Abundance, X")

    # Plot each variable in dataS on the second subplot, if available
    if dataS is not None:
        nS = dataS.shape[1]
        for i in range(nS):
            axs[1].plot(dataS[:, i], label=f"S{str(i)}")
        axs[1].set_title("Abundance, S")

    # Adjust the layout to ensure that the subplots do not overlap
    plt.tight_layout()

    # If the output option is 'save' or 'both', save the figure as a PDF
    if output in ['save', 'both']:
        plt.savefig("plot-data-overlay.pdf")

    # If the output option is 'show' or 'both', show the figure in a new window
    if output in ['show', 'both']:
        plt.show()


def make_plot_stacked(dataX, dataS):
    """
    Creates a stacked plot and a heatmap for the given data.

    Parameters:
    dataX (numpy.ndarray): A 2D array containing the data for the X process. Each column represents a different variable, and each row represents a different time point.
    dataS (numpy.ndarray): A 2D array containing the data for the S process. Each column represents a different variable, and each row represents a different time point.

    Returns:
    None

    This function generates a stacked plot for the X process and a heatmap for the S process. The stacked plot shows the abundance of each variable in the X process over time, and the heatmap shows the values of each variable in the S process over time. The function does not return any value; instead, it saves the plots in a PDF file named "plot-data-XS-stacked.pdf".
    """

    dataX = dataX + 1.0

    #  stacked
    nX = len(dataX[0])  # Number of columns in dataX
    nS = len(dataS[0])  # Number of columns in dataS
    nobs = dataS.shape[0]

    # Create a figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 4))
    # Stack plot for dataX
    axs[0].stackplot(
        range(len(dataX)), *dataX.T, labels=[f"X{str(i)}" for i in range(nX)]
    )
    axs[0].set_title("Abundance, log10 X")
    axs[0].set_ylabel("X")
    axs[0].set_xlim(0, nobs-1)

    sns.heatmap(
        dataS.T,
        annot=False,
        cmap="YlGnBu",
        yticklabels=[f"S{str(i)}" for i in range(nS)],
        ax=axs[1],
        cbar=False,
    )
    axs[1].set_title("S")
    axs[1].set_ylabel("S")
    axs[1].set_xlabel("time (weeks)")
    axs[1].set_xlim(0, nobs)

    plt.tight_layout()  # Adjust the layout
    plt.savefig("plot-data-XS-stacked.pdf")


def make_plot(dataX, dataS=None, output='show'):
    """
    Creates separate line plots for each variable in the given data.

    Parameters:
    dataX (numpy.ndarray): A 2D array containing the data for the X process. Each column represents a different variable, and each row represents a different time point.
    dataS (numpy.ndarray): A 2D array containing the data for the S process. Each column represents a different variable, and each row represents a different time point.

    Returns:
    None

    This function generates a separate line plot for each variable in the X and S processes. The plots are arranged vertically, with the plots for the X process at the top and the plots for the S process at the bottom. Each plot shows the values of the variable over time. The function does not return any value; instead, it saves the plots in a PDF file named "plot-data-XS.pdf".
    """
    nX = len(dataX[0])  # Number of columns in dataX

    # Number of columns in dataS if it is provided
    nS = len(dataS[0]) if dataS is not None else 0

    fig, axs = plt.subplots(nX + nS, 1, figsize=(10, 2*(nX+nS)))

    # Adjust the vertical spacing between subplots
    plt.subplots_adjust(hspace=0.5)

    for i, ax in enumerate(axs):
        if i < nX:
            axs[i].plot(dataX[:, i])
            axs[i].set_title(f"X{str(i)}")
        elif dataS is not None:
            axs[i].plot(dataS[:, i-nX])
            axs[i].set_title(f"S{str(i - nX)}")

        # Set the y-axis label
        axs[i].set_ylabel('Abundance')

    if output in ['save', 'both']:
        plt.savefig("plot-data-stacked.pdf")
    if output in ['show', 'both']:
        plt.show()


if __name__ == "__main__":

    # FIXME: remember to delete this part before deploying
    parametersX = read_parameters('parameters2.json')
    simulator = VARSimulator(**parametersX)
    simulator.run("VARsim")

    make_plot(simulator.data)

    parametersS = read_parameters('parametersS.json')

    simulator.run("MVARsim", **parametersS)

    # make_plot_stacked(simulator.data)

    # simulator = VARSimulator(n_obs=100, coefficients=[
    # [0.8, -0.2], [0.3, 0.5]], initial_values=[[1], [2]], noise_stddev=1.0, output='show')
    # simulator.run("VARsim")
