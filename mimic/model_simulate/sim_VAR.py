from typing import List, Optional, Union, Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from mimic.model_simulate.base_model import BaseModel


class sim_VAR(BaseModel):
    """
    A class for simulating Vector Autoregression (VAR) models.

    Inherits from BaseModel and adds specific functionalities for VAR model simulation,
    including data generation for both univariate and multivariate autoregressive processes.

    This class allows users to simulate data from VAR models, specify model parameters, generate
    simulated data, visualize the results through various plotting methods, and save the generated
    data for further analysis. It supports both single and multi-variable autoregressive models,
    making it versatile for different simulation scenarios.

    Attributes:
        n_obs (int): Number of observations to generate.
        coefficients (np.ndarray): Coefficients of the VAR model.
        initial_values (np.ndarray): Initial values for the VAR model simulation.
        noise_stddev (float): Standard deviation of the noise in the VAR model.
        output (str): Specifies the output action for plots ('show', 'save', or 'both').
        dataM (np.ndarray): Holds the generated data for multivariate simulations.
        coefficientsM (np.ndarray): Coefficients for the multivariate VAR model.
        initial_valuesM (np.ndarray): Initial values for the multivariate VAR model simulation.

    Methods:
        set_parameters: Allows setting or updating model parameters like number of observations,
                        model coefficients, initial values, and noise standard deviation. It supports
                        both univariate and multivariate VAR models.

        generate_var1_data: Simulates data from a VAR(1) process using the specified model parameters
                            and saves the generated data. It can also generate and overlay plots based
                            on the 'output' attribute.

        generate_mvar1_data: Generates data from a multivariate autoregressive process. It can work
                             with complex interactions between multiple variables and supports overlay
                             plotting based on the 'output' attribute.

        simulate: Acts as a controller to execute the simulation based on the specified command. It
                  supports commands for simulating univariate VAR, multivariate VAR, and generating
                  plots as specified.

        make_plot_overlay: Creates overlay plots for visual comparison of simulated data across
                           different variables or processes.

        make_plot_stacked: Generates a stacked plot and heatmap for the given data, offering a
                           detailed visualization of the simulation results.

        make_plot: Produces separate line plots for each variable in the given data, facilitating
                   an in-depth analysis of each variable's behavior over time.
    """

    def __init__(self):
        """
        Initializes the sim_VAR instance with default parameter values.
        """
        super().__init__()

        self.model: str = "VAR"
        self.n_obs: Optional[int] = None
        self.coefficients: Optional[np.ndarray] = None
        self.initial_values: Optional[np.ndarray] = None
        self.noise_stddev: Optional[Union[int, float]] = None
        self.output: Optional[str] = None
        self.dataM: Optional[np.ndarray] = None
        self.coefficientsM: Optional[np.ndarray] = None
        self.initial_valuesM: Optional[np.ndarray] = None

        self.parameters: Dict[str, Optional[Union[int, float, np.ndarray, str]]] = {"n_obs": self.n_obs, "coefficients": self.coefficients,
                                                                                    "initial_values": self.initial_values, "noise_stddev": self.noise_stddev, "output": self.output}

    def set_parameters(self,
                       n_obs: Optional[int] = None,
                       coefficients: Optional[List[List[Union[int, float]]]] = None,
                       initial_values: Optional[List[List[int]]] = None,
                       noise_stddev: Optional[Union[int, float]] = None,
                       output: Optional[str] = None,
                       coefficientsM: Optional[List[List[Union[int, float]]]] = None,
                       initial_valuesM: Optional[List[List[int]]] = None) -> None:
        """
        Sets the parameters for the sim_VAR instance.

        Allows optional specification of all model parameters. Parameters not provided (None) are left unchanged.

        Parameters:
            n_obs (Optional[int]): Number of observations to generate.
            coefficients (Optional[List[List[Union[int, float]]]]): Coefficients of the VAR model.
            initial_values (Optional[List[List[int]]]): Initial values for the VAR model simulation.
            noise_stddev (Optional[Union[int, float]]): Standard deviation of the noise.
            output (Optional[str]): Output action for plots ('show', 'save', or 'both').
            coefficientsM (Optional[List[List[Union[int, float]]]]): Coefficients for the multivariate VAR model.
            initial_valuesM (Optional[List[List[int]]]): Initial values for the multivariate VAR model simulation.
        """
        if n_obs is not None:
            self.n_obs = n_obs
        if coefficients is not None:
            self.coefficients = np.array(coefficients)
        if initial_values is not None:
            self.initial_values = np.array(initial_values)
        if noise_stddev is not None:
            self.noise_stddev = noise_stddev
        if output is not None:
            self.output = output
        if coefficientsM is not None:
            self.coefficientsM = np.array(coefficientsM)
        if initial_valuesM is not None:
            self.initial_valuesM = np.array(initial_valuesM)

        self.parameters = {"n_obs": self.n_obs, "coefficients": self.coefficients,
                           "initial_values": self.initial_values, "noise_stddev": self.noise_stddev, "output": self.output,
                           "coefficientsM": self.coefficientsM, "initial_valuesM": self.initial_valuesM}

    def generate_var1_data(self) -> np.ndarray:
        """
        Generate simulated data from a VAR(1) process.

        Simulates a univariate or multivariate VAR(1) process based on the set parameters. 
        This method populates the `data` attribute with the generated data.

        Returns:
            np.ndarray: The generated data as a numpy array with shape (n_obs, number of variables).
        """
        # Check if the coefficients and initial values are provided
        if self.coefficients is None or self.initial_values is None or self.n_obs is None:
            raise ValueError(
                "coefficients, number of observations and initial_values must be provided for VARsim")

        dim = len(self.initial_values)
        data = np.zeros((self.n_obs, dim))
        data[0, :] = self.initial_values[:, 0]

        if self.noise_stddev is None:
            self.noise_stddev = 0.1
            print("noise_stddev not provided, setting to default value 0.1")

        for t in range(1, self.n_obs):
            # VAR(1) process: X_t = A * X_{t-1} + noise
            noise = np.random.normal(scale=self.noise_stddev, size=dim)
            data[t, :] = np.dot(self.coefficients, data[t - 1, :]) + noise

        if self.output != None:
            self.make_plot_overlay(data, None, self.output)

        self.data = data  # the generated data
        return data

    def generate_mvar1_data(self, coefficientsM: np.ndarray, initial_valuesM: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates synthetic data for a multivariate autoregressive (MVAR) process of order 1.

        Specifically tailored for generating data from complex MVAR processes where interactions
        between multiple variables are considered.

        Parameters:
            coefficientsM (np.ndarray): Coefficients for the MVAR model.
            initial_valuesM (np.ndarray): Initial values for the MVAR model simulation.

        Returns:
            tuple: A tuple containing two numpy.ndarrays. The first array is the generated data
                   for the X process, and the second array is the generated data for the S process.
                   Both arrays have shapes (n_obs, number of X variables) and (n_obs, number of S variables), respectively.
        """

        # Check if the coefficients and initial values are provided
        if coefficientsM is None or initial_valuesM is None:
            raise ValueError(
                "coefficients, number of observations and initial_values must be provided for MVARsim")

        # check if initial_values and n_obs is different from None
        if self.initial_values is None or self.n_obs is None or self.coefficients is None:
            raise ValueError(
                "initial_values and n_obs must be provided for MVARsim")

        nX = len(self.initial_values)
        data = np.zeros((self.n_obs, nX))
        data[0, :] = self.initial_values[:, 0]

        coefficientsM = np.array(coefficientsM)
        initial_valuesM = np.array(initial_valuesM)
        nS = len(initial_valuesM)
        dataM = np.zeros((self.n_obs, nS))
        dataM[0, :] = initial_valuesM[:, 0]

        if self.noise_stddev is None:
            self.noise_stddev = 0.1
            print("noise_stddev not provided, setting to default value 0.1")

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

        if self.output != None:
            self.make_plot_overlay(data, dataM, self.output)

        self.data, self.dataM = data, dataM  # the generated data
        return data, dataM

    def simulate(self, command: str) -> None:
        """
        Simulates data based on the specified command.

        Supports commands for simulating VAR and multivariate VAR (MVAR) processes.
        Adjusts internal state based on simulation results.

        Parameters:
            command (str): The simulation command ('VARsim' for VAR simulation, 'MVARsim' for multivariate VAR simulation).

        Raises:
            ValueError: If an invalid command is provided.
        """
        if command == "MVARsim":
            self.check_params(self.parameters, "sVAR")
            self.coefficientsM = (
                np.array([self.parameters["coefficientsM"]])
                if isinstance(self.parameters["coefficientsM"], (int, float, str))
                else self.parameters["coefficientsM"]
            )
            self.initial_valuesM = (
                np.array([self.parameters["initial_valuesM"]])
                if isinstance(
                    self.parameters["initial_valuesM"], (int, float, str)
                )
                else self.parameters["initial_valuesM"]
            )
            if self.coefficientsM is None or self.initial_valuesM is None:
                raise ValueError(
                    "coefficientsM and initial_valuesM must be provided for MVARsim")
            self.generate_mvar1_data(self.coefficientsM, self.initial_valuesM)
        elif command == "VARsim":
            self.check_params(self.parameters, "VAR")
            self.generate_var1_data()
        else:
            raise ValueError("Invalid command. Must be 'VARsim' or 'MVARsim'")

    def make_plot_overlay(self, dataX: np.ndarray, dataS: Optional[np.ndarray] = None, output: Optional[str] = 'show') -> None:
        """
        Creates an overlay plot of the given data for easy comparison.

        This method visualizes the time series data by overlaying the plots of each variable.
        It supports both the primary VAR process data and an optional secondary process data for comparison.

        Parameters:
            dataX (np.ndarray): The primary data series for the VAR process. Shape: (n_obs, num_variables).
            dataS (np.ndarray, optional): The secondary data series for comparison. Shape: (n_obs, num_variables).
            output (str): Controls the output of the plot ('show', 'save', or 'both').

        Note:
            Saves the plot as "plot-data-overlay.pdf" if 'save' or 'both' is selected as output.
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

    def make_plot_stacked(self, dataX: np.ndarray, dataS: np.ndarray) -> None:
        """
        Creates a stacked plot and a heatmap for the given data, providing a comprehensive view of the data dynamics.

        This method visualizes the time series data from the VAR process in a stacked plot for a clear overview
        of each variable's contribution over time. Additionally, it generates a heatmap for secondary data,
        offering an intuitive representation of data intensity across variables and time points.

        Parameters:
            dataX (np.ndarray): The primary data series from the VAR process. Shape: (n_obs, num_variables).
            dataS (np.ndarray): The secondary data series for heatmap visualization. Shape: (n_obs, num_variables).

        Note:
            Saves the plots as "plot-data-XS-stacked.pdf".
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

    def make_plot(self, dataX: np.ndarray, dataS: Optional[np.ndarray] = None, output: Optional[str] = 'show') -> None:
        """
        Generates separate line plots for each variable in the given data, facilitating detailed analysis.

        This method creates individual line plots for each variable in the primary and, optionally,
        secondary data series. This detailed visualization allows for in-depth analysis of each variable's
        behavior over time.

        Parameters:
            dataX (np.ndarray): The primary data series for the VAR process. Shape: (n_obs, num_variables).
            dataS (np.ndarray, optional): The secondary data series for comparison. Shape: (n_obs, num_variables).
            output (str): Controls the output of the plot ('show', 'save', or 'both').

        Note:
            Saves the individual plots as "plot-data-XS.pdf" if 'save' or 'both' is selected as output.
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

        if output in {'save', 'both'}:
            plt.savefig("plot-data-stacked.pdf")
        if output in {'show', 'both'}:
            plt.show()
