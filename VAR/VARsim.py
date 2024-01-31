import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pretty_errors


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
        axs[0].plot(dataX[:, i], label="X" + str(i))
    axs[0].set_title("Abundance, X")

    # Plot each variable in dataS on the second subplot, if available
    if dataS is not None:
        nS = dataS.shape[1]
        for i in range(nS):
            axs[1].plot(dataS[:, i], label="S" + str(i))
        axs[1].set_title("Metabolites, S")

    # Adjust the layout to ensure that the subplots do not overlap
    plt.tight_layout()

    # If the output option is 'save' or 'both', save the figure as a PDF
    if output == 'save' or output == 'both':
        plt.savefig("plot-data-overlay.pdf")

    # If the output option is 'show' or 'both', show the figure in a new window
    if output == 'show' or output == 'both':
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
    # add 5.6 to add abundance data
    dataX = dataX + 1.0

    #  stacked
    nX = len(dataX[0])  # Number of columns in dataX
    nS = len(dataS[0])  # Number of columns in dataS
    nobs = dataS.shape[0]

    # Create a figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 4))
    # Stack plot for dataX
    axs[0].stackplot(range(len(dataX)), *dataX.T,
                     labels=["X" + str(i) for i in range(nX)])
    axs[0].set_title("Abundance, log10 X")
    axs[0].set_ylabel("X")
    axs[0].set_xlim(0, nobs-1)
    # axs[0].legend(loc='upper left')

    # Stack plot for dataS
    # axs[1].stackplot(range(len(dataS)), *dataS.T, labels=["S" + str(i) for i in range(nS)])
    # axs[1].set_title("Metabolites, S")
    # axs[1].legend(loc='upper left')

    # Heatmap for dataS
    # sns.heatmap(dataS, annot=False, cmap="YlGnBu", xticklabels=range(1, dataS.shape[0] + 1), yticklabels=["S" + str(i) for i in range(nS)], ax=axs[1])
    sns.heatmap(dataS.T, annot=False, cmap="YlGnBu", yticklabels=[
                "S" + str(i) for i in range(nS)], ax=axs[1], cbar=False)
    axs[1].set_title("Metabolites, S")
    axs[1].set_ylabel("S")
    axs[1].set_xlabel("time (weeks)")
    axs[1].set_xlim(0, nobs)

    plt.tight_layout()  # Adjust the layout
    plt.savefig("plot-data-XS-stacked.pdf")


def make_plot(dataX, dataS):
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
    nS = len(dataS[0])  # Number of columns in dataS

    fig, axs = plt.subplots(nX + nS, 1, figsize=(10, 2*(nX+nS)))
    for i, ax in enumerate(axs):
        if i < nX:
            axs[i].plot(dataX[:, i])
            axs[i].set_title("X"+str(i))
        else:
            axs[i].plot(dataS[:, i-nX])
            axs[i].set_title("S"+str(i-nX))
    plt.savefig("plot-data-XS.pdf")


def generate_var1_data(n_obs, coefficients, initial_values, noise_stddev=1, output='show'):
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
    dim = len(initial_values)
    data = np.zeros((n_obs, dim))
    data[0, :] = initial_values[:, 0]

    for t in range(1, n_obs):
        # VAR(1) process: X_t = A * X_{t-1} + noise
        noise = np.random.normal(scale=noise_stddev, size=dim)
        data[t, :] = np.dot(coefficients, data[t - 1, :]) + noise

    if output != 'none':
        make_plot_overlay(data, None, output)

    return data


def generate_mvar1_data(n_obs, coefficients, coefficientsM, initial_values, initial_valuesM, noise_stddev=1, output='show'):
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
    nX = len(initial_values)
    data = np.zeros((n_obs, nX))
    data[0, :] = initial_values[:, 0]

    nS = len(initial_valuesM)
    dataM = np.zeros((n_obs, nS))
    dataM[0, :] = initial_valuesM[:, 0]

    for t in range(1, n_obs):
        # VAR(1) process: X_t = A * X_{t-1} + noise
        noise = np.random.normal(scale=noise_stddev, size=nX)
        # print("A", coefficients.shape)
        # print("X", data[t - 1, :].shape)
        data[t, :] = np.dot(coefficients, data[t - 1, :]) + noise

    for t in range(1, n_obs):
        # process: S_t = B * X_{t-1} + noise
        noise = np.random.normal(scale=noise_stddev, size=(nS))
        # print("B:", coefficientsM.shape)
        # print("X", data[t - 1, :].shape)

        Xt = data[t - 1, :].reshape((nX, 1))
        # print( "mult:", (coefficientsM @ Xt).shape )
        product = coefficientsM @ Xt
        dataM[t, :] = product[:, 0] + noise

    if output != 'none':
        make_plot_overlay(data, dataM, output)

    return data, dataM


def str_to_np_array(str):
    return np.array(eval(str))


def main():
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(
        usage="%(prog)s [command] [options]",
        description="MICOM's VAR and MultivariateVAR Simulations")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    # Sub-parser for VAR method
    var_parser = subparsers.add_parser("VARsim", help="Run the VAR simulation")
    var_parser.add_argument("--n_obs", type=int, default=100,
                            help="Number of observations to generate (default: 100)")
    var_parser.add_argument("--coefficients", type=str_to_np_array, default='[[0.8, -0.2], [0.3, 0.5]]',
                            help="Coefficient matrix for the VAR(1) process")
    var_parser.add_argument("--initial_values", type=str_to_np_array, default='[[1], [2]]',
                            help="Initial values for the process")
    var_parser.add_argument("--noise_stddev", type=float, default=1.0,
                            help="Standard deviation of the normally distributed noise term (default: 1.0)")
    var_parser.add_argument("--output", type=str, choices=['show', 'save', 'both', 'none'], default='show',
                            help="Output option for the plot: 'show' to display the plot, 'save' to save it as a PDF, or 'both' to do both (default: 'show')")

    # Sub-parser for MultivariateVAR method
    multivar_parser = subparsers.add_parser(
        "MVARsim", help="Run the MultivariateVAR simulation")
    multivar_parser.add_argument("--n_obs", type=int, default=100,
                                 help="Number of observations to generate (default: 100)")
    multivar_parser.add_argument("--coefficients", type=str_to_np_array, required=True,
                                 help="The coefficients of the VAR process for X")
    multivar_parser.add_argument("--coefficientsM", type=str_to_np_array, required=True,
                                 help="The coefficients of the process for S")
    multivar_parser.add_argument("--initial_values", type=str_to_np_array, required=True,
                                 help="The initial values of the X process")
    multivar_parser.add_argument("--initial_valuesM", type=str_to_np_array, required=True,
                                 help="The initial values of the S process")
    multivar_parser.add_argument("--noise_stddev", type=float, default=1.0,
                                 help="The standard deviation of the Gaussian noise (default: 1.0)")
    multivar_parser.add_argument("--output", type=str, choices=['show', 'save', 'both', 'none'], default='show',
                                 help="Output option for the plot: 'show' to display the plot, 'save' to save it as a PDF, or 'both' to do both (default: 'show')")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the appropriate method based on the command
    if args.command == "VARsim":
        print(f"Running VAR with parameters: {args}")
        if args.n_obs == 100:
            print("Using default value for n_obs of 100")
        if args.noise_stddev == 1.0:
            print("Using default value for noise_stddev of 1.0")
        # check if args.coefficients np.array is [[0.8, -0.2], [0.3, 0.5]]
        if np.array_equal(args.coefficients, np.array([[0.8, -0.2], [0.3, 0.5]])):
            print(
                "Using default value for coefficients of [[0.8, -0.2], [0.3, 0.5]]")
        if np.array_equal(args.initial_values, np.array([[1], [2]])):
            print("Using default value for initial_values of [[1], [2]]")
        generate_var1_data(args.n_obs, args.coefficients,
                           args.initial_values, args.noise_stddev, args.output)
    elif args.command == "MVARsim":
        print(f"Running MultivariateVAR with parameters: {args}")
        if args.n_obs == 100:
            print("Using default value for n_obs")
        if args.noise_stddev == 1.0:
            print("Using default value for noise_stddev")
        generate_mvar1_data(args.n_obs, args.coefficients, args.coefficientsM,
                            args.initial_values, args.initial_valuesM, args.noise_stddev, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

    np.random.seed(42)  # for reproducibility

    n_obs = 96
    nX = 2
    A = np.array([[0.8, -0.2], [0.3, 0.5]])
    X0 = np.array([1, 2]).reshape((2, 1))

    nS = 3
    # B = np.array([[0, 0], [0.3, 0.5]])

    B = np.zeros((nS, nX))
    B[0, 1] = 0.8
    B[2, 1] = -0.5
    # print(B)

    S0 = np.zeros(nS).reshape((nS, 1))
    # print(S0)

    # dataX, dataS = generate_mvar1_data(n_obs, A, B, X0, S0)

    # make_plot(dataX, dataS)
    # make_plot_stacked(dataX, dataS)
    # make_plot_overlay(dataX, dataS)
