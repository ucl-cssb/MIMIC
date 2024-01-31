import numpy as np
from numpy.linalg import LinAlgError
from statsmodels.tsa.vector_ar.var_model import VARProcess


def simulate_var3(coefs=None, coefs_exog=None, sigma_u=None, steps=100, offset=None, initial_values=None, nsimulations=None, seed=None):
    """
    Simulate a Vector Autoregression (VAR) model.

    Parameters:
    - coefs (numpy.ndarray): Coefficients for lags of endog, shape p x k x k.
    - coefs_exog (numpy.ndarray): Parameters for trend and user-provided exog, shape k_exog x k.
    - sigma_u (numpy.ndarray): Residual covariance, shape k x k.
    - steps (int): The number of steps to simulate.
    - offset (numpy.ndarray): Observation specific intercept, shape steps x neqs.
    - initial_values (array_like): Initial values for simulation, shape (nlags, neqs) or (neqs,).
    - nsimulations (int): Number of simulations to perform.
    - seed (int): Random seed for reproducibility.

    Returns:
    - simulated_data (numpy.ndarray): The simulated dataset.
    """
    # Validate and set default values for inputs
    if coefs is None:
        coefs = np.array([[[0.5, -0.1], [-0.3, 0.2]]])  # Default coefficients
    if coefs_exog is None:
        coefs_exog = np.array([[0], [0]])  # Default exogenous coefficients
    if sigma_u is None:
        sigma_u = np.array([[1, 0.5], [0.5, 1]])  # Default residual covariance

    # Validate input shapes and types
    if not (isinstance(coefs, np.ndarray) and coefs.ndim == 3):
        raise ValueError(
            "coefs must be a 3D numpy array with shape p x k x k.")
    if not (isinstance(coefs_exog, np.ndarray) and coefs_exog.ndim == 2):
        raise ValueError(
            "coefs_exog must be a 2D numpy array with shape k_exog x k.")
    if not (isinstance(sigma_u, np.ndarray) and sigma_u.shape[0] == sigma_u.shape[1]):
        raise ValueError("sigma_u must be a 2D square numpy array (k x k).")
    if not (isinstance(steps, int) and steps > 0):
        raise ValueError("steps must be a positive integer.")
    if offset is not None and not (isinstance(offset, np.ndarray) and offset.shape == (steps, sigma_u.shape[0])):
        raise ValueError(
            f"offset must be a numpy array with shape ({steps}, {sigma_u.shape[0]}).")
    if initial_values is not None and not isinstance(initial_values, (list, np.ndarray)):
        raise ValueError("initial_values must be list or numpy array.")
    if nsimulations is not None and not (isinstance(nsimulations, int) and nsimulations > 0):
        raise ValueError("nsimulations must be a positive integer.")

    # Set the random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Initialize VARProcess
    try:
        var_process = VARProcess(
            coefs=coefs, coefs_exog=coefs_exog, sigma_u=sigma_u)
    except LinAlgError as e:
        raise ValueError(f"Error in initializing VARProcess: {e}")

    # Simulate VAR model
    try:
        simulated_data = var_process.simulate_var(
            steps=steps, offset=offset, seed=seed, initial_values=initial_values, nsimulations=nsimulations)
    except Exception as e:
        raise ValueError(
            f"Error during simulation: {e}. Ensure that the statsmodels library is up-to-date and properly installed.")

    return simulated_data


# Sample usage:
if __name__ == "__main__":
    # User-defined or default values for coefs, coefs_exog, sigma_u, etc.
    # Define these variables as per your specific needs or use the defaults set in the function

    coefs = np.array([[[0.5, -0.1], [-0.3, 0.2]]])  # shape p x k x k
    coefs_exog = np.array([[0], [0]])  # shape k_exog x k
    sigma_u = np.array([[1, 0.5], [0.5, 1]])  # shape k x k
    steps = 100
    offset = None  # or define your own
    initial_values = None  # or define your own
    nsimulations = None  # or define your own
    seed = 42  # or define your own

    simulated_data = simulate_var3(
        coefs, coefs_exog, sigma_u, steps, offset, initial_values, nsimulations, seed)
    print(simulated_data)
