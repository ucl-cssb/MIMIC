import json
import numpy as np
import os


def compute_rmse(true, inferred):
    """Compute the root-mean-square error between two arrays."""
    true = np.array(true)
    inferred = np.array(inferred)
    return np.sqrt(np.mean((true - inferred) ** 2))


def compute_relative_error(true, inferred):
    """Compute a relative error metric (L2 norm difference divided by L2 norm of true)."""
    true = np.array(true)
    inferred = np.array(inferred)
    return np.linalg.norm(true - inferred) / np.linalg.norm(true)


def main():
    # Path to the combined inferred parameters file
    results_file = "inference_results/all_inferred_parameters.json"
    if not os.path.exists(results_file):
        print("File not found:", results_file)
        return

    with open(results_file, "r") as f:
        all_results = json.load(f)

    # Lists to store per-simulation errors
    mu_rmse_list = []
    mu_rel_list = []
    epsilon_rmse_list = []
    epsilon_rel_list = []
    M_rmse_list = []
    M_rel_list = []

    for sim in all_results:
        mu_true = np.array(sim["true_mu"])
        mu_inf = np.array(sim["inferred_mu"])
        eps_true = np.array(sim["true_epsilon"])
        eps_inf = np.array(sim["inferred_epsilon"])
        M_true = np.array(sim["true_M"])
        M_inf = np.array(sim["inferred_M"])

        mu_rmse = compute_rmse(mu_true, mu_inf)
        mu_rel = compute_relative_error(mu_true, mu_inf)
        eps_rmse = compute_rmse(eps_true, eps_inf)
        eps_rel = compute_relative_error(eps_true, eps_inf)
        # Flatten the matrices to compare element-wise
        M_rmse = compute_rmse(M_true.flatten(), M_inf.flatten())
        M_rel = compute_relative_error(M_true.flatten(), M_inf.flatten())

        mu_rmse_list.append(mu_rmse)
        mu_rel_list.append(mu_rel)
        epsilon_rmse_list.append(eps_rmse)
        epsilon_rel_list.append(eps_rel)
        M_rmse_list.append(M_rmse)
        M_rel_list.append(M_rel)

    # Compute average metrics
    avg_mu_rmse = np.mean(mu_rmse_list)
    avg_mu_rel = np.mean(mu_rel_list)
    avg_eps_rmse = np.mean(epsilon_rmse_list)
    avg_eps_rel = np.mean(epsilon_rel_list)
    avg_M_rmse = np.mean(M_rmse_list)
    avg_M_rel = np.mean(M_rel_list)

    # Print summary metrics
    print("Aggregated metrics over {} simulations:".format(len(all_results)))
    print("Intrinsic growth rates (μ): RMSE = {:.4f}, Relative error = {:.4f}".format(
        avg_mu_rmse, avg_mu_rel))
    print("Perturbation terms (ε): RMSE = {:.4f}, Relative error = {:.4f}".format(
        avg_eps_rmse, avg_eps_rel))
    print("Interaction matrix (M): RMSE = {:.4f}, Relative error = {:.4f}".format(
        avg_M_rmse, avg_M_rel))

    # Save the aggregated metrics to a JSON file for later reference.
    aggregated_metrics = {
        "avg_mu_rmse": float(avg_mu_rmse),
        "avg_mu_relative_error": float(avg_mu_rel),
        "avg_epsilon_rmse": float(avg_eps_rmse),
        "avg_epsilon_relative_error": float(avg_eps_rel),
        "avg_M_rmse": float(avg_M_rmse),
        "avg_M_relative_error": float(avg_M_rel)
    }
    out_file = os.path.join("inference_results", "aggregated_metrics.json")
    with open(out_file, "w") as f:
        json.dump(aggregated_metrics, f, indent=4)
    print("Aggregated metrics saved to:", out_file)


if __name__ == "__main__":
    main()
