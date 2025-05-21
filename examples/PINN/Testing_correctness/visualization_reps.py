import json
import matplotlib.pyplot as plt

# Path to the simulation file
file_path = r"sim_004.json"

with open(file_path, "r") as f:
    data = json.load(f)

t_span = data["t_span"]
replicates = data["replicates"]

for i, rep in enumerate(replicates, 1):
    # Plot the trajectory of the first species from each replicate
    noisy_solution = rep["noisy_solution"]
    species1 = [state[0] for state in noisy_solution]
    plt.plot(t_span, species1, label=f"Replicate {i}")

plt.xlabel("Time")
plt.ylabel("Species 1")
plt.title("Simulation Replicates: Species 1 Trajectories")
plt.legend()
plt.show()
