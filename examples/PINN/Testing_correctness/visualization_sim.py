import json
import matplotlib.pyplot as plt

# Path to the simulation file
file_path = r"sim_013.json"

with open(file_path, "r") as f:
    data = json.load(f)

t_span = data["t_span"]
# Choose one replicate (here, the first replicate)
replicate = data["replicates"][1]
noisy_solution = replicate["noisy_solution"]

if not noisy_solution:
    print("No simulation data found in the chosen replicate.")
    exit()

# Determine the number of species from the first time point
n_species = len(noisy_solution[0])

# Plot each species trajectory
for s in range(n_species):
    species_data = [state[s] for state in noisy_solution]
    plt.plot(t_span, species_data, label=f"Species {s+1}")

plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Simulation: All Species Trajectories (Replicate 1)")
plt.legend()
plt.show()
