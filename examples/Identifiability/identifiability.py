import sympy as sp
from sympy import symbols, Function, Matrix, Eq, simplify

# Define time and species
t = sp.symbols('t')
N1, N2, N3 = sp.symbols('N1 N2 N3', cls=Function)

# Define parameters
mu1, mu2, mu3 = sp.symbols('mu1 mu2 mu3')
M11, M12, M13 = sp.symbols('M11 M12 M13')
M21, M22, M23 = sp.symbols('M21 M22 M23')
M31, M32, M33 = sp.symbols('M31 M32 M33')

# Define the ODE system (Lotka-Volterra)
dN1_dt = N1(t).diff(t) - (mu1 * N1(t) + M11 * N1(t)**2 + M12 * N1(t) * N2(t) + M13 * N1(t) * N3(t))
dN2_dt = N2(t).diff(t) - (mu2 * N2(t) + M21 * N1(t) * N2(t) + M22 * N2(t)**2 + M23 * N2(t) * N3(t))
dN3_dt = N3(t).diff(t) - (mu3 * N3(t) + M31 * N1(t) * N3(t) + M32 * N2(t) * N3(t) + M33 * N3(t)**2)

# State vector and parameter vector
X = Matrix([N1(t), N2(t), N3(t)])
theta = Matrix([mu1, mu2, mu3, M11, M12, M13, M21, M22, M23, M31, M32, M33])

# Jacobian of the system wrt parameters
F = Matrix([dN1_dt, dN2_dt, dN3_dt])
J_theta = F.jacobian(theta)

# Print the Jacobian
print("Jacobian of the system with respect to parameters:")
sp.pprint(J_theta)

# Simplify the equations (optional)
simplified_eqs = simplify(F)
print("\nSimplified ODEs:")
sp.pprint(simplified_eqs)

# Check rank of Jacobian for identifiability
rank = J_theta.rank()
print(f"\nRank of the Jacobian: {rank} / {len(theta)}")
if rank == len(theta):
    print("All parameters are structurally identifiable.")
else:
    print("Some parameters may not be identifiable.")

# Assume M11, M22, M33 are known (diagonal elements of interaction matrix)
known_params = {M11: 1, M22: 1, M33: 1}
simplified_J = J_theta.subs(known_params)
new_rank = simplified_J.rank()
print(f"Rank of the simplified Jacobian: {new_rank} / 9")  # Expecting rank for the remaining 9 parameters

