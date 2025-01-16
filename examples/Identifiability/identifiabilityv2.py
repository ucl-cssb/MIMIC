import sympy as sp
from sympy import symbols, Function, Matrix

# Define time and species
t = sp.symbols('t')
N1, N2, N3 = sp.symbols('N1 N2 N3', cls=Function)

# Define parameters
mu1, mu2, mu3 = sp.symbols('mu1 mu2 mu3')
M12, M13, M21, M23, M31, M32 = sp.symbols('M12 M13 M21 M23 M31 M32')

# Assume diagonals of M are -1
M11, M22, M33 = -1, -1, -1

# Define the ODE system with updated M
dN1_dt = N1(t).diff(t) - (mu1 * N1(t) + M11 * N1(t) **
                          2 + M12 * N1(t) * N2(t) + M13 * N1(t) * N3(t))
dN2_dt = N2(t).diff(t) - (mu2 * N2(t) + M21 * N1(t) *
                          N2(t) + M22 * N2(t)**2 + M23 * N2(t) * N3(t))
dN3_dt = N3(t).diff(t) - (mu3 * N3(t) + M31 * N1(t) *
                          N3(t) + M32 * N2(t) * N3(t) + M33 * N3(t)**2)

# State vector and parameter vector
X = Matrix([N1(t), N2(t), N3(t)])
theta = Matrix([mu1, mu2, mu3, M12, M13, M21, M23, M31, M32])

# Jacobian of the system wrt parameters
F = Matrix([dN1_dt, dN2_dt, dN3_dt])
J_theta = F.jacobian(theta)

# Print the Jacobian and assess identifiability
print("Jacobian of the system with respect to parameters:")
sp.pprint(J_theta)

# Check rank of Jacobian
rank = J_theta.rank()
print(f"\nRank of the Jacobian: {rank} / {len(theta)}")
if rank == len(theta):
    print("All parameters are structurally identifiable.")
else:
    print("Some parameters may not be identifiable.")
