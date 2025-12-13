import numpy as np

#parameters
T = 4  # Total simulation time
d = 2  # Number of states
psi = np.array([[-0.2, 0.2], [0.2, -0.2]])  # Transition rate matrix
n_intgr = 100  # Number of intervals for numerical integration
positive_infinity = 15
n= 10
epsilon = 1e-6

# Time span for integration
t_span = (0, 4)  # Integration time from 0 to 10

λ_init, α, β, η, xi, rho = np.array([0.001, 0.001]), np.array([np.array([0.5, 0.5]),np.array([0.5, 0.5])]), np.array([1, 1]), 0.05, 2, 0.5


# Define parameters
pi0 = np.array([0.50001, 0.49999])  # Initial state probabilities at tau[0]

# Example parameters for lambda functions
kappa = np.array([λ_init[0], λ_init[1]])
#kappa = np.array([[λ_init[0], λ_init[1]], [xi*λ_init[0], xi*λ_init[1]]])
fund_price = 50

beta = β
lambda_infty = np.array([1, 1])

rho = 0.1
m1 = 1/η
c = 1.0
e = 0.3
eta = 0.5
c_prime = (c / (e + 1)) * (e / c) ** ((e + 1) / e)
c0 = 0.1
nu_0 = 0.8