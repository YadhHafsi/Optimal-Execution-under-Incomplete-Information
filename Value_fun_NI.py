import numpy as np
from scipy.integrate import solve_ivp
import parameters as params
from numba import njit
from scipy.integrate import quad
import math


#parameters

T = params.T  # Total simulation time
d = params.d  # Number of states
psi = params.psi  # Transition rate matrix
n_intgr = params.n_intgr  # Number of intervals for numerical integration
n= params.n 

t_span = params.t_span

λ_init, α, β, η, xi = params.λ_init, params.α, params.β, params.η, params.xi

pi0 = params.pi0 
kappa = params.kappa

beta = β
lambda_infty = params.lambda_infty

rho = params.rho
c = params.c
e = params.e
eta = params.eta
c_prime = params.c_prime
c0 = params.c0
m1 = params.m1
E_phi = quad(lambda v: (η*v/m1)**eta/math.gamma(1+eta) *η*np.exp(-η*v), 0, np.inf)[0]
E_Q = quad(lambda v: (v*(e/c))**(-e) *η*np.exp(-η*v), 0, np.inf)[0]



# Define the intensity function lambda
@njit
def lambda_ik(t, u, kappa):#ok
    lambda_ik_u = np.zeros((d, 2))
    for i in range(d):
        lambda_ik_u[i, 0] = (kappa[i, 0]-lambda_infty[i]) * np.exp(-beta[i] * (u - t)) + lambda_infty[i]
        lambda_ik_u[i, 1] = (kappa[i, 1]-lambda_infty[i]) * np.exp(-beta[i] * (u - t)) + lambda_infty[i]
    return lambda_ik_u

@njit
def expectation_lambda(t, u, kappa):#ok
    expectation_lambda_ik_u = np.zeros((d, 2))
    
    for i in range(d):
        kappa_delta0 = (kappa[i, 0]+kappa[i, 1])/2
        kappa_delta1 = (kappa[i, 0]-kappa[i, 1])/2
        I = E_phi
        
        I1 = (beta[i]*lambda_infty[i])/(beta[i] - I)
        
        expectation_lambda_ik_u[i, 0] = I1 + (kappa_delta0-I1) * np.exp((-beta[i]+I) * (u - t)) + kappa_delta1*np.exp(-beta[i] * (u - t))
        expectation_lambda_ik_u[i, 1] = I1 + (kappa_delta0-I1) * np.exp((-beta[i]+I) * (u - t)) - kappa_delta1*np.exp(-beta[i] * (u - t))

    return expectation_lambda_ik_u

@njit
def expectation_pi(t, u, mu):#ok
    """
    Compute the transition probability matrix P(t) based on the closed-form expression.
    
    Parameters:
    t (float): The time at which to evaluate the transition matrix.
    
    Returns:
    numpy.ndarray: The transition probability matrix P(t).
    """
    delta_t = u - t
    
    # Compute the exponential term
    exp_term = np.exp(-0.4 * delta_t) #2
    
    # Compute P_11(t - t_0) and P_21(t - t_0)
    P_11 = 0.5 * (1 + exp_term)
    P_21 = 0.5 * (1 - exp_term)
    
    # Compute P(I_t = 1)
    pi = mu[0] * P_11 + mu[1] * P_21
    
    return np.array([pi, 1-pi])

@njit
def numba_dot(a, b):
    """Numba-compatible dot product function."""
    result = 0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result

@njit
def numba_dot_matrix(a, b):
    """Numba-compatible dot product for 2D matrices."""
    rows_a = a.shape[0]
    cols_a = a.shape[1]
    cols_b = b.shape[1]
    
    # Initialize result matrix with zeros
    result = np.zeros((rows_a, cols_b))
    
    # Perform matrix multiplication
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i, j] += a[i, k] * b[k, j]
    
    return result

@njit
def matrix_exponential(A, tol=1e-16, max_iter=10):
    n = A.shape[0]
    I = np.eye(n)
    normA = np.linalg.norm(A, ord=np.inf)
    s = max(0, int(np.log2(normA)) + 1)
    A_scaled = A / (2 ** s)
    
    expA = I.copy()
    term = I.copy()
    k = 1

    while np.linalg.norm(term, ord=np.inf) > tol and k < max_iter:
        term = term @ A_scaled / k
        expA = expA + term
        k += 1

    for _ in range(s):
        expA = expA @ expA

    return expA

@njit
def W0(t, y):#ok
    x, d, kappa, mu = y
    
    h = (T - t) / n_intgr
    outer_integral = 0.0
    
    for j in range(n_intgr+1):
        u = t + j * h
        if j == 0 or j == n_intgr:
            weight = 0.5
        else:
            weight = 1.0
        
        temp = np.sum((expectation_lambda(t, u, kappa)[:,0] - expectation_lambda(t, u, kappa)[:,1])* expectation_pi(t, u, mu))
        
        inner_integrand = (params.nu_0 + (1 - params.nu_0) *np.exp(-rho * (T - u))) *temp
        
        outer_integral += weight * inner_integrand
        
    outer_integral *= h
    
    return x*d*np.exp(-rho*(T-t)) - c_prime * x ** ((e + 1) / e) - c0 + x*outer_integral*E_Q#s+ 


# Define the ODE system function
def ode_system_old(u, pi, t, kappa):
    d_pi = np.zeros(d)
    lambd = lambda_ik(t, u, kappa)
    
    for i in range(d):
        term1 = np.dot(pi, psi[:, i])
        term2 = np.sum([pi[i] * (-lambd[i, k] + np.sum(pi[j] * lambd[j, k] for j in range(d))) for k in range(2)])
        d_pi[i] = term1 + term2
    
    return d_pi


def ode_system(u, pi, t, kappa):
    lambd = lambda_ik(t, u, kappa)
    
    # Vectorized term1 calculation: pi is (d,), psi is (d, 2)
    term1 = np.dot(pi, psi)
    
    # Vectorized term2 calculation
    # pi[:, np.newaxis] expands pi to shape (d, 1), lambd is (d, 2)
    # np.sum(pi[:, np.newaxis] * lambd, axis=0) sums over the first axis, resulting in shape (2,)
    term2 = np.sum(pi[:, np.newaxis] * (-lambd + np.sum(pi[:, np.newaxis] * lambd, axis=0)), axis=1)
    
    # Sum the vectorized terms
    d_pi = term1 + term2
    
    return d_pi

# Define the normalization function
def normalize(pi):
    return pi / np.sum(pi)

# Solve the ODE system with constraints
def solve_ode_with_constraints(t_span, pi0, kappa_0, nbr_pnts = 10):
    t_span_0 = t_span
    if t_span_0[0] == t_span_0[-1]:
        t_span_0 = (t_span_0[0], t_span_0[0] + 1e-10)
    sol = solve_ivp(lambda u, pi: ode_system(u, pi, t_span_0[0], kappa_0), t_span_0, pi0, method='RK45', t_eval=np.linspace(t_span_0[0], t_span_0[-1], nbr_pnts))   
    return sol

# Solve the ODE system with constraints after jump
def solve_ode_with_constraints_with_jump(t_span, pi0, kappa_0, jump):
    sol = solve_ivp(lambda u, pi: ode_system(u, pi, t_span[0], kappa_0), t_span, pi0, method='RK45', t_eval=np.linspace(t_span[0], t_span[-1], n_intgr + 1))
    
    sol.y[:, -1] += jump
        
    # Post-process to enforce unit simplex constraint
    for i in range(len(sol.y[0])):
        sol.y[:, i] = normalize(sol.y[:, i])
    return sol

def first_part_I0_ts(t, kappa, mu):
    
    h_outer = (T - t) / n_intgr
    outer_integral = 0.0
    t_span0 = (t,T+1e-15)
    
    pi_u = solve_ode_with_constraints(t_span0, mu, kappa, n_intgr + 1).y#complexity here
    
    for j in range(n_intgr+1):
        u = t + j * h_outer
        if j == 0 or j == n_intgr:
            weight = 0.5
        else:
            weight = 1.0
            
        inner_integrand = np.exp(-rho * (T - u)) *np.sum(pi_u[:,j] *E_Q * (lambda_ik(t, u, kappa)[:,1] - lambda_ik(t, u, kappa))[:,0])

        outer_integral += weight * inner_integrand
        
    outer_integral *= h_outer
    return outer_integral

