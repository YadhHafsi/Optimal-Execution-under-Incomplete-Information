from numba import njit
import numpy as np
import Value_fun_NI as res0
import parameters as params
import math
from scipy.optimize import minimize_scalar 
from scipy.integrate import quad

from scipy.interpolate import interp1d

#parameters

# Time span for integration#parameters

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
m1 = params.m1
c = params.c
e = params.e
eta = params.eta
c_prime = params.c_prime
c0 = params.c0
nu_0 = params.nu_0
epsilon = params.epsilon

positive_infinity = params.positive_infinity

E_Q = quad(lambda v: (v*(e/c))**(-e) *η*np.exp(-η*v), 0, np.inf)[0] 


# Define the price impact function

@njit
def Q(v):#ok
    if v==0:
        return 0
    else:
        return (e * v / c) ** (-e)


# Define the transaction price function
@njit
def C(p, v):#ok
    if v ==0:
        return -c0
    else:
        return p * v - c_prime * v ** ((e + 1) / e) - c0

@njit
def varphi(v):
    if v==0:
        return 0
    else:
        return np.sqrt(η*v)/math.gamma(1+ eta)
    
@njit
def nu(v):#ok
    return η*np.exp(-η*v)

def m(t, u, y, i):#ok
    _, _, kappa, mu = y
    
    kappa_plus, kappa_minus  = kappa[:,0], kappa[:,1]

    D0 = np.diag(-(kappa_plus + kappa_minus - 2*lambda_infty) * (1 - np.exp(-beta * (u - t))) / beta - 2*lambda_infty * (u - t))
    
    first_integral = np.dot(res0.matrix_exponential(psi*(u-t) + D0), mu)[i]
    
    def integrand_plus(s):
        D = np.diag(-(kappa_plus + kappa_minus - 2 * lambda_infty) * (np.exp(-beta * (u - t)) - np.exp(-beta * (s - t))) / beta - 2 * lambda_infty * (s - u))
        lambda_plus_s = (kappa_plus - lambda_infty) * np.exp(-beta * (s - t)) + lambda_infty
        exp_term = res0.matrix_exponential(psi * (s - u) + D)
        return np.dot(lambda_plus_s, exp_term[i, :])

    # Perform adaptive integration using quad
    integral_result1, _ = quad(integrand_plus, u, 10, epsabs=1e-1, epsrel=1e-1)
    return first_integral, first_integral * integral_result1, first_integral- first_integral * integral_result1


@njit
def m_simple(t, u, y):#ok
    _, _, kappa, mu = y
    kappa_plus, kappa_minus = kappa[:, 0], kappa[:, 1]
    
    # Compute D0 matrices for both u1 and u2
    D0 = np.diag(-(kappa_plus + kappa_minus - 2 * lambda_infty) * (1 - np.exp(-beta * (u - t))) / beta - 2 * lambda_infty * (u - t))
    
    expm = res0.matrix_exponential(psi * (u - t) + D0)
    # Compute the first integral for both u1 and u2
    
    first_integral = np.dot(mu, expm)        
        
    # Return the desired outputs for both u1 and u2
    return first_integral

@njit
def vector_matrix_multiplication(v, M):
    # Get the number of rows
    n_rows = M.shape[0]
    
    # Initialize the result as a scalar
    result = np.zeros(n_rows)
    
    # Perform the dot product manually
    for i in range(n_rows):
        result += v[i] * M[i]
    
    return result


@njit
def m_double(t, u1, u2, y):#ok
    _, _, kappa, mu = y
    kappa_plus, kappa_minus = kappa[:, 0], kappa[:, 1]
    
    # Compute D0 matrices for both u1 and u2
    D0_1 = np.diag(-(kappa_plus + kappa_minus - 2 * lambda_infty) * (1 - np.exp(-beta * (u1 - t))) / beta - 2 * lambda_infty * (u1 - t))
    D0_2 = np.diag(-(kappa_plus + kappa_minus - 2 * lambda_infty) * (1 - np.exp(-beta * (u2 - t))) / beta - 2 * lambda_infty * (u2 - t))
    
    expm1 = res0.matrix_exponential(psi * (u1 - t) + D0_1)
    expm2 = res0.matrix_exponential(psi * (u2 - t) + D0_2)
    # Compute the first integral for both u1 and u2
    
    first_integral_1 = np.dot(mu, expm1)
    first_integral_2 = np.dot(mu, expm2)
    
    h = 10 / n_intgr
    
    integral_result1_1 = np.zeros(2)
    integral_result1_2 = np.zeros(2)
    integral_result2_1 = np.zeros(2)
    integral_result2_2 = np.zeros(2)

    s1 = u1
    s2 = u2
    
    for j in range(n_intgr):
        if j == 0 or j == n_intgr:
            weight = 0.5
        else:
            weight = 1.0
        # For u1
        D1 = np.diag(-(kappa_plus + kappa_minus - 2 * lambda_infty) * (np.exp(-beta * (u1 - t)) - np.exp(-beta * (s1 - t))) / beta - 2 * lambda_infty * (s1 - u1))
        lambda_plus_s1 = (kappa_plus - lambda_infty) * np.exp(-beta * (s1 - t)) + lambda_infty
        lambda_minus_s1 = (kappa_minus - lambda_infty) * np.exp(-beta * (s1 - t)) + lambda_infty
        exp_term1 = res0.matrix_exponential(psi * (s1 - u1) + D1)
        integral_result1_1 += np.dot(lambda_plus_s1, exp_term1) * h*weight
        integral_result2_1 += np.dot(lambda_minus_s1, exp_term1) * h*weight
        s1 += h
        
        # For u2
        D2 = np.diag(-(kappa_plus + kappa_minus - 2 * lambda_infty) * (np.exp(-beta * (u2 - t)) - np.exp(-beta * (s2 - t))) / beta - 2 * lambda_infty * (s2 - u2))
        lambda_plus_s2 = (kappa_plus - lambda_infty) * np.exp(-beta * (s2 - t)) + lambda_infty
        lambda_minus_s2 = (kappa_minus - lambda_infty) * np.exp(-beta * (s2 - t)) + lambda_infty
        exp_term2 = res0.matrix_exponential(psi * (s2 - u2) + D2)
        integral_result1_2 += np.dot(lambda_plus_s2, exp_term2) * h*weight
        integral_result2_2 += np.dot(lambda_minus_s2, exp_term2) * h*weight
        s2 += h
        
        
    # Return the desired outputs for both u1 and u2
    return (first_integral_1, first_integral_1 * integral_result1_1, first_integral_1 * integral_result2_1, first_integral_2, first_integral_2 * integral_result1_2, first_integral_2 * integral_result2_2)


#@njit
def H_k_0_w(u, w, k, kappa, mu):
    # Extract parameters from kappa and mu
    kappa1_plus, kappa2_plus, kappa1_minus, kappa2_minus = kappa[0, 0], kappa[1, 0], kappa[0, 1], kappa[1, 1]
    mu1, mu2 = mu[0], mu[1]
    v_max = 130  # Upper limit for v, as np.inf cannot be used in practice
    h = v_max / n_intgr  # Step size for numerical integration
    integral_result = 0.0
    #n = int(n_intgr/2)

    for i in range(n_intgr + 1):
        v = i * h
        if i == 0 or i == n_intgr:
            weight = 0.5
        else:
            weight = 1.0

        kappa1_plus_prime = kappa1_plus + varphi(v / m1)
        kappa2_plus_prime = kappa2_plus + varphi(v / m1)
        kappa1_minus_prime = kappa1_minus + varphi(v / m1)
        kappa2_minus_prime = kappa2_minus + varphi(v / m1)
        if (mu1 * kappa1_plus + mu2 * kappa2_plus) == 0 or (mu1 * kappa1_minus + mu2 * kappa2_minus) == 0:
            print(kappa1_plus, kappa2_plus, kappa1_minus, kappa2_minus, mu1, mu2)
        if k == 0:
            mu1_prime = (mu1 * kappa1_plus) / (mu1 * kappa1_plus + mu2 * kappa2_plus)
            mu2_prime = (mu2 * kappa2_plus) / (mu1 * kappa1_plus + mu2 * kappa2_plus)
        else:
            mu1_prime = (mu1 * kappa1_minus) / (mu1 * kappa1_minus + mu2 * kappa2_minus)
            mu2_prime = (mu2 * kappa2_minus) / (mu1 * kappa1_minus + mu2 * kappa2_minus)
        mu_prime = [mu1_prime, mu2_prime]
        kappa_prime = np.array([[kappa1_plus_prime, kappa1_minus_prime], [kappa2_plus_prime, kappa2_minus_prime]])
        y = (kappa_prime, mu_prime)
        integral_result += weight * w(u, y) * nu(v)

    integral_result *= h
    return integral_result

@njit
def integrand_H_k(y, v, k):
    # Extract parameters from kappa and mu
    x, d, kappa, mu = y 
    kappa1_plus, kappa2_plus, kappa1_minus, kappa2_minus = kappa[0, 0], kappa[1, 0], kappa[0, 1], kappa[1, 1]
    mu1, mu2 = mu[0], mu[1]
    kappa1_plus_prime = kappa1_plus + varphi(v / m1)
    kappa2_plus_prime = kappa2_plus + varphi(v / m1)
    kappa1_minus_prime = kappa1_minus + varphi(v / m1)
    kappa2_minus_prime = kappa2_minus + varphi(v / m1)
        
    if k == 0:
        mu1_prime = (mu1 * kappa1_plus) / (mu1 * kappa1_plus + mu2 * kappa2_plus)
        d_prime = (1 - nu_0) * Q(v) + d
    else:
        mu1_prime = (mu1 * kappa1_minus) / (mu1 * kappa1_minus + mu2 * kappa2_minus)
        d_prime = -(1 - nu_0) * Q(v) + d
        
    mu_prime = [mu1_prime, 1 - mu1_prime]
    kappa_prime = np.array([[kappa1_plus_prime, kappa1_minus_prime], [kappa2_plus_prime, kappa2_minus_prime]])
    
    y_prime = (x, d_prime, kappa_prime, mu_prime)
    return y_prime
    

def H_k_1_w(u, w, k, y):
    
    v_max = 130  # Upper limit for v, as np.inf cannot be used in practice      
    
    # Perform the adaptive integration using quad
    integral_result, _ = quad(
        lambda v: (w(u, integrand_H_k(y, v, k))) * nu(v),
        0, v_max,
        epsabs=1,  # Tighten absolute tolerance
        epsrel=1,  # Tighten relative tolerance
    )
    
    return integral_result


# Define the intervention operator

@njit
def gamma_operator(y, x_0):
    x, d, kappa, mu = y
    return (x - x_0, d - (1-nu_0)*Q(x_0), kappa, mu)

def M_value_function_test(u, w, y):#faster here
    # Example of an intervention operator
    if y[0] >= 0:
        result = minimize_scalar(lambda x_0 : -C(y[1], x_0) + nu_0*Q(x_0)*(y[0] - x_0) - w(u, gamma_operator(y, x_0)), bounds=(0, y[0]), method='bounded')
        return -result.fun, result.x, C(y[1], result.x), result.x
    else:
        return 0, y[0]

def M_value_function(u, w, y):#faster here
    # Example of an intervention operator
    if y[0] >= 0:
        result = minimize_scalar(lambda x_0 : -C(y[1], x_0) + nu_0*Q(x_0)*(y[0] - x_0) - w(u, gamma_operator(y, x_0)), bounds=(0, y[0]), method='bounded')
        return -result.fun, result.x
    else:
        return 0, result.x
    
def interior_integral_operator(t, u, w, y, pi):#faster here
    x, d, kappa, _ = y
    
    # Interpolate the pi values to create a continuous function for quad
    r_values = np.linspace(t, u, len(pi))
    pi_interp = interp1d(r_values, pi, kind='linear', fill_value="interpolate")

    def integrand(r):
        
        lambda_ik_values= (kappa - lambda_infty[:, np.newaxis]) * np.exp(-beta * (r - t))[:, np.newaxis] + lambda_infty[:, np.newaxis]
        
        # Interpolate pi_r at the point r
        pi_r = np.array([pi_interp(r), 1 - pi_interp(r)])
        
        y_prime_r = (x, d * np.exp(-rho * (r - t)), lambda_ik_values, pi_r)
        
        h0 = H_k_1_w(r, w, 0, y_prime_r) + nu_0*x*E_Q
        h1 = H_k_1_w(r, w, 1, y_prime_r) - nu_0*x*E_Q
        
        _, m_plus_value, m_minus_value, _, m_plus_value_h, m_minus_value_h = m_double(t, r-epsilon, r+epsilon, y)
        
        integrand_value_sum = -np.sum(
                (m_plus_value_h-m_plus_value)/(2*epsilon) * h0 +
                (m_minus_value_h-m_minus_value)/(2*epsilon) * h1
            )
        #print('test: ', t, r, y[-1], integrand_value_sum, h0, h1, (m_plus_value_h-m_plus_value)/(2*epsilon), (m_minus_value_h-m_minus_value)/(2*epsilon))
        return integrand_value_sum
    
    # Perform adaptive integration using quad over the range [t, u]
    integral_result, _ = quad(integrand, t, u, epsabs=1, epsrel=1)
    
    return integral_result


def I1(t, u, w, y):
    x, d, kappa, mu = y
    pi = res0.solve_ode_with_constraints((t, u), mu, kappa, n_intgr + 1).y
    
    pi_0 = np.array(pi[0])

    integral = interior_integral_operator(t, u, w, y, pi_0)
    
    lambda_ik_values_u = res0.lambda_ik(t, u, kappa)
    pi_u = np.array([pi[0][-1], 1-pi[0][-1]])
    y_prime_u = (x, d*np.exp(-rho*(u-t)), lambda_ik_values_u, pi_u)
    MV, _ = M_value_function(u, w, y_prime_u)
    return integral + MV *np.sum(m_simple(t, u, y))