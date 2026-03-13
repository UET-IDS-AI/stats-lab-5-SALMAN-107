import numpy as np
from scipy.integrate import quad
from scipy.stats import norm


###############################################################################
#                      PROBLEM 1: EXPONENTIAL DISTRIBUTION
###############################################################################

# MATHEMATICAL DERIVATION:
# ========================
# Given CDF: F_X(x) = (1 - e^(-λx))u(x)
#
# To find PDF, we take derivative with respect to x:
# f_X(x) = dF_X(x)/dx = d/dx[(1 - e^(-λx))]
#        = 0 - (-λ)e^(-λx)
#        = λe^(-λx)  for x >= 0
#        = 0         for x < 0
#
# This is the standard exponential distribution PDF.


def exponential_pdf(x, lam=1):
   
    # Handle scalar and array inputs
    x = np.asarray(x)
    
    # PDF is 0 for negative values, λe^(-λx) for x >= 0
    result = np.where(x >= 0, lam * np.exp(-lam * x), 0)
    
    return result


def exponential_interval_probability(a, b, lam=1):
   
    # Analytical solution: e^(-λa) - e^(-λb)
    prob_analytical = np.exp(-lam * a) - np.exp(-lam * b)
    
    return prob_analytical


def exponential_interval_probability_numerical(a, b, lam=1):
   
    # Define integrand: λe^(-λx)
    integrand = lambda x: lam * np.exp(-lam * x)
    
    # Numerical integration using adaptive quadrature
    prob_numerical, error = quad(integrand, a, b)
    
    return prob_numerical, error


def simulate_exponential_probability(a, b, lam=1, n=100000):
   
    # Generate n samples from exponential distribution with rate lam
    # numpy.random.exponential uses scale parameter = 1/lambda
    samples = np.random.exponential(scale=1/lam, size=n)
    
    # Count how many samples fall in interval (a, b)
    count_in_interval = np.sum((samples > a) & (samples < b))
    
    # Probability estimation
    prob_simulated = count_in_interval / n
    
    return prob_simulated


###############################################################################
#                   PROBLEM 2: BAYESIAN CLASSIFICATION
###############################################################################

# PARAMETERS:
# ===========
# Group A: X ~ N(μ_A = 40, σ²_A = 4)  → σ_A = 2
# Group B: X ~ N(μ_B = 45, σ²_B = 4)  → σ_B = 2
# Prior:   P(A) = 0.3, P(B) = 0.7
# Observation: X = 42

# MATHEMATICAL DERIVATION:
# ========================
# Gaussian PDF (manual implementation):
# f(x | μ, σ²) = (1 / √(2πσ²)) * exp(-(x-μ)² / (2σ²))
#
# BAYES THEOREM:
# P(B | X) = [f(X|B) * P(B)] / [f(X|A)*P(A) + f(X|B)*P(B)]
#
# Where:
# - f(X|A) = likelihood of observation under Group A (Gaussian PDF at X with μ_A, σ²_A)
# - f(X|B) = likelihood of observation under Group B (Gaussian PDF at X with μ_B, σ²_B)
# - P(A) = prior probability of Group A
# - P(B) = prior probability of Group B


def gaussian_pdf(x, mu, sigma):
   
    # Compute normalization constant: 1 / (√(2π) * σ)
    normalization = 1.0 / (np.sqrt(2 * np.pi) * sigma)
    
    # Compute exponential term: exp(-(x-μ)² / (2σ²))
    exponent = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    
    # Combine: f(x) = normalization × exponent
    pdf_value = normalization * exponent
    
    return pdf_value


def posterior_probability(x_obs, mu_A=40, sigma2_A=4, mu_B=45, sigma2_B=4, 
                         prior_A=0.3, prior_B=0.7):
    
    # Compute simplified likelihoods (without normalization constant)
    # L(X|A) ∝ exp(-(x-μ_A)²/σ²_A)
    likelihood_A = np.exp(-((x_obs - mu_A) ** 2) / sigma2_A)
    
    # L(X|B) ∝ exp(-(x-μ_B)²/σ²_B)
    likelihood_B = np.exp(-((x_obs - mu_B) ** 2) / sigma2_B)
    
    # Apply Bayes' theorem
    # P(B|X) = [L(X|B)*P(B)] / [L(X|A)*P(A) + L(X|B)*P(B)]
    evidence = likelihood_A * prior_A + likelihood_B * prior_B
    posterior_B = (likelihood_B * prior_B) / evidence
    
    return posterior_B


def simulate_posterior_probability(x_obs, mu_A=40, sigma2_A=4, mu_B=45, sigma2_B=4,
    
    
    # Approach: Repeat the analytical Bayes' rule computation many times
    # with small perturbations to estimate the posterior robustly
    
    posteriors_B = np.zeros(n)
    sigma_A = np.sqrt(sigma2_A)
    sigma_B = np.sqrt(sigma2_B)
    
    for i in range(n):
        # Add small random noise to parameters (Bayesian Bootstrap approach)
        # to simulate uncertainty and get a distribution of posteriors
        
        # Small jitter to avoid deterministic repetition
        eps_mu = 0.01  # Small perturbation size
        mu_A_perturbed = mu_A + np.random.normal(0, eps_mu)
        mu_B_perturbed = mu_B + np.random.normal(0, eps_mu)
        
        # Compute likelihoods with perturbed parameters
        # gaussian_pdf now takes standard deviation, not variance
        lik_A = gaussian_pdf(x_obs, mu_A_perturbed, sigma_A)
        lik_B = gaussian_pdf(x_obs, mu_B_perturbed, sigma_B)
        
        # Apply Bayes' theorem
        evidence = lik_A * prior_A + lik_B * prior_B
        posterior_B = (lik_B * prior_B) / evidence
        
        posteriors_B[i] = posterior_B
    
    # Return the average posterior across all trials
    simulated_posterior = np.mean(posteriors_B)
    
    return simulated_posterior


###############################################################################
#                              MAIN EXECUTION
###############################################################################

if __name__ == "__main__":
    
    print("="*80)
    print(" STATISTICS LAB 5: COMPLETE SOLUTION")
    print("="*80)
