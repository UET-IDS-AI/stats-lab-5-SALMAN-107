"""
Statistics Lab 5: Continuous Random Variables
Exponential Distribution and Bayesian Classification
"""

import numpy as np


# ============================================================================
# QUESTION 1: EXPONENTIAL DISTRIBUTION (CDF, PDF, PROBABILITY)
# ============================================================================

def exponential_pdf(x, lam=1):
    """
    Compute the PDF of an exponential distribution.
    
    Derivation from CDF F_X(x) = (1 - e^(-λx))u(x):
    f_X(x) = dF/dx = λ * e^(-λx) for x ≥ 0, and 0 for x < 0
    
    Args:
        x: value(s) at which to evaluate the PDF
        lam: rate parameter (λ > 0), default = 1
    
    Returns:
        PDF value(s) at x
    """
    x = np.asarray(x)
    # PDF is λ * e^(-λx) for x ≥ 0, else 0
    pdf = np.where(x >= 0, lam * np.exp(-lam * x), 0)
    return pdf


def exponential_interval_probability(a, b, lam=1):
    """
    Compute P(a < X < b) using the analytical CDF formula.
    
    For exponential distribution:
    F_X(x) = 1 - e^(-λx)
    P(a < X < b) = F_X(b) - F_X(a)
                 = (1 - e^(-λb)) - (1 - e^(-λa))
                 = e^(-λa) - e^(-λb)
    
    Args:
        a: lower bound
        b: upper bound
        lam: rate parameter, default = 1
    
    Returns:
        Probability P(a < X < b)
    """
    prob = np.exp(-lam * a) - np.exp(-lam * b)
    return prob


def simulate_exponential_probability(a, b, lam=1, n_samples=100000):
    """
    Estimate P(a < X < b) using Monte Carlo simulation.
    
    Steps:
    1. Generate n_samples from exponential distribution with rate λ
    2. Count how many fall in the interval (a, b)
    3. Estimate probability as count/n_samples
    
    Args:
        a: lower bound
        b: upper bound
        lam: rate parameter, default = 1
        n_samples: number of samples for simulation, default = 100000
    
    Returns:
        Estimated probability
    """
    # Generate exponential samples: X ~ Exp(λ)
    samples = np.random.exponential(scale=1/lam, size=n_samples)
    
    # Count samples in interval (a, b)
    in_interval = np.sum((samples > a) & (samples < b))
    
    # Estimate probability
    estimated_prob = in_interval / n_samples
    return estimated_prob


# ============================================================================
# QUESTION 2: BAYESIAN CLASSIFICATION (GAUSSIAN MODEL)
# ============================================================================

def gaussian_pdf(x, mean, variance):
    """
    Compute the PDF of a Gaussian (normal) distribution.
    
    Formula: f(x) = (1 / √(2π*σ)) * exp(-(x - μ)² / σ²)
    
    where σ = sqrt(variance)
    
    Args:
        x: value(s) at which to evaluate the PDF
        mean: mean (μ)
        variance: parameter σ (used in exponent as σ² and denominator as √(2π)*σ)
    
    Returns:
        PDF value(s) at x
    """
    x = np.asarray(x)
    variance = float(variance)
    
    # Gaussian PDF formula: (1 / (√(2π)*σ)) * exp(-(x-μ)²/σ²)
    numerator = np.exp(-((x - mean) ** 2) / (variance ** 2))
    denominator = np.sqrt(2 * np.pi) * variance
    pdf = numerator / denominator
    
    return pdf


def posterior_probability(time):
    """
    Compute P(B | X = time) using Bayes rule.
    
    Given:
    - Group A (fast): X ~ N(40, 4), P(A) = 0.3
    - Group B (slow): X ~ N(45, 4), P(B) = 0.7
    - Variance = 4, so parameter σ = sqrt(4) = 2
    
    Bayes theorem:
    P(B | X) = P(X | B) * P(B) / [P(X | A) * P(A) + P(X | B) * P(B)]
    
    Args:
        time: observed finishing time
    
    Returns:
        Posterior probability P(B | X = time)
    """
    # Group parameters
    mean_A = 40
    var_A = 4
    sigma_A = np.sqrt(var_A)  # σ = sqrt(variance)
    
    mean_B = 45
    var_B = 4
    sigma_B = np.sqrt(var_B)  # σ = sqrt(variance)
    
    p_A = 0.3
    p_B = 0.7
    
    # Compute likelihoods
    likelihood_A = gaussian_pdf(time, mean_A, sigma_A)
    likelihood_B = gaussian_pdf(time, mean_B, sigma_B)
    
    # Apply Bayes theorem
    numerator = likelihood_B * p_B
    denominator = likelihood_A * p_A + likelihood_B * p_B
    posterior = numerator / denominator
    
    return posterior
