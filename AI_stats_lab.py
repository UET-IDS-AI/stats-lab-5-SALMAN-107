"""
Statistics Lab 5: Continuous Random Variables
Exponential Distribution and Bayesian Classification
"""

import numpy as np
from scipy import stats
import math


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
        variance: variance parameter (σ)
    
    Returns:
        PDF value(s) at x
    """
    x = np.asarray(x)
    variance = float(variance)
    
    # Gaussian PDF formula
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
    - Variance = 4, so standard deviation sigma = sqrt(4) = 2
    
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
    sigma_A = np.sqrt(var_A)  # sigma = sqrt(variance)
    
    mean_B = 45
    var_B = 4
    sigma_B = np.sqrt(var_B)  # sigma = sqrt(variance)
    
    p_A = 0.3
    p_B = 0.7
    
    # Compute likelihoods (note: gaussian_pdf parameter is sigma, not variance)
    likelihood_A = gaussian_pdf(time, mean_A, sigma_A)
    likelihood_B = gaussian_pdf(time, mean_B, sigma_B)
    
    # Apply Bayes theorem
    numerator = likelihood_B * p_B
    denominator = likelihood_A * p_A + likelihood_B * p_B
    posterior = numerator / denominator
    
    return posterior


def probability_between(a, b, lam):
    return np.exp(-lam * a) - np.exp(-lam * b)


def simulate_probability(a, b, lam, n_samples=1000000, seed=42):
    np.random.seed(seed)
    samples = np.random.exponential(scale=1/lam, size=n_samples)
    in_interval = np.sum((samples > a) & (samples < b))
    return in_interval / n_samples


def solve_question_1(a=2, b=5, lam=1, n_samples=1000000):
    analytical_prob = probability_between(a, b, lam)
    expected_prob = np.exp(-a) - np.exp(-b)
    simulated_prob = simulate_probability(a, b, lam, n_samples)
    error = abs(analytical_prob - simulated_prob)

    return {
        "analytical": analytical_prob,
        "expected": expected_prob,
        "simulated": simulated_prob,
        "error": error
    }


# =========================
# QUESTION 2
# =========================

def gaussian_pdf(x, mean, variance):
    x = np.asarray(x)
    numerator = np.exp(-((x - mean) ** 2) / (2 * variance))
    denominator = np.sqrt(2 * np.pi * variance)
    return numerator / denominator


def simulate_swimmers(n_samples=1000000, observation=42, tolerance=0.5, seed=42):
    np.random.seed(seed)

    # Priors
    p_A = 0.3
    p_B = 0.7

    # Group distributions
    mean_A, var_A = 40, 4
    mean_B, var_B = 45, 4

    groups = np.random.choice([0, 1], size=n_samples, p=[p_A, p_B])

    times_A = np.random.normal(mean_A, np.sqrt(var_A), n_samples)
    times_B = np.random.normal(mean_B, np.sqrt(var_B), n_samples)
    times = np.where(groups == 0, times_A, times_B)

    near_obs = np.abs(times - observation) <= tolerance
    groups_near = groups[near_obs]

    if len(groups_near) == 0:
        return np.nan

    count_B = np.sum(groups_near == 1)
    return count_B / len(groups_near)


def solve_question_2(observation=42, n_samples=1000000, tolerance=0.5):
    # Priors
    p_A = 0.3
    p_B = 0.7

    # Means and variances
    mean_A, var_A = 40, 4
    mean_B, var_B = 45, 4

    # Likelihoods
    likelihood_A = gaussian_pdf(observation, mean_A, var_A)
    likelihood_B = gaussian_pdf(observation, mean_B, var_B)

    # Analytical posterior
    numerator = likelihood_B * p_B
    denominator = likelihood_A * p_A + likelihood_B * p_B
    analytical_posterior = numerator / denominator

    # Simulated posterior
    simulated_posterior = simulate_swimmers(n_samples=n_samples, observation=observation, tolerance=tolerance)

    error = abs(analytical_posterior - simulated_posterior)

    return {
        "likelihood_A": likelihood_A,
        "likelihood_B": likelihood_B,
        "analytical_posterior": analytical_posterior,
        "simulated_posterior": simulated_posterior,
        "error": error
    }


# =========================
# MAIN EXECUTION
# =========================

if __name__ == "__main__":
    # Solve Question 1
    q1_results = solve_question_1()
    # Solve Question 2
    q2_results = solve_question_2()
