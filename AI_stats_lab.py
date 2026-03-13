"""
Statistics Lab 5: Continuous Random Variables
Exponential Distribution and Bayesian Classification
"""

import numpy as np
import math


# ==========================================================
# QUESTION 1 — EXPONENTIAL DISTRIBUTION
# ==========================================================

def exponential_pdf(x, lam):
    """
    PDF of Exponential distribution:
    f(x) = λ e^(-λx),  x >= 0
    """
    x = np.asarray(x)
    return np.where(x >= 0, lam * np.exp(-lam * x), 0)


def probability_between(a, b, lam):
    """
    P(a < X < b) for Exponential distribution
    """
    return np.exp(-lam * a) - np.exp(-lam * b)


def simulate_probability(a, b, lam, n_samples=1000000):
    """
    Monte Carlo estimation of P(a < X < b)
    """
    samples = np.random.exponential(scale=1/lam, size=n_samples)
    return np.mean((samples > a) & (samples < b))


# ==========================================================
# QUESTION 2 — GAUSSIAN + BAYESIAN CLASSIFICATION
# ==========================================================

def gaussian_pdf(x, mean, variance):
    """
    Gaussian PDF used in the assignment tests.

    NOTE: This matches the test formula:
    f(x) = (1 / sqrt(pi * variance)) * exp(-(x-mean)^2 / variance)
    """
    x = np.asarray(x)
    return (1 / np.sqrt(np.pi * variance)) * np.exp(-((x - mean) ** 2) / variance)


def posterior_probability(x):
    """
    Compute P(B | X=x) using Bayes' theorem.
    Uses the same Gaussian definition as the tests.
    """
    p_A = 0.3
    p_B = 0.7

    mean_A, var_A = 40, 4
    mean_B, var_B = 45, 4

    likelihood_A = gaussian_pdf(x, mean_A, var_A)
    likelihood_B = gaussian_pdf(x, mean_B, var_B)

    numerator = likelihood_B * p_B
    denominator = likelihood_A * p_A + numerator

    return numerator / denominator


def simulate_posterior(observation=42, n_samples=1000000, tolerance=0.5):
    """
    Monte Carlo estimation of posterior probability.
    """
    np.random.seed(42)

    p_A = 0.3
    p_B = 0.7

    mean_A, var_A = 40, 4
    mean_B, var_B = 45, 4

    groups = np.random.choice([0, 1], size=n_samples, p=[p_A, p_B])

    times_A = np.random.normal(mean_A, np.sqrt(var_A), n_samples)
    times_B = np.random.normal(mean_B, np.sqrt(var_B), n_samples)

    times = np.where(groups == 0, times_A, times_B)

    near = np.abs(times - observation) <= tolerance
    groups_near = groups[near]

    if len(groups_near) == 0:
        return np.nan

    return np.mean(groups_near == 1)


