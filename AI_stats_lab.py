import numpy as np
from scipy.integrate import quad


# ==============================
# Q1 — EXPONENTIAL DISTRIBUTION
# ==============================

def exponential_pdf(x, lam):
    x = np.asarray(x)
    return np.where(x >= 0, lam * np.exp(-lam * x), 0.0)


def exponential_interval(a, b, lam):
    return np.exp(-lam * a) - np.exp(-lam * b)


def exponential_simulation(a, b, lam, N=100000):
    samples = np.random.exponential(scale=1/lam, size=N)
    return np.mean((samples > a) & (samples < b))


# ==============================
# Q2 — GAUSSIAN BAYESIAN MODEL
# ==============================

def gaussian_pdf(x, mu, sigma2):
    sigma = np.sqrt(sigma2)
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * \
           np.exp(-((x - mu) ** 2) / (2 * sigma2))


def posterior(x, mu_A, sigma2_A, mu_B, sigma2_B,
              prior_A, prior_B):

    f_A = gaussian_pdf(x, mu_A, sigma2_A)
    f_B = gaussian_pdf(x, mu_B, sigma2_B)

    evidence = f_A * prior_A + f_B * prior_B

    return (f_B * prior_B) / evidence
