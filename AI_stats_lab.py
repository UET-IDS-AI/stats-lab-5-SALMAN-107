import numpy as np


# -------------------------------------------------
# Question 1 – Exponential Distribution
# -------------------------------------------------

def exponential_pdf(x, lam=1):
    """
    Return PDF of exponential distribution.

    Derived from CDF: F(x) = (1 - e^{-λx}) * u(x)
    PDF: f(x) = dF/dx = λ * e^{-λx}  for x >= 0
                         0              for x < 0
    """
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0, lam * np.exp(-lam * x), 0.0)


def exponential_interval_probability(a, b, lam=1):
    """
    Compute P(a < X < b) using the analytical CDF formula.

    P(a < X < b) = F(b) - F(a)
                 = (1 - e^{-λb}) - (1 - e^{-λa})
                 = e^{-λa} - e^{-λb}
    """
    return np.exp(-lam * a) - np.exp(-lam * b)


def simulate_exponential_probability(a, b, lam=1, n=100_000):
    """
    Simulate exponential samples and estimate P(a < X < b).

    Uses numpy to draw n samples from Exp(λ), then counts
    the fraction that fall in the open interval (a, b).
    """
    samples = np.random.exponential(scale=1.0 / lam, size=n)
    return np.mean((samples > a) & (samples < b))


# -------------------------------------------------
# Question 2 – Bayesian Classification
# -------------------------------------------------

def gaussian_pdf(x, mu, sigma):
    """
    Return Gaussian (Normal) PDF evaluated at x.

    f(x) = 1 / (σ√(2π)) * exp( -(x - μ)² / (2σ²) )
    """
    return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )


def posterior_probability(time):
    """
    Compute P(B | X = time) using Bayes' rule.

    Priors:   P(A) = 0.3,  P(B) = 0.7
    Likelihoods:
        f(x | A) ~ N(μ=40, σ=2)
        f(x | B) ~ N(μ=45, σ=2)

    Note: the problem states variance = 4, so σ = √4 = 2.

    Bayes' rule:
        P(B | X) = P(B) * f(X|B) / [P(A)*f(X|A) + P(B)*f(X|B)]
    """
    prior_A, prior_B = 0.3, 0.7
    # N(μ, 4) uses 4 as the exponent denominator: exp(-(x-μ)²/4)
    # → 2σ² = 4 → σ = √2
    sigma = np.sqrt(2)

    likelihood_A = gaussian_pdf(time, mu=40, sigma=sigma)
    likelihood_B = gaussian_pdf(time, mu=45, sigma=sigma)

    evidence = prior_A * likelihood_A + prior_B * likelihood_B
    return (prior_B * likelihood_B) / evidence


def simulate_posterior_probability(time, n=100_000):
    """
    Estimate P(B | X ≈ time) via simulation.

    Strategy (likelihood-weighted / rejection-style):
      1. Draw n swimmers; assign group A with P=0.3, group B with P=0.7.
      2. Sample finishing times from the corresponding Gaussian.
      3. Keep only swimmers whose time falls within a narrow window
         around `time` (± tolerance).
      4. Among those, compute the fraction that belong to group B.

    This mimics Bayes' rule empirically via conditional counting.
    """
    tolerance = 0.5     # acceptance window around the observed time

    # Assign groups: 0 → A, 1 → B
    groups  = np.random.choice([0, 1], size=n, p=[0.3, 0.7])

    sigma   = np.sqrt(2)
    times_A = np.random.normal(loc=40, scale=sigma, size=n)
    times_B = np.random.normal(loc=45, scale=sigma, size=n)

    # Each swimmer's actual simulated finishing time
    finishing_times = np.where(groups == 0, times_A, times_B)

    # Retain only those close to the observed time
    mask = np.abs(finishing_times - time) < tolerance
    if mask.sum() == 0:
        return np.nan   # No samples near this time; increase n or tolerance

    return np.mean(groups[mask] == 1)
