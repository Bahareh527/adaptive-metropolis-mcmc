"""Small, transparent MCMC implementations for teaching and experimentation."""

from .diagnostics import (
    autocorrelation,
    effective_sample_size,
    integrated_autocorrelation_time,
    running_mean,
    summarize_1d,
    summarize_nd,
)
from .samplers import (
    MCMCResult,
    adaptive_metropolis,
    multiplicative_metropolis_1d,
    random_walk_metropolis_1d,
    random_walk_metropolis_nd,
)
from .targets import correlated_gaussian_logpdf, lognormal_logpdf, lognormal_pdf

__all__ = [
    "MCMCResult",
    "adaptive_metropolis",
    "autocorrelation",
    "correlated_gaussian_logpdf",
    "effective_sample_size",
    "integrated_autocorrelation_time",
    "lognormal_logpdf",
    "lognormal_pdf",
    "multiplicative_metropolis_1d",
    "random_walk_metropolis_1d",
    "random_walk_metropolis_nd",
    "running_mean",
    "summarize_1d",
    "summarize_nd",
]

