from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

LogDensity1D = Callable[[float], float]
LogDensityND = Callable[[NDArray[np.float64]], float]


@dataclass(frozen=True, slots=True)
class MCMCResult:
    samples: NDArray[np.float64]
    acceptance_rate: float


def _validate(n_steps: int, scale: float) -> None:
    if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps <= 0:
        raise ValueError("n_steps must be a positive integer.")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("The proposal scale must be finite and positive.")


def random_walk_metropolis_1d(
    target_logpdf: LogDensity1D,
    initial_state: float,
    n_steps: int,
    step_scale: float,
    rng: np.random.Generator,
) -> MCMCResult:
    """Metropolis sampler with a symmetric uniform random-walk proposal."""
    _validate(n_steps, step_scale)
    current = float(initial_state)
    current_logpdf = float(target_logpdf(current))
    if not np.isfinite(current_logpdf):
        raise ValueError("initial_state must have finite target log-density.")
    samples = np.empty(n_steps)
    accepted = 0
    for index in range(n_steps):
        candidate = current + rng.uniform(-step_scale, step_scale)
        candidate_logpdf = float(target_logpdf(candidate))
        if np.log(rng.uniform()) < min(0.0, candidate_logpdf - current_logpdf):
            current, current_logpdf = candidate, candidate_logpdf
            accepted += 1
        samples[index] = current
    return MCMCResult(samples, accepted / n_steps)


def multiplicative_metropolis_1d(
    target_logpdf: LogDensity1D,
    initial_state: float,
    n_steps: int,
    log_step_scale: float,
    rng: np.random.Generator,
) -> MCMCResult:
    """Positive-domain MH sampler with the required Hastings/Jacobian correction."""
    _validate(n_steps, log_step_scale)
    current = float(initial_state)
    if current <= 0:
        raise ValueError("initial_state must be positive for a multiplicative proposal.")
    current_logpdf = float(target_logpdf(current))
    samples = np.empty(n_steps)
    accepted = 0
    for index in range(n_steps):
        candidate = current * np.exp(rng.uniform(-log_step_scale, log_step_scale))
        candidate_logpdf = float(target_logpdf(candidate))
        log_ratio = candidate_logpdf - current_logpdf + np.log(candidate / current)
        if np.log(rng.uniform()) < min(0.0, log_ratio):
            current, current_logpdf = candidate, candidate_logpdf
            accepted += 1
        samples[index] = current
    return MCMCResult(samples, accepted / n_steps)


def random_walk_metropolis_nd(
    target_logpdf: LogDensityND,
    initial_state: NDArray[np.float64],
    n_steps: int,
    proposal_covariance: NDArray[np.float64],
    rng: np.random.Generator,
) -> MCMCResult:
    """Multivariate random-walk Metropolis with a Gaussian proposal."""
    _validate(n_steps, 1.0)
    current = np.asarray(initial_state, dtype=float).copy()
    covariance = np.asarray(proposal_covariance, dtype=float)
    if current.ndim != 1 or covariance.shape != (current.size, current.size):
        raise ValueError("proposal_covariance must match the one-dimensional state.")
    current_logpdf = float(target_logpdf(current))
    if not np.isfinite(current_logpdf):
        raise ValueError("initial_state must have finite target log-density.")
    samples = np.empty((n_steps, current.size))
    accepted = 0
    for index in range(n_steps):
        candidate = rng.multivariate_normal(current, covariance)
        candidate_logpdf = float(target_logpdf(candidate))
        if np.log(rng.uniform()) < min(0.0, candidate_logpdf - current_logpdf):
            current, current_logpdf = candidate, candidate_logpdf
            accepted += 1
        samples[index] = current
    return MCMCResult(samples, accepted / n_steps)


def adaptive_metropolis(
    target_logpdf: LogDensityND,
    initial_state: NDArray[np.float64],
    n_steps: int,
    rng: np.random.Generator,
    adapt_start: int = 250,
    initial_scale: float = 0.2,
    epsilon: float = 1e-6,
) -> MCMCResult:
    """Adaptive Metropolis using an online empirical proposal covariance."""
    _validate(n_steps, initial_scale)
    if adapt_start < 2 or epsilon <= 0:
        raise ValueError("adapt_start must be at least 2 and epsilon must be positive.")
    current = np.asarray(initial_state, dtype=float).copy()
    if current.ndim != 1:
        raise ValueError("initial_state must be one-dimensional.")
    dimension = current.size
    current_logpdf = float(target_logpdf(current))
    if not np.isfinite(current_logpdf):
        raise ValueError("initial_state must have finite target log-density.")
    samples = np.empty((n_steps, dimension))
    proposal_covariance = np.eye(dimension) * initial_scale**2
    empirical_mean = np.zeros(dimension)
    m2 = np.zeros((dimension, dimension))
    scaling = 2.38**2 / dimension
    accepted = 0

    for index in range(n_steps):
        candidate = rng.multivariate_normal(current, proposal_covariance)
        candidate_logpdf = float(target_logpdf(candidate))
        if np.log(rng.uniform()) < min(0.0, candidate_logpdf - current_logpdf):
            current, current_logpdf = candidate, candidate_logpdf
            accepted += 1
        samples[index] = current

        count = index + 1
        delta = current - empirical_mean
        empirical_mean += delta / count
        m2 += np.outer(delta, current - empirical_mean)
        if count >= adapt_start:
            empirical_covariance = m2 / (count - 1)
            proposal_covariance = scaling * (
                empirical_covariance + epsilon * np.eye(dimension)
            )

    return MCMCResult(samples, accepted / n_steps)

