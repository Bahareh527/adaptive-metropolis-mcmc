from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def autocorrelation(samples: ArrayLike, max_lag: int = 100) -> NDArray[np.float64]:
    """Estimate the normalized autocorrelation function through ``max_lag``."""
    values = np.asarray(samples, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("samples must be a one-dimensional array with at least two values.")
    if not 0 <= max_lag < values.size:
        raise ValueError("max_lag must lie between 0 and len(samples) - 1.")
    centered = values - values.mean()
    denominator = float(np.dot(centered, centered))
    if denominator == 0:
        return np.ones(max_lag + 1)
    result = np.empty(max_lag + 1)
    result[0] = 1.0
    for lag in range(1, max_lag + 1):
        result[lag] = np.dot(centered[:-lag], centered[lag:]) / denominator
    return result


def integrated_autocorrelation_time(samples: ArrayLike, max_lag: int = 200) -> float:
    """Estimate IACT with Geyer's initial-positive-pair truncation."""
    values = np.asarray(samples, dtype=float)
    lag = min(max_lag, len(values) - 1)
    acf = autocorrelation(values, lag)
    total = 0.0
    for index in range(1, len(acf) - 1, 2):
        pair = float(acf[index] + acf[index + 1])
        if pair <= 0:
            break
        total += pair
    return max(1.0, 1.0 + 2.0 * total)


def effective_sample_size(samples: ArrayLike, max_lag: int = 200) -> float:
    values = np.asarray(samples, dtype=float)
    return min(float(len(values)), len(values) / integrated_autocorrelation_time(values, max_lag))


def running_mean(samples: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(samples, dtype=float)
    return np.cumsum(values) / np.arange(1, len(values) + 1)


def summarize_1d(samples: ArrayLike, burn_in: int = 0) -> dict[str, float]:
    values = np.asarray(samples, dtype=float)[burn_in:]
    if values.size < 2:
        raise ValueError("At least two post-burn-in samples are required.")
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "median": float(np.median(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
        "ess": effective_sample_size(values),
        "iact": integrated_autocorrelation_time(values),
    }


def summarize_nd(samples: ArrayLike, burn_in: int = 0) -> dict[str, NDArray[np.float64]]:
    values = np.asarray(samples, dtype=float)[burn_in:]
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("Expected at least two post-burn-in multivariate samples.")
    return {
        "mean": values.mean(axis=0),
        "cov": np.cov(values.T),
        "ess": np.array([effective_sample_size(values[:, i]) for i in range(values.shape[1])]),
        "iact": np.array(
            [integrated_autocorrelation_time(values[:, i]) for i in range(values.shape[1])]
        ),
    }

