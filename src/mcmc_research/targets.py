from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


def lognormal_logpdf(x: ArrayLike, mu: float = 2.0, sigma: float = 0.75) -> NDArray[np.float64]:
    values = np.asarray(x, dtype=float)
    result = np.full_like(values, -np.inf)
    positive = values > 0
    z = (np.log(values[positive]) - mu) / sigma
    result[positive] = -np.log(values[positive] * sigma * math.sqrt(2 * math.pi)) - z**2 / 2
    return result


def lognormal_pdf(x: ArrayLike, mu: float = 2.0, sigma: float = 0.75) -> NDArray[np.float64]:
    return np.exp(lognormal_logpdf(x, mu, sigma))


def correlated_gaussian_logpdf(
    x: ArrayLike, covariance: ArrayLike | None = None
) -> float:
    values = np.asarray(x, dtype=float)
    cov = (
        np.array([[1.0, 0.92], [0.92, 1.0]])
        if covariance is None
        else np.asarray(covariance, dtype=float)
    )
    if cov.shape != (values.size, values.size):
        raise ValueError("covariance shape must match x.")
    sign, log_determinant = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("covariance must be positive definite.")
    quadratic = values @ np.linalg.solve(cov, values)
    normalization = values.size * math.log(2 * math.pi) + log_determinant
    return float(-0.5 * (normalization + quadratic))
