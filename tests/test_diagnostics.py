import numpy as np
import pytest

from mcmc_research import autocorrelation, effective_sample_size, integrated_autocorrelation_time


def test_independent_samples_have_near_unit_iact() -> None:
    samples = np.random.default_rng(2).normal(size=20_000)
    assert integrated_autocorrelation_time(samples) == pytest.approx(1.0, abs=0.2)
    assert effective_sample_size(samples) > 0.75 * len(samples)


def test_correlated_samples_have_smaller_effective_size() -> None:
    rng = np.random.default_rng(3)
    values = np.empty(10_000)
    values[0] = rng.normal()
    for index in range(1, len(values)):
        values[index] = 0.9 * values[index - 1] + rng.normal(scale=np.sqrt(1 - 0.9**2))
    assert integrated_autocorrelation_time(values) > 8
    assert effective_sample_size(values) < len(values) / 5


def test_autocorrelation_validates_lag() -> None:
    with pytest.raises(ValueError):
        autocorrelation([1.0, 2.0], max_lag=2)

