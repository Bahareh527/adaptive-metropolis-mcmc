import numpy as np
import pytest

from mcmc_research import (
    adaptive_metropolis,
    correlated_gaussian_logpdf,
    lognormal_logpdf,
    multiplicative_metropolis_1d,
    random_walk_metropolis_1d,
)


def test_random_walk_is_deterministic() -> None:
    def target(x: float) -> float:
        return -0.5 * x**2
    first = random_walk_metropolis_1d(target, 0.0, 500, 1.0, np.random.default_rng(8))
    second = random_walk_metropolis_1d(target, 0.0, 500, 1.0, np.random.default_rng(8))
    assert np.array_equal(first.samples, second.samples)
    assert 0 < first.acceptance_rate < 1


def test_multiplicative_sampler_recovers_lognormal_mean() -> None:
    result = multiplicative_metropolis_1d(
        lambda x: float(lognormal_logpdf(x, mu=0.4, sigma=0.5)),
        1.0,
        25_000,
        1.0,
        np.random.default_rng(24),
    )
    expected = np.exp(0.4 + 0.5**2 / 2)
    assert np.mean(result.samples[2_500:]) == pytest.approx(expected, rel=0.08)
    assert np.all(result.samples > 0)


def test_adaptive_metropolis_learns_correlated_target() -> None:
    covariance = np.array([[1.0, 0.9], [0.9, 1.0]])
    result = adaptive_metropolis(
        lambda x: correlated_gaussian_logpdf(x, covariance),
        np.array([3.0, -3.0]),
        15_000,
        np.random.default_rng(31),
        adapt_start=200,
    )
    post = result.samples[2_000:]
    assert np.mean(post, axis=0) == pytest.approx(np.zeros(2), abs=0.12)
    assert np.corrcoef(post.T)[0, 1] == pytest.approx(0.9, abs=0.06)


@pytest.mark.parametrize(
    "call",
    [
        lambda: random_walk_metropolis_1d(lambda x: -x**2, 0.0, 0, 1.0, np.random.default_rng()),
        lambda: multiplicative_metropolis_1d(
            lambda x: -x**2, 0.0, 10, 1.0, np.random.default_rng()
        ),
        lambda: adaptive_metropolis(
            lambda x: -float(x @ x), np.zeros(2), 10, np.random.default_rng(), adapt_start=1
        ),
    ],
)
def test_invalid_sampler_inputs(call) -> None:
    with pytest.raises(ValueError):
        call()
