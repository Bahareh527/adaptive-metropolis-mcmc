from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mcmc_research import adaptive_metropolis, correlated_gaussian_logpdf, random_walk_metropolis_nd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    covariance = np.array([[1.0, 0.92], [0.92, 1.0]])
    def target(point: np.ndarray) -> float:
        return correlated_gaussian_logpdf(point, covariance)
    initial = np.array([5.0, -5.0])
    results = {
        "Small random walk": random_walk_metropolis_nd(
            target, initial, 12_000, np.eye(2) * 0.05, np.random.default_rng(3100)
        ),
        "Large random walk": random_walk_metropolis_nd(
            target, initial, 12_000, np.eye(2) * 2.0, np.random.default_rng(3200)
        ),
        "Adaptive Metropolis": adaptive_metropolis(
            target, initial, 12_000, np.random.default_rng(3300), adapt_start=250
        ),
    }
    x = np.linspace(-4, 4, 160)
    xx, yy = np.meshgrid(x, x)
    inverse = np.linalg.inv(covariance)
    positions = np.dstack((xx, yy))
    density = np.exp(-0.5 * np.einsum("...i,ij,...j->...", positions, inverse, positions))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True, sharey=True)
    colors = ["#2878B5", "#E07A1F", "#2A9D58"]
    for axis, (name, result), color in zip(axes, results.items(), colors, strict=True):
        samples = result.samples[2_000:]
        axis.contour(xx, yy, density, levels=6, colors="#333333", linewidths=0.8)
        axis.scatter(samples[::8, 0], samples[::8, 1], s=7, alpha=0.25, color=color)
        axis.set_title(f"{name}\nacceptance = {result.acceptance_rate:.2f}")
        axis.set_xlabel("x")
        axis.set_xlim(-4, 4)
        axis.set_ylim(-4, 4)
    axes[0].set_ylabel("y")
    fig.suptitle("Proposal geometry controls MCMC efficiency", fontweight="bold")
    fig.tight_layout()
    output = ROOT / "docs" / "figures" / "adaptive_comparison.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
