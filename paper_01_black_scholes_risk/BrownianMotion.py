"""
Generate sample paths of a standard Brownian motion.

The script is self-contained, reproducible and saves the figure
as both PNG and PDF.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Fast non-interactive backend

import matplotlib.pyplot as plt
import numpy as np


def simulate_brownian_paths(
    n_paths: int = 6,
    n_steps: int = 500,
    time_horizon: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate sample paths of a standard Brownian motion.

    Parameters
    ----------
    n_paths : int
        Number of Brownian paths to simulate.
    n_steps : int
        Number of time steps.
    time_horizon : float
        Final time T.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    times : np.ndarray
        Time grid of shape (n_steps + 1,).
    paths : np.ndarray
        Brownian paths of shape (n_paths, n_steps + 1).
    """
    rng = np.random.default_rng(seed)

    dt = time_horizon / n_steps
    times = np.linspace(0.0, time_horizon, n_steps + 1)

    increments = np.sqrt(dt) * rng.standard_normal(size=(n_paths, n_steps))
    paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(increments, axis=1)],
        axis=1,
    )

    return times, paths


def plot_brownian_paths(
    times: np.ndarray,
    paths: np.ndarray,
    output_dir: str | Path = "figures",
    filename: str = "brownian_paths",
) -> None:
    """
    Plot and save Brownian sample paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for path in paths:
        ax.plot(times, path, linewidth=1.4)

    ax.axhline(0.0, linewidth=0.8, linestyle="--")
    ax.set_title("Sample paths of a standard Brownian motion")
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$B_t$")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    fig.savefig(output_dir / f"{filename}.png", dpi=300)
    fig.savefig(output_dir / f"{filename}.pdf")

    plt.close(fig)


def main() -> None:
    times, paths = simulate_brownian_paths(
        n_paths=6,
        n_steps=500,
        time_horizon=1.0,
        seed=42,
    )

    plot_brownian_paths(
        times=times,
        paths=paths,
        output_dir="figures",
        filename="brownian_paths",
    )


if __name__ == "__main__":
    main()