import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def plot_samples(
    real: torch.Tensor,
    generated: torch.Tensor,
    title: str = "Real vs Generated",
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """Side-by-side scatter plot of real and generated 2D samples."""
    real_np = real.detach().cpu().numpy()
    gen_np = generated.detach().cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.scatter(real_np[:, 0], real_np[:, 1], s=1, alpha=0.5)
    ax1.set_title("Real Data")
    ax1.set_aspect("equal")
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)

    ax2.scatter(gen_np[:, 0], gen_np[:, 1], s=1, alpha=0.5, color="C1")
    ax2.set_title("Generated Samples")
    ax2.set_aspect("equal")
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)

    fig.suptitle(title)
    fig.tight_layout()
    return fig


@torch.no_grad()
def plot_score_field(
    model: torch.nn.Module,
    sigma: float,
    xlim: tuple = (-1.5, 1.5),
    ylim: tuple = (-1.5, 1.5),
    grid_size: int = 30,
    device: str = "cuda",
    data: Optional[torch.Tensor] = None,
) -> plt.Figure:
    """Quiver plot of the learned score field at a given noise level.

    This is the primary diagnostic for DSM on 2D data — arrows should
    point toward high-density regions.
    """
    model.eval()
    xs = np.linspace(xlim[0], xlim[1], grid_size)
    ys = np.linspace(ylim[0], ylim[1], grid_size)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=-1)

    grid_t = torch.tensor(grid, dtype=torch.float32, device=device)
    sigma_t = torch.full((grid_t.shape[0], 1), sigma, device=device)

    scores = model(grid_t, sigma_t).cpu().numpy()
    u = scores[:, 0].reshape(grid_size, grid_size)
    v = scores[:, 1].reshape(grid_size, grid_size)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    if data is not None:
        data_np = data.detach().cpu().numpy()
        ax.scatter(data_np[:, 0], data_np[:, 1], s=1, alpha=0.2, color="gray", zorder=0)

    # Color arrows by magnitude
    magnitude = np.sqrt(u**2 + v**2)
    ax.quiver(xx, yy, u, v, magnitude, cmap="viridis", alpha=0.8, scale=None)
    ax.set_title(f"Score field at σ = {sigma:.4f}")
    ax.set_aspect("equal")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    fig.tight_layout()
    return fig


def plot_training_curves(history: dict, figsize: tuple = (8, 4)) -> plt.Figure:
    """Plot loss vs epoch."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(history["loss"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("DSM Loss")
    ax.set_title("Training Loss")
    ax.set_yscale("log")
    fig.tight_layout()
    return fig


@torch.no_grad()
def plot_score_comparison(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    sigma: float,
    label_a: str = "Model A",
    label_b: str = "Model B",
    xlim: tuple = (-1.5, 1.5),
    ylim: tuple = (-1.5, 1.5),
    grid_size: int = 30,
    device: str = "cuda",
    data: Optional[torch.Tensor] = None,
) -> plt.Figure:
    """Side-by-side quiver plots comparing two score models at a given sigma."""
    model_a.eval()
    model_b.eval()

    xs = np.linspace(xlim[0], xlim[1], grid_size)
    ys = np.linspace(ylim[0], ylim[1], grid_size)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=-1)

    grid_t = torch.tensor(grid, dtype=torch.float32, device=device)
    sigma_t = torch.full((grid_t.shape[0], 1), sigma, device=device)

    scores_a = model_a(grid_t, sigma_t).cpu().numpy()
    scores_b = model_b(grid_t, sigma_t).cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, scores, label in [(ax1, scores_a, label_a), (ax2, scores_b, label_b)]:
        u = scores[:, 0].reshape(grid_size, grid_size)
        v = scores[:, 1].reshape(grid_size, grid_size)
        mag = np.sqrt(u**2 + v**2)

        if data is not None:
            data_np = data.detach().cpu().numpy()
            ax.scatter(data_np[:, 0], data_np[:, 1], s=1, alpha=0.2, color="gray", zorder=0)

        ax.quiver(xx, yy, u, v, mag, cmap="viridis", alpha=0.8)
        ax.set_title(f"{label} (σ={sigma:.4f})")
        ax.set_aspect("equal")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    fig.tight_layout()
    return fig


def plot_sampling_trajectory(
    trajectories: torch.Tensor,
    real_data: Optional[torch.Tensor] = None,
    n_traces: int = 50,
    figsize: tuple = (7, 7),
) -> plt.Figure:
    """Plot the path of sampled points through the denoising process.

    Args:
        trajectories: (num_steps, n_samples, 2) from annealed_langevin_dynamics.
        real_data: Optional real data to overlay.
        n_traces: Number of particle traces to draw.
    """
    traj_np = trajectories.numpy()
    n_steps, n_samples, _ = traj_np.shape
    indices = np.random.choice(n_samples, size=min(n_traces, n_samples), replace=False)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if real_data is not None:
        data_np = real_data.detach().cpu().numpy()
        ax.scatter(data_np[:, 0], data_np[:, 1], s=1, alpha=0.1, color="gray", zorder=0)

    for idx in indices:
        trace = traj_np[:, idx, :]
        # Color from light to dark as sampling progresses
        ax.plot(trace[:, 0], trace[:, 1], alpha=0.3, linewidth=0.5, color="C0")
        ax.scatter(trace[-1, 0], trace[-1, 1], s=10, color="C1", zorder=5)

    ax.set_title("Sampling Trajectories")
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    fig.tight_layout()
    return fig
