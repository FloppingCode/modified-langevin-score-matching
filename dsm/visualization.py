import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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


def animate_sampling(
    trajectories: torch.Tensor,
    real_data: Optional[torch.Tensor] = None,
    n_particles: int = 200,
    interval: int = 50,
    xlim: tuple = (-1.5, 1.5),
    ylim: tuple = (-1.5, 1.5),
    figsize: tuple = (7, 7),
    title: str = "Sampling Process",
    trail_length: int = 10,
    save_path: Optional[str] = None,
) -> FuncAnimation:
    """Animate particles moving from noise to data distribution.

    Args:
        trajectories: (n_frames, n_samples, 2) from any sampler.
        real_data: Optional target data to show as gray background.
        n_particles: Number of particles to animate (subsampled).
        interval: Milliseconds between frames.
        trail_length: Number of past positions to show as fading trail.
        save_path: If given, save as .gif.

    Returns:
        FuncAnimation. Display in Colab with:
            from IPython.display import HTML
            HTML(anim.to_html5_video())
    """
    traj_np = trajectories.detach().cpu().numpy()
    n_frames, n_samples, _ = traj_np.shape
    indices = np.random.choice(
        n_samples, size=min(n_particles, n_samples), replace=False
    )
    traj_sub = traj_np[:, indices, :]  # (n_frames, n_particles, 2)

    fig, ax = plt.subplots(figsize=figsize)

    if real_data is not None:
        data_np = real_data.detach().cpu().numpy()
        ax.scatter(
            data_np[:, 0], data_np[:, 1],
            s=1, alpha=0.15, color="gray", zorder=0,
        )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    txt = ax.set_title(f"{title} — frame 0/{n_frames}")

    # Current particle positions
    scat = ax.scatter(
        traj_sub[0, :, 0], traj_sub[0, :, 1],
        s=8, color="C0", alpha=0.7, zorder=5,
    )

    # Trail lines (one per particle)
    trail_lines = []
    for _ in range(len(indices)):
        (line,) = ax.plot([], [], color="C0", alpha=0.15, linewidth=0.5, zorder=2)
        trail_lines.append(line)

    def update(frame):
        # Update scatter positions
        scat.set_offsets(traj_sub[frame])

        # Update trails
        if trail_length > 0:
            start = max(0, frame - trail_length)
            for i, line in enumerate(trail_lines):
                line.set_data(
                    traj_sub[start : frame + 1, i, 0],
                    traj_sub[start : frame + 1, i, 1],
                )

        txt.set_text(f"{title} — frame {frame}/{n_frames}")
        return (scat, txt, *trail_lines)

    anim = FuncAnimation(
        fig, update, frames=n_frames, interval=interval, blit=True,
    )

    if save_path:
        from matplotlib.animation import PillowWriter
        anim.save(save_path, writer=PillowWriter(fps=max(1, 1000 // interval)))
        print(f"Animation saved to {save_path}")

    plt.close(fig)
    return anim


def display_animation(anim):
    """Display animation in Jupyter/Colab. Returns HTML object."""
    from IPython.display import HTML
    try:
        return HTML(anim.to_html5_video())
    except Exception:
        return HTML(anim.to_jshtml())
