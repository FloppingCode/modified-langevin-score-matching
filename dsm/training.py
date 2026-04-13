import torch
import torch.nn as nn
from typing import Dict, List


def dsm_loss(
    model: nn.Module,
    x: torch.Tensor,
    noise_schedule,
) -> torch.Tensor:
    """Denoising Score Matching loss.

    1. Sample sigma ~ Uniform({sigma_1, ..., sigma_L}) per item
    2. Sample epsilon ~ N(0, I)
    3. x_tilde = x + sigma * epsilon
    4. target = -epsilon / sigma
    5. Loss = ||s_theta(x_tilde, sigma) - target||^2

    Args:
        model: Score network s_theta(x, sigma).
        x: (batch, dim) clean data.
        noise_schedule: GeometricNoiseSchedule instance.

    Returns:
        Scalar loss (mean over batch).
    """
    sigma = noise_schedule.sample_sigma(x.shape[0]).to(x.device)  # (batch, 1)
    epsilon = torch.randn_like(x)  # (batch, dim)
    x_tilde = x + sigma * epsilon
    target = -epsilon / sigma
    score_pred = model(x_tilde, sigma)
    loss = ((score_pred - target) ** 2).sum(dim=-1).mean()
    return loss


def train(
    model: nn.Module,
    dataloader,
    noise_schedule,
    n_epochs: int = 200,
    lr: float = 1e-3,
    device: str = "cuda",
    log_every: int = 20,
) -> Dict[str, List[float]]:
    """Train the score network with DSM.

    Uses Adam + cosine annealing LR + gradient clipping.

    Args:
        model: ScoreNetwork instance.
        dataloader: DataLoader yielding (batch_x,) tuples.
        noise_schedule: GeometricNoiseSchedule instance.
        n_epochs: Number of training epochs.
        lr: Initial learning rate.
        device: "cuda" or "cpu".
        log_every: Print loss every N epochs.

    Returns:
        Dictionary with key "loss" mapping to list of per-epoch losses.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history = {"loss": []}

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for (batch_x,) in dataloader:
            batch_x = batch_x.to(device)
            loss = dsm_loss(model, batch_x, noise_schedule)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        history["loss"].append(avg_loss)

        if epoch % log_every == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch:4d}/{n_epochs}  loss={avg_loss:.4f}")

    return history
