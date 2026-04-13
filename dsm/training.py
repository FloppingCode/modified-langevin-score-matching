import torch
import torch.nn as nn
from typing import Dict, List
from .losses import dsm_loss


def train(
    model: nn.Module,
    dataloader,
    noise_schedule=None,
    loss_fn=None,
    n_epochs: int = 200,
    lr: float = 1e-3,
    device: str = "cuda",
    log_every: int = 20,
    sigma_weighting: bool = False,
) -> Dict[str, List[float]]:
    """Train a score network.

    Uses Adam + cosine annealing LR + gradient clipping.

    Either provide noise_schedule (uses dsm_loss) or a custom loss_fn
    that takes (model, batch_x) and returns a scalar loss.
    """
    if loss_fn is None:
        assert noise_schedule is not None, "Provide noise_schedule or loss_fn"
        _loss_fn = lambda m, x: dsm_loss(m, x, noise_schedule, sigma_weighting)
    else:
        _loss_fn = loss_fn

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history = {"loss": []}

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for (batch_x,) in dataloader:
            batch_x = batch_x.to(device)
            loss = _loss_fn(model, batch_x)

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
