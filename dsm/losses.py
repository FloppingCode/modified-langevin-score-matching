import torch
import torch.nn as nn


def dsm_loss(
    model: nn.Module,
    x: torch.Tensor,
    noise_schedule,
    sigma_weighting: bool = False,
) -> torch.Tensor:
    """Denoising Score Matching loss (multi-noise-level).

    Samples sigma uniformly from the noise schedule for each batch item.
    When sigma_weighting=True, weights by sigma^2 to equalize contributions.
    """
    sigma = noise_schedule.sample_sigma(x.shape[0]).to(x.device)
    epsilon = torch.randn_like(x)
    x_tilde = x + sigma * epsilon
    target = -epsilon / sigma
    score_pred = model(x_tilde, sigma)
    per_sample = ((score_pred - target) ** 2).sum(dim=-1)

    if sigma_weighting:
        per_sample = sigma.squeeze(-1) ** 2 * per_sample

    return per_sample.mean()


def single_sigma_dsm_loss(
    model: nn.Module,
    x: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Denoising Score Matching loss at a single fixed noise level.

    Same math as dsm_loss but sigma is a fixed scalar, not sampled.
    Used for the single-sigma baseline.
    """
    sigma_t = torch.full((x.shape[0], 1), sigma, device=x.device)
    epsilon = torch.randn_like(x)
    x_tilde = x + sigma_t * epsilon
    target = -epsilon / sigma_t
    score_pred = model(x_tilde, sigma_t)
    return ((score_pred - target) ** 2).sum(dim=-1).mean()
