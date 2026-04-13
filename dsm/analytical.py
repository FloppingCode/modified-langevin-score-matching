import math
import numpy as np
import torch
import torch.nn as nn


class AnalyticalGaussianMixtureScore(nn.Module):
    """Exact score function for a mixture of isotropic Gaussians.

    For mixture p(x) = (1/K) sum_k N(x; mu_k, sigma_data^2 I), the
    noised score at noise level sigma is:

        p_sigma(x) = (1/K) sum_k N(x; mu_k, (sigma_data^2 + sigma^2) I)
        nabla_x log p_sigma(x) = sum_k w_k(x) * (mu_k - x) / (sigma_data^2 + sigma^2)

    where w_k(x) = N(x; mu_k, ...) / sum_j N(x; mu_j, ...) are posterior
    responsibility weights, computed via log-softmax for numerical stability.

    Matches the ScoreNetwork.forward(x, sigma) interface so it can be
    passed directly to annealed_langevin_dynamics and plot_score_field.
    """

    def __init__(self, centers: torch.Tensor, sigma_data: float):
        """
        Args:
            centers: (K, data_dim) tensor of mixture component means.
            sigma_data: Per-component std dev (scalar, isotropic).
        """
        super().__init__()
        self.register_buffer("centers", centers)
        self.register_buffer("sigma_data_sq", torch.tensor(sigma_data**2))

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, data_dim) — query points.
            sigma: (batch, 1) — noise level.
        Returns:
            score: (batch, data_dim) — exact score vector.
        """
        var = self.sigma_data_sq + sigma**2  # (batch, 1)

        # diff: (batch, K, data_dim)
        diff = self.centers.unsqueeze(0) - x.unsqueeze(1)

        # log weights: -0.5 * ||x - mu_k||^2 / var
        log_weights = -0.5 * (diff**2).sum(dim=-1) / var  # (batch, K)
        weights = torch.softmax(log_weights, dim=-1)  # (batch, K)

        # score = sum_k w_k * (mu_k - x) / var
        score = (weights.unsqueeze(-1) * diff).sum(dim=1) / var  # (batch, data_dim)
        return score


def make_8gaussians_analytical_score() -> AnalyticalGaussianMixtureScore:
    """Create the analytical score for the 8gaussians dataset.

    Mirrors the exact centers and normalization from data.py:
    8 Gaussians at radius 2.0, std 0.1, then divided by 3.0.
    """
    centers = []
    for i in range(8):
        angle = 2 * math.pi * i / 8
        centers.append([2.0 * math.cos(angle) / 3.0, 2.0 * math.sin(angle) / 3.0])
    centers = torch.tensor(centers, dtype=torch.float32)
    sigma_data = 0.1 / 3.0
    return AnalyticalGaussianMixtureScore(centers, sigma_data)
