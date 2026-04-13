import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks (used by ScoreNetwork)
# ---------------------------------------------------------------------------

class SinusoidalSigmaEmbedding(nn.Module):
    """Map scalar sigma to a vector via sinusoidal positional encoding."""

    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        half = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=sigma.device).float() / half
        )
        args = sigma * freqs
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    """Pre-norm residual block: LayerNorm -> SiLU -> Linear -> SiLU -> Linear + skip."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


# ---------------------------------------------------------------------------
# Score models
# ---------------------------------------------------------------------------

class ScoreNetwork(nn.Module):
    """Noise-conditioned residual MLP for score estimation (NCSN).

    Architecture:
        sigma -> SinusoidalEmbedding -> Linear -> sigma_feat
        x     -> Linear -> x_feat
        h = x_feat + sigma_feat   (additive conditioning)
        h -> [ResBlock] * num_res_blocks -> Linear -> score
    """

    def __init__(
        self,
        data_dim: int = 2,
        hidden_dim: int = 256,
        num_res_blocks: int = 3,
        sigma_embed_dim: int = 64,
    ):
        super().__init__()
        self.sigma_embed = SinusoidalSigmaEmbedding(sigma_embed_dim)
        self.sigma_proj = nn.Linear(sigma_embed_dim, hidden_dim)
        self.input_proj = nn.Linear(data_dim, hidden_dim)
        self.res_blocks = nn.ModuleList(
            [ResBlock(hidden_dim) for _ in range(num_res_blocks)]
        )
        self.output_proj = nn.Linear(hidden_dim, data_dim)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma_emb = self.sigma_embed(sigma)
        sigma_feat = self.sigma_proj(sigma_emb)
        x_feat = self.input_proj(x)
        h = x_feat + sigma_feat
        for block in self.res_blocks:
            h = block(h)
        return self.output_proj(h)


class SimpleScoreNetwork(nn.Module):
    """Plain MLP for single-sigma score estimation.

    No noise conditioning, no residual blocks, no LayerNorm.
    ~8k parameters — visibly simpler than ScoreNetwork's ~414k.
    """

    def __init__(self, data_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x: torch.Tensor, sigma: torch.Tensor = None) -> torch.Tensor:
        """Score at x. sigma is accepted but ignored."""
        return self.net(x)


class KDEScore(nn.Module):
    """Score estimate via kernel density estimation (no learning).

    Given data points {x_i}, the KDE score is:
        nabla log p_h(x) = sum_i w_i(x) * (x_i - x) / h^2
    where w_i = softmax(-0.5 * ||x - x_i||^2 / h^2).
    """

    def __init__(self, data: torch.Tensor, bandwidth: float):
        super().__init__()
        self.register_buffer("centers", data)
        self.register_buffer("h_sq", torch.tensor(bandwidth**2))

    def forward(self, x: torch.Tensor, sigma: torch.Tensor = None) -> torch.Tensor:
        """Score at x. sigma is accepted but ignored."""
        diff = self.centers.unsqueeze(0) - x.unsqueeze(1)  # (batch, N, D)
        log_w = -0.5 * (diff**2).sum(dim=-1) / self.h_sq  # (batch, N)
        w = torch.softmax(log_w, dim=-1)
        return (w.unsqueeze(-1) * diff).sum(dim=1) / self.h_sq


# ---------------------------------------------------------------------------
# Model registry for checkpoint save/load
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "ScoreNetwork": ScoreNetwork,
    "SimpleScoreNetwork": SimpleScoreNetwork,
}
