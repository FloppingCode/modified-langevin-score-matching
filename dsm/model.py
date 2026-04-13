import math
import torch
import torch.nn as nn


class SinusoidalSigmaEmbedding(nn.Module):
    """Map scalar sigma to a vector via sinusoidal positional encoding.

    Sigma spans orders of magnitude (0.01 to 1.0); log-frequency encoding
    captures that scale naturally and provides a richer representation
    than raw concatenation.
    """

    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sigma: (batch, 1)
        Returns:
            embedding: (batch, embed_dim)
        """
        half = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=sigma.device).float() / half
        )
        args = sigma * freqs  # (batch, half)
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


class ScoreNetwork(nn.Module):
    """Noise-conditioned residual MLP for score estimation.

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
        """
        Args:
            x: (batch, data_dim) — noisy data points
            sigma: (batch, 1) — noise level for each point
        Returns:
            score: (batch, data_dim) — estimated score vector
        """
        sigma_emb = self.sigma_embed(sigma)
        sigma_feat = self.sigma_proj(sigma_emb)
        x_feat = self.input_proj(x)

        h = x_feat + sigma_feat
        for block in self.res_blocks:
            h = block(h)
        return self.output_proj(h)
