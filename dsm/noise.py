import math
import torch


class GeometricNoiseSchedule:
    """Geometric sequence of noise levels from sigma_max to sigma_min.

    Following Song & Ermon (2019), sigma_max should be large enough to
    cover the data range, and sigma_min small enough for fine detail.
    """

    def __init__(
        self,
        sigma_min: float = 0.01,
        sigma_max: float = 1.0,
        num_levels: int = 10,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigmas = torch.exp(
            torch.linspace(math.log(sigma_max), math.log(sigma_min), num_levels)
        )  # shape: (L,), descending

    def sample_sigma(self, batch_size: int) -> torch.Tensor:
        """Uniformly sample a noise level for each item in a batch.

        Returns:
            sigma: (batch_size, 1)
        """
        indices = torch.randint(0, len(self.sigmas), (batch_size,))
        return self.sigmas[indices].unsqueeze(-1)
