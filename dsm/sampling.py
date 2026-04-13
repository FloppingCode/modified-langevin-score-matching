import torch
from typing import Optional, Tuple, Union


@torch.no_grad()
def annealed_langevin_dynamics(
    model: torch.nn.Module,
    noise_schedule,
    n_samples: int = 1000,
    data_dim: int = 2,
    steps_per_sigma: int = 100,
    step_size_factor: float = 5e-5,
    device: str = "cuda",
    return_trajectories: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Sample via annealed Langevin dynamics (Song & Ermon, 2019).

    Algorithm:
        Initialize x ~ N(0, sigma_max^2 * I)
        For each sigma_i in [sigma_1, ..., sigma_L] (descending):
            alpha_i = step_size_factor * (sigma_i / sigma_L)^2
            For t = 1 to steps_per_sigma:
                z ~ N(0, I)
                x = x + (alpha_i / 2) * s_theta(x, sigma_i) + sqrt(alpha_i) * z

    Args:
        model: Trained ScoreNetwork.
        noise_schedule: GeometricNoiseSchedule instance.
        n_samples: Number of samples to generate.
        data_dim: Dimensionality of data space.
        steps_per_sigma: Langevin steps per noise level.
        step_size_factor: epsilon in the NCSN paper. Step size at sigma_i is
            epsilon * (sigma_i / sigma_min)^2. Default 5e-5 is tuned for toy 2D data.
        device: "cuda" or "cpu".
        return_trajectories: If True, also return intermediate states.

    Returns:
        samples: (n_samples, data_dim) final samples.
        If return_trajectories: (samples, trajectories) where trajectories
            is (num_levels * steps_per_sigma, n_samples, data_dim).
    """
    model.eval()
    sigmas = noise_schedule.sigmas.to(device)
    sigma_min = sigmas[-1]

    # Initialize from the prior at the largest noise level
    x = torch.randn(n_samples, data_dim, device=device) * sigmas[0]

    trajectories = [] if return_trajectories else None

    for sigma_i in sigmas:
        alpha = step_size_factor * (sigma_i / sigma_min) ** 2
        sigma_batch = sigma_i.expand(n_samples, 1)  # (n_samples, 1)

        for _ in range(steps_per_sigma):
            score = model(x, sigma_batch)
            z = torch.randn_like(x)
            x = x + (alpha / 2) * score + torch.sqrt(alpha) * z

            if return_trajectories:
                trajectories.append(x.cpu().clone())

    if return_trajectories:
        return x, torch.stack(trajectories)
    return x
