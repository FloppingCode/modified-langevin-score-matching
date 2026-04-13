import math
import torch
from typing import Optional, Tuple, Union


@torch.no_grad()
def vanilla_langevin_dynamics(
    model,
    n_samples: int = 1000,
    data_dim: int = 2,
    n_steps: int = 1000,
    step_size: float = 1e-4,
    init_std: float = 2.0,
    device: str = "cuda",
    return_trajectories: bool = False,
    save_every: int = 1,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Standard (non-annealed) Langevin dynamics.

    x_{t+1} = x_t + (eps/2) * score(x_t) + sqrt(eps) * z_t

    No noise schedule — single constant step size. Used with KDE and
    single-sigma score models.
    """
    model.eval()
    x = torch.randn(n_samples, data_dim, device=device) * init_std
    dummy_sigma = torch.zeros(n_samples, 1, device=device)
    trajectories = [] if return_trajectories else None

    for t in range(n_steps):
        score = model(x, dummy_sigma)
        z = torch.randn_like(x)
        x = x + (step_size / 2) * score + math.sqrt(step_size) * z

        if return_trajectories and (t % save_every == 0 or t == n_steps - 1):
            trajectories.append(x.cpu().clone())

    if return_trajectories:
        return x, torch.stack(trajectories)
    return x


@torch.no_grad()
def annealed_langevin_dynamics(
    model,
    noise_schedule,
    n_samples: int = 1000,
    data_dim: int = 2,
    steps_per_sigma: int = 100,
    step_size_factor: float = 5e-5,
    device: str = "cuda",
    return_trajectories: bool = False,
    save_every: int = 1,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Annealed Langevin dynamics (Song & Ermon, 2019).

    Iterates through noise levels from sigma_max to sigma_min,
    running Langevin steps at each level.
    """
    model.eval()
    sigmas = noise_schedule.sigmas.to(device)
    sigma_min = sigmas[-1]
    x = torch.randn(n_samples, data_dim, device=device) * sigmas[0]
    trajectories = [] if return_trajectories else None
    step_count = 0

    for sigma_i in sigmas:
        alpha = step_size_factor * (sigma_i / sigma_min) ** 2
        sigma_batch = sigma_i.expand(n_samples, 1)

        for _ in range(steps_per_sigma):
            score = model(x, sigma_batch)
            z = torch.randn_like(x)
            x = x + (alpha / 2) * score + torch.sqrt(alpha) * z

            if return_trajectories and (step_count % save_every == 0):
                trajectories.append(x.cpu().clone())
            step_count += 1

    if return_trajectories:
        return x, torch.stack(trajectories)
    return x


@torch.no_grad()
def modified_langevin_dynamics(
    model,
    noise_schedule,
    correction_fn=None,
    n_samples: int = 1000,
    data_dim: int = 2,
    steps_per_sigma: int = 100,
    step_size_factor: float = 5e-5,
    device: str = "cuda",
    return_trajectories: bool = False,
    save_every: int = 1,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Annealed Langevin dynamics with a bias-correction term.

    Update rule:
        x_{t+1} = x_t + (alpha/2) * score + correction + sqrt(alpha) * z

    The correction_fn receives (x, score, sigma, alpha) and returns a
    tensor of shape (n_samples, data_dim) to add to the update.
    If correction_fn is None, falls back to standard annealed Langevin.
    """
    model.eval()
    sigmas = noise_schedule.sigmas.to(device)
    sigma_min = sigmas[-1]
    x = torch.randn(n_samples, data_dim, device=device) * sigmas[0]
    trajectories = [] if return_trajectories else None
    step_count = 0

    for sigma_i in sigmas:
        alpha = step_size_factor * (sigma_i / sigma_min) ** 2
        sigma_batch = sigma_i.expand(n_samples, 1)

        for _ in range(steps_per_sigma):
            score = model(x, sigma_batch)
            z = torch.randn_like(x)

            correction = torch.zeros_like(x)
            if correction_fn is not None:
                correction = correction_fn(x, score, sigma_i, alpha)

            x = x + (alpha / 2) * score + correction + torch.sqrt(alpha) * z

            if return_trajectories and (step_count % save_every == 0):
                trajectories.append(x.cpu().clone())
            step_count += 1

    if return_trajectories:
        return x, torch.stack(trajectories)
    return x
