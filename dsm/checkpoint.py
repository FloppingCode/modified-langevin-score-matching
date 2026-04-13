import os
import torch
from .model import ScoreNetwork


def save_checkpoint(
    model: ScoreNetwork,
    noise_schedule,
    path: str,
    config: dict = None,
    history: dict = None,
):
    """Save model weights, architecture config, and noise schedule.

    The architecture config is extracted from the model itself, so
    load_checkpoint can reconstruct it without any external info.

    Args:
        model: Trained ScoreNetwork.
        noise_schedule: GeometricNoiseSchedule instance.
        path: File path for the .pt file.
        config: Optional full CONFIG dict for reproducibility.
        history: Optional training history dict.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "data_dim": model.input_proj.in_features,
            "hidden_dim": model.input_proj.out_features,
            "num_res_blocks": len(model.res_blocks),
            "sigma_embed_dim": model.sigma_embed.embed_dim,
        },
        "noise_schedule": {
            "sigma_min": noise_schedule.sigma_min,
            "sigma_max": noise_schedule.sigma_max,
            "num_levels": len(noise_schedule.sigmas),
        },
    }
    if config is not None:
        checkpoint["config"] = config
    if history is not None:
        checkpoint["history"] = history

    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path: str, device: str = "cpu"):
    """Load a checkpoint and reconstruct the model and noise schedule.

    Args:
        path: Path to the .pt checkpoint file.
        device: Device to load onto.

    Returns:
        Tuple of (model, noise_schedule, config_or_None, history_or_None).
    """
    from .noise import GeometricNoiseSchedule

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    mc = checkpoint["model_config"]
    model = ScoreNetwork(
        data_dim=mc["data_dim"],
        hidden_dim=mc["hidden_dim"],
        num_res_blocks=mc["num_res_blocks"],
        sigma_embed_dim=mc["sigma_embed_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    ns = checkpoint["noise_schedule"]
    noise_schedule = GeometricNoiseSchedule(
        sigma_min=ns["sigma_min"],
        sigma_max=ns["sigma_max"],
        num_levels=ns["num_levels"],
    )

    return (
        model,
        noise_schedule,
        checkpoint.get("config"),
        checkpoint.get("history"),
    )
