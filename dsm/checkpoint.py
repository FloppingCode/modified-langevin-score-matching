import os
import torch


def save_checkpoint(
    model,
    path: str,
    noise_schedule=None,
    config: dict = None,
    history: dict = None,
):
    """Save model weights and architecture config.

    Supports ScoreNetwork and SimpleScoreNetwork. The architecture config
    is extracted from the model itself so load_checkpoint can reconstruct it.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    model_type = type(model).__name__
    checkpoint = {
        "model_type": model_type,
        "model_state_dict": model.state_dict(),
    }

    if model_type == "ScoreNetwork":
        checkpoint["model_config"] = {
            "data_dim": model.input_proj.in_features,
            "hidden_dim": model.input_proj.out_features,
            "num_res_blocks": len(model.res_blocks),
            "sigma_embed_dim": model.sigma_embed.embed_dim,
        }
    elif model_type == "SimpleScoreNetwork":
        checkpoint["model_config"] = {
            "data_dim": model.net[0].in_features,
            "hidden_dim": model.net[0].out_features,
        }

    if noise_schedule is not None:
        checkpoint["noise_schedule"] = {
            "sigma_min": noise_schedule.sigma_min,
            "sigma_max": noise_schedule.sigma_max,
            "num_levels": len(noise_schedule.sigmas),
        }
    if config is not None:
        checkpoint["config"] = config
    if history is not None:
        checkpoint["history"] = history

    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path: str, device: str = "cpu"):
    """Load a checkpoint and reconstruct the model.

    Returns:
        Tuple of (model, noise_schedule_or_None, config_or_None, history_or_None).
    """
    from .models import MODEL_REGISTRY
    from .noise import GeometricNoiseSchedule

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model_type = checkpoint.get("model_type", "ScoreNetwork")
    model_cls = MODEL_REGISTRY[model_type]
    model = model_cls(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    noise_schedule = None
    if "noise_schedule" in checkpoint:
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
