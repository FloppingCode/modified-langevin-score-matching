import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def make_dataset(name: str, n_samples: int = 10_000, seed: int = 42) -> TensorDataset:
    """Generate a 2D toy dataset.

    All datasets are normalized to roughly [-1, 1]^2 so that the noise
    schedule magnitudes are meaningful relative to the data scale.

    Args:
        name: One of "moons", "swiss_roll", "circles", "8gaussians".
        n_samples: Number of points to generate.
        seed: Random seed for reproducibility.

    Returns:
        TensorDataset containing a single tensor of shape (n_samples, 2).
    """
    rng = np.random.RandomState(seed)

    if name == "moons":
        from sklearn.datasets import make_moons

        data, _ = make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
        # Center and scale to [-1, 1]
        data = (data - data.mean(axis=0)) / data.std()

    elif name == "swiss_roll":
        from sklearn.datasets import make_swiss_roll

        data_3d, _ = make_swiss_roll(n_samples=n_samples, noise=0.3, random_state=seed)
        # Take the (x, z) projection for a 2D spiral
        data = data_3d[:, [0, 2]]
        data = (data - data.mean(axis=0)) / data.std()

    elif name == "circles":
        from sklearn.datasets import make_circles

        data, _ = make_circles(
            n_samples=n_samples, noise=0.03, factor=0.5, random_state=seed
        )
        data = (data - data.mean(axis=0)) / data.std()

    elif name == "8gaussians":
        # 8 isotropic Gaussians equally spaced on a circle
        n_per_mode = n_samples // 8
        centers = []
        for i in range(8):
            angle = 2 * np.pi * i / 8
            centers.append([2.0 * np.cos(angle), 2.0 * np.sin(angle)])
        centers = np.array(centers)

        points = []
        for i in range(8):
            n = n_per_mode if i < 7 else n_samples - 7 * n_per_mode
            cluster = rng.randn(n, 2) * 0.1 + centers[i]
            points.append(cluster)
        data = np.concatenate(points, axis=0)
        # Normalize so the ring fits in roughly [-1, 1]
        data = data / 3.0

    else:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Choose from: moons, swiss_roll, circles, 8gaussians"
        )

    tensor = torch.tensor(data, dtype=torch.float32)
    return TensorDataset(tensor)


def make_dataloader(
    dataset: TensorDataset, batch_size: int = 512, shuffle: bool = True
) -> DataLoader:
    """Wrap a TensorDataset in a DataLoader."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
