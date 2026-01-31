from typing import Dict, Tuple

import numpy as np
import tensorflow as tf


def build_focal_loss(
    y_train: np.ndarray,
    *,
    num_classes: int,
    gamma: float,
    alpha_clip: Tuple[float, float],
) -> Tuple[tf.keras.losses.Loss, Dict[str, float]]:
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    inv = 1.0 / np.maximum(counts, 1.0)

    alpha_vec = inv / inv.mean()
    alpha_vec = np.clip(alpha_vec, alpha_clip[0], alpha_clip[1]).astype(np.float32)

    loss_fn = tf.keras.losses.CategoricalFocalCrossentropy(
        alpha=alpha_vec.tolist(),
        gamma=gamma,
        from_logits=False,
    )

    loss_info = {
        "name": "CategoricalFocalCrossentropy",
        "gamma": float(gamma),
        "alpha_clip": list(alpha_clip),
        "alpha_min": float(alpha_vec.min()),
        "alpha_mean": float(alpha_vec.mean()),
        "alpha_max": float(alpha_vec.max()),
    }

    return loss_fn, loss_info
