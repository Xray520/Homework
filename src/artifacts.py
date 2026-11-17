"""Deterministic, lightweight medical-image artifact transforms."""

from __future__ import annotations

from typing import Callable

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage


# Severity values are intentionally small for the 28x28 PneumoniaMNIST images.
# They are illustrative settings; the original study used TorchIO-based and
# modality-specific transformations at its own weak/strong settings.
SEVERITY = {
    "weak": {
        "noise_sigma": 0.035,
        "bias_amp": 0.14,
        "motion_shift": 0.75,
        "motion_mix": 0.28,
        "crop_fraction": 0.08,
        "rotation_deg": 5.0,
        "blur_radius": 0.45,
    },
    "strong": {
        "noise_sigma": 0.12,
        "bias_amp": 0.38,
        "motion_shift": 2.0,
        "motion_mix": 0.52,
        "crop_fraction": 0.22,
        "rotation_deg": 18.0,
        "blur_radius": 1.35,
    },
}

ARTIFACTS = ("noise", "bias_field", "motion", "crop", "rotation")


def _as_gray_u8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3:
        array = array.mean(axis=-1)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got shape {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def apply_artifact(
    image: np.ndarray,
    artifact: str,
    level: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return one corrupted image, preserving shape and uint8 range."""
    if artifact not in ARTIFACTS:
        raise ValueError(f"Unknown artifact: {artifact}")
    if level not in SEVERITY:
        raise ValueError(f"Unknown severity level: {level}")

    params = SEVERITY[level]
    src = _as_gray_u8(image)
    height, width = src.shape
    arr = src.astype(np.float32) / 255.0

    if artifact == "noise":
        arr = arr + rng.normal(0.0, params["noise_sigma"], arr.shape)
        return np.clip(arr * 255.0, 0, 255).astype(np.uint8)

    if artifact == "bias_field":
        yy = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
        xx = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
        amp = params["bias_amp"]
        field = 1.0 + amp * (0.50 * xx + 0.35 * yy + 0.30 * xx * yy)
        return np.clip(arr * field * 255.0, 0, 255).astype(np.uint8)

    if artifact == "motion":
        shift = params["motion_shift"] * float(rng.choice([-1, 1]))
        ghost = ndimage.shift(arr, shift=(0.0, shift), order=1, mode="nearest")
        mixed = (1.0 - params["motion_mix"]) * arr + params["motion_mix"] * ghost
        blurred = Image.fromarray(np.clip(mixed * 255.0, 0, 255).astype(np.uint8))
        blurred = blurred.filter(ImageFilter.GaussianBlur(radius=params["blur_radius"]))
        return np.asarray(blurred, dtype=np.uint8)

    if artifact == "crop":
        fraction = params["crop_fraction"]
        cut_y = max(1, int(round(height * fraction)))
        cut_x = max(1, int(round(width * fraction)))
        max_top = max(0, height - 2 * cut_y)
        max_left = max(0, width - 2 * cut_x)
        top = int(rng.integers(0, max_top + 1)) if max_top else 0
        left = int(rng.integers(0, max_left + 1)) if max_left else 0
        cropped = src[top : top + height - cut_y, left : left + width - cut_x]
        return np.asarray(
            Image.fromarray(cropped).resize((width, height), Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )

    angle = float(rng.uniform(-params["rotation_deg"], params["rotation_deg"]))
    rotated = Image.fromarray(src).rotate(
        angle,
        resample=Image.Resampling.BILINEAR,
        fillcolor=0,
    )
    return np.asarray(rotated, dtype=np.uint8)


def corrupt_dataset(
    images: np.ndarray,
    artifact: str,
    level: str,
    seed: int,
) -> np.ndarray:
    """Apply a repeatable but sample-specific artifact to each image."""
    result = np.empty_like(images, dtype=np.uint8)
    level_offset = 0 if level == "weak" else 100_000
    artifact_offset = ARTIFACTS.index(artifact) * 10_000
    for index, image in enumerate(images):
        rng = np.random.default_rng(seed + level_offset + artifact_offset + index)
        result[index] = apply_artifact(image, artifact, level, rng)
    return result
