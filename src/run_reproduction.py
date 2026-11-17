"""Small CPU reproduction of medical-image artifact robustness evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import urllib.request
import warnings
from pathlib import Path

import joblib
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score

from artifacts import ARTIFACTS, corrupt_dataset


DATA_URL = "https://zenodo.org/records/10519652/files/pneumoniamnist.npz?download=1"
DATA_MD5 = "28209eda62fecd6e6a2d98b1501bb15f"
DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def download_dataset(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest == DATA_MD5:
            return
        path.unlink()

    request = urllib.request.Request(
        DATA_URL,
        headers={"User-Agent": "medical-artifact-robustness-repro/1.0"},
    )
    print(f"Downloading PneumoniaMNIST from Zenodo to {path} ...")
    with urllib.request.urlopen(request, timeout=90) as response, path.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)

    digest = hashlib.md5(path.read_bytes()).hexdigest()
    if digest != DATA_MD5:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"Dataset checksum mismatch: got {digest}, expected {DATA_MD5}")


def load_split(path: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as archive:
        images = np.asarray(archive[f"{split}_images"], dtype=np.uint8)
        labels = np.asarray(archive[f"{split}_labels"], dtype=np.int64).reshape(-1)
    if images.ndim == 4 and images.shape[-1] == 1:
        images = images[..., 0]
    if images.ndim != 3:
        raise ValueError(f"Expected N×H×W images in {split}, got {images.shape}")
    return images, labels


def balanced_subset(
    images: np.ndarray,
    labels: np.ndarray,
    limit: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select a deterministic class-balanced subset, then shuffle it."""
    if limit <= 0 or limit >= len(labels):
        return images.copy(), labels.copy()

    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    per_class = max(1, limit // len(classes))
    chosen: list[int] = []
    remaining: list[int] = []
    for label in classes:
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        take = min(per_class, len(indices))
        chosen.extend(indices[:take].tolist())
        remaining.extend(indices[take:].tolist())

    if len(chosen) < limit:
        rng.shuffle(remaining)
        chosen.extend(remaining[: limit - len(chosen)].tolist())

    chosen_array = np.asarray(chosen[:limit], dtype=np.int64)
    rng.shuffle(chosen_array)
    return images[chosen_array], labels[chosen_array]



def make_synthetic_split(count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic 28x28 chest-radiograph-like patterns for offline demos.

    Class 1 contains a randomly positioned soft opacity; these are synthetic
    images for validating the evaluation pipeline, not patient data.
    """
    if count < 2:
        raise ValueError("Synthetic splits need at least two images")
    rng = np.random.default_rng(seed)
    height = width = 28
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    labels = np.arange(count, dtype=np.int64) % 2
    rng.shuffle(labels)
    images = np.empty((count, height, width), dtype=np.uint8)

    for index, label in enumerate(labels):
        shift_x = rng.uniform(-0.8, 0.8)
        shift_y = rng.uniform(-0.7, 0.7)
        x = xx - shift_x
        y = yy - shift_y
        left_lung = np.exp(-(((x - 8.5) / 5.5) ** 2 + ((y - 14) / 10.5) ** 2) * 1.3)
        right_lung = np.exp(-(((x - 19.5) / 5.5) ** 2 + ((y - 14) / 10.5) ** 2) * 1.3)
        tissue = 0.74 - 0.37 * (left_lung + right_lung)
        spine = 0.10 * np.exp(-((x - 14) / 2.1) ** 2) * np.exp(-((y - 14) / 13) ** 2)
        image = tissue + spine

        # Faint curved rib traces add low-amplitude structure to both classes.
        for rib in range(5):
            level = 7.0 + rib * 3.5
            curve = level + 0.08 * ((xx - 14) ** 2) / 14
            trace = np.exp(-((yy - curve) / 0.48) ** 2)
            image += 0.025 * trace * (left_lung + right_lung)

        if label == 1:
            side = int(rng.integers(0, 2))
            center_x = rng.uniform(5.5, 11.5) if side == 0 else rng.uniform(16.5, 22.5)
            center_y = rng.uniform(14.5, 22.5)
            sigma_x = rng.uniform(2.0, 4.0)
            sigma_y = rng.uniform(2.0, 4.0)
            opacity = np.exp(-(((xx - center_x) / sigma_x) ** 2 + ((yy - center_y) / sigma_y) ** 2) * 0.5)
            image += rng.uniform(0.16, 0.30) * opacity

        contrast = rng.uniform(0.88, 1.12)
        brightness = rng.uniform(-0.035, 0.035)
        image = (image - 0.5) * contrast + 0.5 + brightness
        image += rng.normal(0.0, 0.018, size=image.shape)
        images[index] = np.clip(image * 255, 0, 255).astype(np.uint8)

    return images, labels


def features(images: np.ndarray) -> np.ndarray:
    return images.reshape(len(images), -1).astype(np.float32) / 255.0


def metrics_for(
    model: MLPClassifier,
    images: np.ndarray,
    labels: np.ndarray,
    confidence_threshold: float,
) -> dict[str, float]:
    probabilities = model.predict_proba(features(images))
    predictions = model.classes_[np.argmax(probabilities, axis=1)]
    confidence = probabilities.max(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "pneumonia_sensitivity": float(
            recall_score(labels, predictions, pos_label=1, zero_division=0)
        ),
        "low_confidence_rate": float(np.mean(confidence <= confidence_threshold)),
    }


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def make_plot(
    rows: list[dict[str, object]],
    clean_accuracy: float,
    model_name: str,
    dataset_name: str,
    train_count: int,
    test_count: int,
    output_path: Path,
) -> None:
    labels = list(ARTIFACTS)
    weak_acc = [
        float(row["accuracy"])
        for row in rows
        if row["level"] == "weak"
    ]
    strong_acc = [
        float(row["accuracy"])
        for row in rows
        if row["level"] == "strong"
    ]
    weak_change = [
        float(row["performance_change_pct"])
        for row in rows
        if row["level"] == "weak"
    ]
    strong_change = [
        float(row["performance_change_pct"])
        for row in rows
        if row["level"] == "strong"
    ]

    x = np.arange(len(labels))
    width = 0.36
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.patch.set_facecolor("#f7f8fa")
    for axis in (ax1, ax2):
        axis.set_facecolor("white")
        axis.grid(axis="y", color="#dbe1e8", linewidth=0.7, alpha=0.8)
        axis.set_axisbelow(True)

    ax1.bar(x - width / 2, weak_acc, width, color="#3b82f6", label="Weak artifacts")
    ax1.bar(x + width / 2, strong_acc, width, color="#ef4444", label="Strong artifacts")
    ax1.axhline(clean_accuracy, color="#111827", linestyle="--", linewidth=1.5,
                label=f"Clean baseline ({clean_accuracy:.3f})")
    ax1.set_title("Classification accuracy", fontsize=13, weight="bold")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1)
    ax1.set_xticks(x, labels, rotation=24, ha="right")
    ax1.legend(frameon=False, fontsize=9)

    ax2.bar(x - width / 2, weak_change, width, color="#3b82f6", label="Weak artifacts")
    ax2.bar(x + width / 2, strong_change, width, color="#ef4444", label="Strong artifacts")
    ax2.axhline(0, color="#111827", linewidth=1)
    ax2.set_title("Relative accuracy change vs. clean", fontsize=13, weight="bold")
    ax2.set_ylabel("Change (%)")
    ax2.set_xticks(x, labels, rotation=24, ha="right")
    ax2.legend(frameon=False, fontsize=9)

    fig.suptitle("Medical image artifact robustness — lightweight reproduction",
                 fontsize=16, weight="bold", y=0.99)
    fig.text(
        0.5,
        0.015,
        f"{model_name}  |  {dataset_name}  |  train n={train_count}  |  test n={test_count}  |  "
        "28×28  |  low-confidence proxy is not VLM self-report",
        ha="center",
        fontsize=9,
        color="#374151",
    )
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small, CPU-friendly medical image artifact robustness experiment."
    )
    parser.add_argument("--dataset", choices=("synthetic", "medmnist"), default="synthetic",
                        help="Use the offline synthetic demonstration or official PneumoniaMNIST")
    parser.add_argument("--data", type=Path, default=DEFAULT_ROOT / "data" / "pneumoniamnist.npz")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "results")
    parser.add_argument("--train-limit", type=int, default=1200)
    parser.add_argument("--val-limit", type=int, default=400)
    parser.add_argument("--test-limit", type=int, default=468)
    parser.add_argument("--max-iter", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    if args.dataset == "synthetic":
        train_x, train_y = make_synthetic_split(args.train_limit, args.seed)
        val_x, val_y = make_synthetic_split(args.val_limit, args.seed + 1)
        test_x, test_y = make_synthetic_split(args.test_limit, args.seed + 2)
        dataset_name = "SyntheticXrayLike (offline demo)"
    else:
        download_dataset(args.data)
        train_x, train_y = load_split(args.data, "train")
        val_x, val_y = load_split(args.data, "val")
        test_x, test_y = load_split(args.data, "test")
        train_x, train_y = balanced_subset(train_x, train_y, args.train_limit, args.seed)
        val_x, val_y = balanced_subset(val_x, val_y, args.val_limit, args.seed + 1)
        test_x, test_y = balanced_subset(test_x, test_y, args.test_limit, args.seed + 2)
        dataset_name = "PneumoniaMNIST"

    model = MLPClassifier(
        hidden_layer_sizes=(48,),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        batch_size=64,
        learning_rate_init=1e-3,
        max_iter=args.max_iter,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=4,
        random_state=args.seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(features(train_x), train_y)

    val_probabilities = model.predict_proba(features(val_x))
    val_confidence = val_probabilities.max(axis=1)
    confidence_threshold = float(np.quantile(val_confidence, 0.05))

    clean = metrics_for(model, test_x, test_y, confidence_threshold)
    rows: list[dict[str, object]] = [{
        "artifact": "clean",
        "level": "clean",
        "n_test": len(test_y),
        "accuracy": clean["accuracy"],
        "pneumonia_sensitivity": clean["pneumonia_sensitivity"],
        "performance_change_pct": 0.0,
        "low_confidence_rate": clean["low_confidence_rate"],
    }]

    for artifact in ARTIFACTS:
        for level in ("weak", "strong"):
            corrupted = corrupt_dataset(test_x, artifact, level, args.seed)
            result = metrics_for(model, corrupted, test_y, confidence_threshold)
            change_pct = 100.0 * (result["accuracy"] - clean["accuracy"]) / max(clean["accuracy"], 1e-8)
            rows.append({
                "artifact": artifact,
                "level": level,
                "n_test": len(test_y),
                "accuracy": result["accuracy"],
                "pneumonia_sensitivity": result["pneumonia_sensitivity"],
                "performance_change_pct": change_pct,
                "low_confidence_rate": result["low_confidence_rate"],
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "metrics.csv")
    joblib.dump(model, args.output_dir / "model.joblib")
    summary = {
        "dataset": dataset_name,
        "model": "scikit-learn MLPClassifier(hidden_layer_sizes=(48,))",
        "seed": args.seed,
        "train_n": int(len(train_y)),
        "validation_n": int(len(val_y)),
        "test_n": int(len(test_y)),
        "requested_max_iter": args.max_iter,
        "actual_iterations": int(model.n_iter_),
        "clean_accuracy": clean["accuracy"],
        "clean_pneumonia_sensitivity": clean["pneumonia_sensitivity"],
        "clean_low_confidence_rate": clean["low_confidence_rate"],
        "low_confidence_threshold": confidence_threshold,
        "quality_metric_note": (
            "low_confidence_rate is a confidence-based proxy calibrated at the "
            "5th percentile on clean validation images; it is not VLM self-report."
        ),
        "method_note": (
            "Single-modality CPU reproduction of the artifact-generation and robustness "
            "evaluation pipeline. SyntheticXrayLike is simulated data; medmnist mode "
            "uses official PneumoniaMNIST. Neither mode reproduces the original VLM models or their numerical results."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    make_plot(
        rows,
        clean["accuracy"],
        summary["model"],
        summary["dataset"],
        len(train_y),
        len(test_y),
        args.output_dir / "results_screenshot.png",
    )

    print(f"Dataset: {summary['dataset']} | train={len(train_y)} val={len(val_y)} test={len(test_y)}")
    print(f"Model iterations: {model.n_iter_}/{args.max_iter}")
    print(f"Clean accuracy: {clean['accuracy']:.3f} | pneumonia sensitivity: {clean['pneumonia_sensitivity']:.3f}")
    print(f"Confidence threshold (clean-val 5th percentile): {confidence_threshold:.3f}")
    print("\nartifact,level,accuracy,sensitivity,change_pct,low_confidence_rate")
    for row in rows:
        print(
            f"{row['artifact']},{row['level']},{float(row['accuracy']):.3f},"
            f"{float(row['pneumonia_sensitivity']):.3f},"
            f"{float(row['performance_change_pct']):+.1f},"
            f"{float(row['low_confidence_rate']):.3f}"
        )
    print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
