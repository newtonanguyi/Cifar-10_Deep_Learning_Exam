"""
Evaluate AnguyiNet on the official CIFAR-10 test set (10k images).

Loads best checkpoint from training, reports sklearn metrics, saves plots.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import torchvision
import torchvision.transforms as T

from model import AnguyiNet

# Paths align with train.py defaults; override via env if needed.
CHECKPOINT_PATH = os.environ.get("ANGUYINET_CKPT", "./checkpoints/best_model.pt")
TRAINING_LOG = os.environ.get("ANGUYINET_LOG", "./training_log.csv")
OUT_DIR = os.environ.get("ANGUYINET_OUT", ".")

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def get_test_loader(
    data_root: str,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    batch_size: int = 128,
) -> torch.utils.data.DataLoader:
    eval_tfm = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])
    test_set = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=eval_tfm
    )
    return torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )


def load_checkpoint(path: str, device: torch.device) -> Tuple[AnguyiNet, Dict]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    cfg = ckpt.get("config", {})
    dropout_p = cfg.get("dropout_p", 0.4)
    model = AnguyiNet(num_classes=10, dropout_p=dropout_p).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_images: List[np.ndarray] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(targets.numpy().tolist())
        # Store CPU copies for visualization (NCHW).
        all_images.append(images.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.concatenate(all_images, axis=0),
    )


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], out_path: str
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("CIFAR-10 Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_curves(log_path: str, acc_out: str, loss_out: str) -> None:
    if not os.path.isfile(log_path):
        print(f"Warning: {log_path} not found; skipping curve plots.")
        return

    df = pd.read_csv(log_path)
    epochs = df["epoch"]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, df["train_acc"], label="Train accuracy")
    plt.plot(epochs, df["val_acc"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(acc_out, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, df["train_loss"], label="Train loss")
    plt.plot(epochs, df["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_out, dpi=150)
    plt.close()


def denormalize_batch(
    images: np.ndarray,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> np.ndarray:
    m = np.array(mean).reshape(1, 3, 1, 1)
    s = np.array(std).reshape(1, 3, 1, 1)
    x = images * s + m
    return np.clip(x, 0.0, 1.0)


def plot_sample_predictions(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    class_names: List[str],
    out_path: str,
    n_show: int = 16,
) -> None:
    imgs = denormalize_batch(images[:n_show], mean, std)
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    axes = axes.flatten()
    for i in range(n_show):
        ax = axes[i]
        im = np.transpose(imgs[i], (1, 2, 0))
        ax.imshow(im)
        ax.axis("off")
        ok = y_true[i] == y_pred[i]
        color = "green" if ok else "red"
        ax.set_title(
            f"T: {class_names[y_true[i]]}\nP: {class_names[y_pred[i]]}",
            fontsize=8,
            color=color,
        )
    plt.suptitle("Sample predictions (green=correct, red=wrong)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def print_summary_table(report_dict: dict, overall_acc: float) -> None:
    print("\n" + "=" * 72)
    print(f"Overall test accuracy: {overall_acc * 100:.2f}%")
    print("=" * 72)
    print(
        f"{'Class':<12} {'Precision':>12} {'Recall':>12} {'F1-score':>12} {'Support':>10}"
    )
    print("-" * 72)
    for c in CIFAR10_CLASSES:
        row = report_dict.get(c)
        if not isinstance(row, dict):
            continue
        print(
            f"{c:<12} {row['precision']:>12.4f} {row['recall']:>12.4f} "
            f"{row['f1-score']:>12.4f} {int(row['support']):>10d}"
        )
    print("-" * 72)
    for avg in ("macro avg", "weighted avg"):
        row = report_dict.get(avg)
        if not isinstance(row, dict):
            continue
        print(
            f"{avg:<12} {row['precision']:>12.4f} {row['recall']:>12.4f} "
            f"{row['f1-score']:>12.4f} {int(row['support']):>10d}"
        )
    print("=" * 72 + "\n")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found at {CHECKPOINT_PATH}. Run train.py first."
        )

    model, ckpt = load_checkpoint(CHECKPOINT_PATH, device)
    cfg = ckpt.get("config", {})
    data_root = cfg.get("data_root", "./data")
    mean = tuple(cfg.get("normalize_mean", (0.4914, 0.4822, 0.4465)))
    std = tuple(cfg.get("normalize_std", (0.2470, 0.2435, 0.2616)))
    batch_size = cfg.get("batch_size", 128)

    test_loader = get_test_loader(data_root, mean, std, batch_size=batch_size)

    y_true, y_pred, all_images = collect_predictions(model, test_loader, device)

    overall_acc = (y_true == y_pred).mean()
    report = classification_report(
        y_true,
        y_pred,
        target_names=CIFAR10_CLASSES,
        output_dict=True,
        zero_division=0,
    )

    print("\n sklearn classification_report (text):\n")
    print(
        classification_report(
            y_true, y_pred, target_names=CIFAR10_CLASSES, zero_division=0
        )
    )

    print_summary_table(report, float(overall_acc))

    cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
    acc_path = os.path.join(OUT_DIR, "accuracy_curve.png")
    loss_path = os.path.join(OUT_DIR, "loss_curve.png")
    samples_path = os.path.join(OUT_DIR, "sample_predictions.png")

    plot_confusion_matrix(y_true, y_pred, CIFAR10_CLASSES, cm_path)
    plot_curves(TRAINING_LOG, acc_path, loss_path)
    plot_sample_predictions(
        all_images, y_true, y_pred, mean, std, CIFAR10_CLASSES, samples_path
    )

    print(f"Saved: {cm_path}")
    print(f"Saved: {acc_path}")
    print(f"Saved: {loss_path}")
    print(f"Saved: {samples_path}")


if __name__ == "__main__":
    main()
