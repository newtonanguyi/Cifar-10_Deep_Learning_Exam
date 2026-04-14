set (10k images).

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