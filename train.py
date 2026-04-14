"""
CIFAR-10 training pipeline for AnguyiNet.

Hyperparameters live only in CONFIG (top of file). Splits 50k official train
into 45k train / 5k validation; official 10k test is held out for evaluate.py.
"""

from __future__ import annotations

import csv
import os
import pickle
import random
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

from model import AnguyiNet, count_parameters, print_model_summary

# ---------------------------------------------------------------------------
# All hyperparameters and paths — single source of truth (no mid-file literals).
# ---------------------------------------------------------------------------
CONFIG: Dict = {
    "data_root": "./data",
    "checkpoint_dir": "./checkpoints",
    "best_checkpoint": "./checkpoints/best_model.pt",
    "training_log_csv": "./training_log.csv",
    "seed": 42,
    "batch_size": 128,
    "epochs": 50,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "cosine_t_max": 50,
    "early_stopping_patience": 10,
    "dropout_p": 0.4,
    "num_workers": 2,
    "train_subset_size": 45000,
    "val_subset_size": 5000,
    # CIFAR-10 channel statistics (not ImageNet).
    "normalize_mean": (0.4914, 0.4822, 0.4465),
    "normalize_std": (0.2470, 0.2435, 0.2616),
    # If True, never allocate the full 50k×32×32×3 train tensor (~150MiB); read batches
    # from disk (~30MiB at a time). Required on very low-RAM systems.
    "cifar10_train_lazy": True,
    "cifar_download": True,
}

CIFAR10_CLASSES = (
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
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    train: bool,
) -> T.Compose:
    """Build transforms; augmentation only for training (val/test stay clean)."""
    normalize = T.Normalize(mean=mean, std=std)
    if not train:
        return T.Compose([T.ToTensor(), normalize])

    # WHY RandomCrop(32, padding=4): zero-pad then crop back to 32x32 to simulate
    # mild translation / partial framing, improving spatial invariance on small images.
    # WHY RandomHorizontalFlip(0.5): CIFAR-10 categories are broadly left-right invariant
    # (vehicles, animals in profile); flips multiply effective data without distorting semantics.
    # WHY ColorJitter: cheap photometric robustness to exposure/white-balance shifts common
    # in real cameras, reducing overfitting to fixed color histograms of the training set.
    # WHY RandomErasing: randomly masks a rectangle so the network cannot rely on one salient
    # patch; encourages distributed evidence (similar spirit to dropout in image space).
    return T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            normalize,
            T.RandomErasing(p=0.1, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0),
        ]
    )


def ensure_cifar10_on_disk(data_root: str, download: bool) -> None:
    """Download / verify CIFAR-10 without loading all training images into RAM."""
    root = os.path.expanduser(data_root)
    ds = torchvision.datasets.CIFAR10
    folder = os.path.join(root, ds.base_folder)

    def _integrity_ok() -> bool:
        for fname, md5 in ds.train_list + ds.test_list:
            fpath = os.path.join(folder, fname)
            if not check_integrity(fpath, md5):
                return False
        return True

    if not _integrity_ok():
        if not download:
            raise RuntimeError(
                f"CIFAR-10 not found under {folder}. Set CONFIG['data_root'] or enable download."
            )
        download_and_extract_archive(ds.url, root, filename=ds.filename, md5=ds.tgz_md5)
    if not _integrity_ok():
        raise RuntimeError("CIFAR-10 archive extracted but integrity check failed.")


class LazyCifar10Train:
    """
    CIFAR-10 training split (50k) with at most one batch file resident in RAM.

    ``torchvision.datasets.CIFAR10`` concatenates all batches into one ndarray
    (~146MiB contiguous), which can raise ``ArrayMemoryError`` on constrained PCs.
    This class mirrors HWC ``data`` / ``targets`` indexing while loading each
    ``data_batch_*`` pickle on demand (~30MiB).
    """

    _LENGTH = 50_000
    _BATCH_SIZE = 10_000

    def __init__(self, data_root: str) -> None:
        self._root = os.path.expanduser(data_root)
        self._dir = os.path.join(self._root, torchvision.datasets.CIFAR10.base_folder)
        self._batch_files: List[str] = [pair[0] for pair in torchvision.datasets.CIFAR10.train_list]
        self._cache_batch: int | None = None
        self._cache_hwc: np.ndarray | None = None
        self._cache_targets: List[int] | None = None

    def __len__(self) -> int:
        return self._LENGTH

    def _load_batch(self, batch_index: int) -> None:
        if self._cache_batch == batch_index:
            return
        path = os.path.join(self._dir, self._batch_files[batch_index])
        with open(path, "rb") as f:
            entry = pickle.load(f, encoding="latin1")
        raw = entry["data"].reshape(self._BATCH_SIZE, 3, 32, 32).transpose(0, 2, 3, 1)
        self._cache_hwc = np.ascontiguousarray(raw, dtype=np.uint8)
        self._cache_targets = [int(x) for x in entry["labels"]]
        self._cache_batch = batch_index

    def _row(self, global_index: int) -> Tuple[np.ndarray, int]:
        b = global_index // self._BATCH_SIZE
        j = global_index % self._BATCH_SIZE
        self._load_batch(b)
        assert self._cache_hwc is not None and self._cache_targets is not None
        return self._cache_hwc[j], self._cache_targets[j]

    @property
    def data(self) -> Any:
        return _LazyCifar10DataView(self)

    @property
    def targets(self) -> Any:
        return _LazyCifar10TargetView(self)


class _LazyCifar10DataView:
    def __init__(self, parent: LazyCifar10Train) -> None:
        self._p = parent

    def __getitem__(self, idx: int) -> np.ndarray:
        img, _ = self._p._row(idx)
        return img


class _LazyCifar10TargetView:
    def __init__(self, parent: LazyCifar10Train) -> None:
        self._p = parent

    def __getitem__(self, idx: int) -> int:
        _, t = self._p._row(idx)
        return t


class Cifar10IndexedView(Dataset):
    """
    Train/val slices over one CIFAR-10 train backend (eager ``CIFAR10`` or ``LazyCifar10Train``).

    Applies ``transform`` in ``__getitem__`` so train and val can share storage
    (or lazy batch caches) while using different augmentation policies.
    """

    def __init__(self, base: Any, indices: List[int], transform: T.Compose) -> None:
        self.base = base
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        img_np = self.base.data[idx]
        target = int(self.base.targets[idx])
        img = Image.fromarray(img_np)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def _is_allocation_failure(exc: BaseException) -> bool:
    text = str(exc).lower()
    name = type(exc).__name__.lower()
    return "allocate" in text or "memoryerror" in name or "memory" in name


def get_dataloaders(cfg: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Load CIFAR-10 train metadata; split into 45k/5k train/val with different transforms.
    """
    mean = cfg["normalize_mean"]
    std = cfg["normalize_std"]

    train_tfm = build_transforms(mean, std, train=True)
    eval_tfm = build_transforms(mean, std, train=False)

    ensure_cifar10_on_disk(cfg["data_root"], download=cfg.get("cifar_download", True))

    if cfg.get("cifar10_train_lazy", True):
        full_train: Any = LazyCifar10Train(cfg["data_root"])
    else:
        try:
            full_train = torchvision.datasets.CIFAR10(
                root=cfg["data_root"], train=True, download=False, transform=None
            )
        except Exception as e:
            if _is_allocation_failure(e):
                full_train = LazyCifar10Train(cfg["data_root"])
            else:
                raise

    n = len(full_train)
    assert n == 50_000, "Expected 50k CIFAR-10 training images."
    indices = list(range(n))
    random.Random(cfg["seed"]).shuffle(indices)

    train_idx = indices[: cfg["train_subset_size"]]
    val_idx = indices[cfg["train_subset_size"] : cfg["train_subset_size"] + cfg["val_subset_size"]]

    train_set = Cifar10IndexedView(full_train, train_idx, train_tfm)
    val_set = Cifar10IndexedView(full_train, val_idx, eval_tfm)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler | None,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    n_batches = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with autocast():
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        running_acc += accuracy_from_logits(logits.detach(), targets)
        n_batches += 1

    return running_loss / max(n_batches, 1), running_acc / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    n_batches = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if use_amp:
            with autocast():
                logits = model(images)
                loss = criterion(logits, targets)
        else:
            logits = model(images)
            loss = criterion(logits, targets)

        running_loss += loss.item()
        running_acc += accuracy_from_logits(logits, targets)
        n_batches += 1

    return running_loss / max(n_batches, 1), running_acc / max(n_batches, 1)


def main() -> None:
    cfg = CONFIG
    set_seed(cfg["seed"])

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    scaler: GradScaler | None = GradScaler() if use_amp else None

    train_loader, val_loader = get_dataloaders(cfg)

    model = AnguyiNet(num_classes=10, dropout_p=cfg["dropout_p"]).to(device)

    # CrossEntropyLoss = log_softmax + NLL internally; model outputs logits (softmax
    # probabilities obtained via softmax(logits) at inference for interpretability).
    criterion = nn.CrossEntropyLoss()

    # AdamW decouples L2-style regularization from the adaptive momentum estimate:
    # weight_decay is applied directly to weights (AdamW fix), whereas vanilla Adam
    # effectively couples decay with the second-moment scaling, often weakening
    # regularization. Here weight_decay=1e-4 adds L2 penalty to weights only.
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    # Cosine annealing smoothly lowers LR following a half-cosine; no hand-tuned step
    # boundaries. For 50-epoch CIFAR runs this often finds wider minima than aggressive
    # step decay, which can stall in suboptimal plateaus when steps are mistimed.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["cosine_t_max"]
    )

    print_model_summary(model, device)
    print(f"Trainable parameters (exact): {count_parameters(model):,}")
    print(f"Device: {device}, AMP: {use_amp}")

    best_val_acc = 0.0
    epochs_no_improve = 0

    with open(cfg["training_log_csv"], "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "time_sec"]
        )

    global_start = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, use_amp
        )
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        with open(cfg["training_log_csv"], "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [epoch, train_loss, train_acc, val_loss, val_acc, lr, elapsed]
            )

        print(
            f"Epoch {epoch:02d}/{cfg['epochs']} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f} | lr={lr:.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "config": cfg,
                    "classes": CIFAR10_CLASSES,
                },
                cfg["best_checkpoint"],
            )
            print(f"  -> Saved new best checkpoint (val_acc={val_acc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg["early_stopping_patience"]:
                print(
                    f"Early stopping: no val improvement for {cfg['early_stopping_patience']} epochs."
                )
                break

    total_time = time.time() - global_start
    print(f"Training finished in {total_time:.1f}s. Best val acc: {best_val_acc:.4f}")
    print(f"Best model: {cfg['best_checkpoint']}")


if __name__ == "__main__":
    main()
