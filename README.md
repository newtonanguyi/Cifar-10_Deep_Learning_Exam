# AnguyiNet — CIFAR-10 Classification

Custom PyTorch project: **AnguyiNet** (convolutional blocks + channel attention + depthwise separable stages) trained on CIFAR-10 with AdamW, cosine LR schedule, mixed precision on GPU, and full evaluation plots.

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py
```

Creates `./data/` (download), `./checkpoints/best_model.pt`, and `./training_log.csv`.

Training uses **lazy CIFAR-10 loading** by default (`CONFIG["cifar10_train_lazy"]`): only one batch file (~30 MiB) is in RAM at a time, avoiding the full 50k-image contiguous allocation that fails on very low-memory machines. If you have plenty of RAM and want the standard in-memory dataset, set `"cifar10_train_lazy": False` in `train.py`.

## Evaluate

```bash
python evaluate.py
```

Optional environment variables: `ANGUYINET_CKPT`, `ANGUYINET_LOG`, `ANGUYINET_OUT` for checkpoint path, training log path, and output directory.

Generates `confusion_matrix.png`, `accuracy_curve.png`, `loss_curve.png`, and `sample_predictions.png` in the output directory (default: current directory).

## LaTeX

From this folder (after evaluation so figures exist):

```bash
pdflatex report.tex
pdflatex slides.tex
```

For Beamer, add optional assets if desired: `ucu_logo` (e.g. `ucu_logo.png`), `cifar10_samples` for the sample grid slide. The document compiles without them if you comment out those lines or supply files.

## Files

| File | Role |
|------|------|
| `model.py` | AnguyiNet definition |
| `train.py` | Data splits, augmentation, training loop, logging |
| `evaluate.py` | Test metrics, plots, sample grid |
| `report.tex` | Two-column article |
| `slides.tex` | Beamer slides |
| `requirements.txt` | Python dependencies |

## Model summary

```bash
python model.py
```
