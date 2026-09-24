"""Train ByteFormer with random byte-level corruption augmentation.

Each training sample receives exactly one corruption: a one-bit flip or a
single-byte deletion. Validation remains clean so that best.pt is selected on
the same clean validation protocol as the baseline experiment.
"""

import argparse
import csv
import json
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from build_course_dataset import DATA_DIR
from byteformer_model import build_model
from data_utils import ENCODING
from prepare import ROOT, WEIGHT_NAME, digest
from train import choose_device


class ArrayDataset(Dataset):
    def __init__(self, tokens, labels, random_variant=False):
        self.tokens = tokens
        self.labels = labels
        self.random_variant = random_variant

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        tokens = self.tokens[index]
        if tokens.ndim == 2:
            variant = int(torch.randint(tokens.shape[0], (1,))) if self.random_variant else 0
            tokens = tokens[variant]
        return torch.from_numpy(tokens.astype(np.int64)), int(self.labels[index])


def augment_byte_batch(clean_tokens, rng, flip_probability=0.5):
    """Apply one medium corruption per sample while preserving [B, L] shape.

    Input values are bytes 0..255 and right padding -1. For byte loss, the
    remaining valid bytes shift left and the last valid position becomes -1.
    """
    if clean_tokens.ndim != 2:
        raise ValueError(f"expected [batch, length], got {tuple(clean_tokens.shape)}")
    output = clean_tokens.detach().cpu().numpy().astype(np.int16, copy=True)
    for row in output:
        valid = np.flatnonzero(row >= 0)
        length = int(valid.size)
        if length < 2:
            continue
        position = int(rng.integers(0, length))
        if rng.random() < flip_probability:
            bit = 1 << int(rng.integers(0, 8))
            row[position] = np.int16(int(row[position]) ^ bit)
        else:
            row[position:length - 1] = row[position + 1:length]
            row[length - 1] = -1
    return torch.from_numpy(output.astype(np.int64, copy=False))


@torch.no_grad()
def evaluate_clean(model, loader, device):
    model.eval()
    loss_sum = correct = count = 0
    for tokens, labels in loader:
        tokens, labels = tokens.to(device), labels.to(device)
        logits = model(tokens)
        loss = nn.functional.cross_entropy(logits, labels)
        loss_sum += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        count += len(labels)
    return {"loss": loss_sum / count, "accuracy": correct / count,
            "correct": correct, "samples": count}


def save_curves(output, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    axes[0].plot(epochs, [r["train_loss"] for r in history], "o-", label="Corrupted train")
    axes[0].plot(epochs, [r["val_clean_loss"] for r in history], "o-", label="Clean validation")
    axes[1].plot(epochs, [100 * r["train_accuracy"] for r in history], "o-", label="Corrupted train")
    axes[1].plot(epochs, [100 * r["val_clean_accuracy"] for r in history], "o-", label="Clean validation")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.2)
        axis.legend()
    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("Accuracy (%)")
    fig.savefig(output / "curves.png", dpi=160)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-milestones", nargs="*", type=int, default=[7, 10])
    parser.add_argument("--lr-gamma", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--flip-probability", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dataset", type=Path, default=DATA_DIR / "mnist_clean_balanced.npz")
    parser.add_argument("--weights", type=Path, default=ROOT / "checkpoints" / WEIGHT_NAME)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError("epochs, batch-size and lr must be positive")
    if not 0 <= args.flip_probability <= 1:
        raise ValueError("flip-probability must be in [0, 1]")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"{args.output} is not empty; choose a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = choose_device(args.device)
    print(f"[ENV] Python {platform.python_version()} | PyTorch {torch.__version__} | device={device}", flush=True)
    if device.type == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(device)}", flush=True)

    with np.load(args.dataset, allow_pickle=False) as data:
        train_key = "train_aug_tokens" if "train_aug_tokens" in data else "train_tokens"
        train_tokens = data[train_key]
        train_labels = data["train_labels"]
        val_tokens = data["val_tokens"]
        val_labels = data["val_labels"]
    print(f"[DATA] train={train_tokens.shape}; validation={val_tokens.shape}; method=flip/loss mix", flush=True)

    train_loader = DataLoader(
        ArrayDataset(train_tokens, train_labels, random_variant=(train_tokens.ndim == 3)),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(ArrayDataset(val_tokens, val_labels), batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    model = build_model(args.weights, num_classes=10).to(device)
    backbone = [p for name, p in model.named_parameters() if not name.startswith("classifier.")]
    optimizer = torch.optim.AdamW(
        [{"params": backbone, "lr": args.lr},
         {"params": model.classifier.parameters(), "lr": args.lr * 10}],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma)
    rng = np.random.default_rng(args.seed)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(method="corrupt_mix_flip_loss", encoding=ENCODING,
                  corrected_masks=True, selection="clean validation accuracy",
                  corruption="one medium corruption per training sample")

    history = []
    best_accuracy, best_epoch = -1.0, 0
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = correct = count = 0
        epoch_start = time.perf_counter()
        for clean_tokens, labels in train_loader:
            augmented = augment_byte_batch(clean_tokens, rng, args.flip_probability)
            augmented, labels = augmented.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(augmented)
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            count += len(labels)
        clean_val = evaluate_clean(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / count,
            "train_accuracy": correct / count,
            "val_clean_loss": clean_val["loss"],
            "val_clean_accuracy": clean_val["accuracy"],
            "selection_score": clean_val["accuracy"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        with (args.output / "history.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row), lineterminator="\n")
            writer.writeheader(); writer.writerows(history)
        if clean_val["accuracy"] > best_accuracy:
            best_accuracy, best_epoch = clean_val["accuracy"], epoch
            state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save({"model": state, "config": config,
                        "best_epoch": best_epoch,
                        "best_selection_score": best_accuracy,
                        "best_clean_val_accuracy": best_accuracy},
                       args.output / "best.pt")
        scheduler.step()
        print(f"[EPOCH {epoch:02d}] corrupted_train={100*row['train_accuracy']:.2f}% "
              f"clean_val={100*row['val_clean_accuracy']:.2f}% time={row['seconds']:.1f}s", flush=True)

    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "best_epoch": best_epoch,
        "best_selection_score": best_accuracy,
        "elapsed_seconds": time.perf_counter() - start,
        "history": history,
        "pretrained_sha256": digest(args.weights),
        "dataset_sha256": digest(args.dataset),
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                         "device": str(device),
                         "hardware": torch.cuda.get_device_name(device) if device.type == "cuda"
                         else platform.processor()},
    }
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    save_curves(args.output, history)
    print(f"[DONE] best_epoch={best_epoch}; clean_val={100*best_accuracy:.2f}%", flush=True)


if __name__ == "__main__":
    main()
