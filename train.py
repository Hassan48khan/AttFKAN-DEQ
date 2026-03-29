"""
train.py
--------
Training script for AttFKAN-DEQ on BreakHis / IDC datasets.

Usage
-----
# BreakHis
python train.py --dataset breakhis --data_dir /path/to/BreaKHis_v1 \
                --epochs 50 --batch_size 32 --lr 1e-3

# IDC
python train.py --dataset idc --data_dir /path/to/IDC_regular_ps50_idx5 \
                --epochs 50 --batch_size 32 --lr 1e-3
"""

import os
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

from attfkan_deq import AttFKAN_DEQ


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Data Transforms ───────────────────────────────────────────────────────────

def get_transforms(img_size: int = 50, train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ── Training / Evaluation Helpers ─────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_labels.extend(y.numpy())
        all_preds.extend(preds)
        all_probs.extend(probs)

    acc  = accuracy_score(all_labels, all_preds) * 100
    prec = precision_score(all_labels, all_preds, zero_division=0) * 100
    rec  = recall_score(all_labels, all_preds, zero_division=0) * 100
    f1   = f1_score(all_labels, all_preds, zero_division=0) * 100
    auc  = roc_auc_score(all_labels, all_probs) * 100
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    spec = tn / (tn + fp + 1e-8) * 100
    return {"acc": acc, "prec": prec, "rec": rec, "spec": spec, "f1": f1, "auc": auc}


# ── Main Training Loop ────────────────────────────────────────────────────────

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Full dataset (no transform at load time; applied per-split below)
    full_dataset = ImageFolder(args.data_dir)
    targets = np.array(full_dataset.targets)

    img_size = 50 if args.dataset == "idc" else 224

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    fold_results = []

    for fold, (train_val_idx, test_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        print(f"\n{'='*60}")
        print(f"  Fold {fold + 1}/5")
        print(f"{'='*60}")

        # Inner split: 80% train / 20% val from train_val
        inner_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        train_sub_idx, val_idx = next(
            inner_skf.split(train_val_idx, targets[train_val_idx])
        )
        train_idx = train_val_idx[train_sub_idx]
        val_idx   = train_val_idx[val_idx]

        # Datasets with correct transforms
        train_ds = ImageFolder(args.data_dir, transform=get_transforms(img_size, train=True))
        val_ds   = ImageFolder(args.data_dir, transform=get_transforms(img_size, train=False))
        test_ds  = ImageFolder(args.data_dir, transform=get_transforms(img_size, train=False))

        train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=args.batch_size,
                                  shuffle=True,  num_workers=4, pin_memory=True)
        val_loader   = DataLoader(Subset(val_ds,   val_idx),   batch_size=args.batch_size,
                                  shuffle=False, num_workers=4, pin_memory=True)
        test_loader  = DataLoader(Subset(test_ds,  test_idx),  batch_size=args.batch_size,
                                  shuffle=False, num_workers=4, pin_memory=True)

        # Model
        model = AttFKAN_DEQ(
            in_channels=3,
            hidden_dim=args.hidden_dim,
            num_classes=len(full_dataset.classes),
            grid_size=args.grid_size,
            max_iters=args.max_iters,
            alpha=args.alpha,
            backbone=args.backbone,
            dropout=args.dropout,
        ).to(device)

        if fold == 0:
            print(f"Parameters: {model.count_parameters():,}")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        best_val_loss = float("inf")
        patience_counter = 0
        best_path = os.path.join(args.save_dir, f"best_fold{fold+1}.pt")
        os.makedirs(args.save_dir, exist_ok=True)

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate(model, val_loader, device)

            # Compute val loss for scheduler
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    logits = model(x.to(device))
                    val_loss += criterion(logits, y.to(device)).item() * x.size(0)
            val_loss /= len(val_loader.dataset)
            model.train()

            scheduler.step(val_loss)

            print(f"  Ep {epoch:3d} | train_loss={train_loss:.4f} | "
                  f"val_acc={val_metrics['acc']:.2f}% | val_auc={val_metrics['auc']:.2f}%")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_path)
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping at epoch {epoch}.")
                    break

        # Test on best checkpoint
        model.load_state_dict(torch.load(best_path, map_location=device))
        test_metrics = evaluate(model, test_loader, device)
        fold_results.append(test_metrics)

        print(f"\n  Fold {fold+1} Test Results:")
        for k, v in test_metrics.items():
            print(f"    {k:6s}: {v:.2f}%")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  5-Fold Cross-Validation Summary")
    print(f"{'='*60}")
    for key in fold_results[0]:
        vals = [r[key] for r in fold_results]
        print(f"  {key:6s}: {np.mean(vals):.2f}% ± {np.std(vals):.2f}%")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AttFKAN-DEQ")

    # Data
    parser.add_argument("--dataset", type=str, default="idc",
                        choices=["breakhis", "idc"])
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory of the dataset (ImageFolder structure).")
    parser.add_argument("--save_dir", type=str, default="checkpoints")

    # Model
    parser.add_argument("--backbone",    type=str,   default="custom",
                        choices=["custom", "resnet18", "resnet50"])
    parser.add_argument("--hidden_dim",  type=int,   default=128)
    parser.add_argument("--grid_size",   type=int,   default=8)
    parser.add_argument("--max_iters",   type=int,   default=10)
    parser.add_argument("--alpha",       type=float, default=0.5)
    parser.add_argument("--dropout",     type=float, default=0.2)

    # Training
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience",     type=int,   default=10)
    parser.add_argument("--seed",         type=int,   default=42)

    args = parser.parse_args()
    main(args)
