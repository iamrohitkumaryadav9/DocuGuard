"""
train.py
--------
End-to-end training loop for the DocuGuard fusion model. CPU-friendly by
design: compact from-scratch CNN branches (see model.py), modest input
resolution (160x160 / 96x96), and a dataset sized for a fast iteration loop
rather than a leaderboard-chasing epoch count.

Usage:
    python src/train.py --epochs 12 --batch_size 32 --lr 1e-3
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

from dataset import DocuGuardDataset
from model import DocuGuardFusionModel


def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device)
            ela = batch["ela"].to(device)
            labels = batch["label"].to(device)
            logits = model(rgb, ela)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    preds = (np.array(all_probs) >= 0.5).astype(int)
    acc = accuracy_score(all_labels, preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")
    return total_loss / len(loader.dataset), acc, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="models")
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cpu")
    torch.manual_seed(0)

    train_ds = DocuGuardDataset(args.manifest, args.raw, "train")
    val_ds = DocuGuardDataset(args.manifest, args.raw, "val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    model = DocuGuardFusionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss()

    history = []
    best_auc = -1.0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            rgb = batch["rgb"].to(device)
            ela = batch["ela"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(rgb, ela)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_ds)
        val_loss, val_acc, val_auc = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        print(f"epoch {epoch:2d}/{args.epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  val_auc={val_auc:.4f}  "
              f"[{elapsed:.1f}s elapsed]")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                         "val_acc": val_acc, "val_auc": val_auc})

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), os.path.join(args.out, "docuguard_best.pt"))

    torch.save(model.state_dict(), os.path.join(args.out, "docuguard_last.pt"))
    with open(os.path.join(args.out, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"Done in {time.time()-t0:.1f}s. Best val AUC={best_auc:.4f}. "
          f"Checkpoints saved to {args.out}/")


if __name__ == "__main__":
    main()
