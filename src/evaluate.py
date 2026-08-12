"""
evaluate.py
-----------
Full test-set evaluation for the trained fusion model: accuracy, precision,
recall, F1, ROC-AUC, confusion matrix, a per-attack-type breakdown (splice
vs copy_move vs retype vs recompress -- some are intrinsically harder), and
saved plots.

Usage:
    python src/evaluate.py --ckpt models/docuguard_best.pt
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, roc_curve, confusion_matrix)
import matplotlib.pyplot as plt

from dataset import DocuGuardDataset
from model import DocuGuardFusionModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--ckpt", default="models/docuguard_best.pt")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)

    device = torch.device("cpu")
    model = DocuGuardFusionModel().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    test_ds = DocuGuardDataset(args.manifest, args.raw, "test")
    loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    all_probs, all_labels, all_attacks, all_files = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["rgb"].to(device), batch["ela"].to(device))
            probs = torch.sigmoid(logits).numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(batch["label"].numpy().tolist())
            all_attacks.extend(batch["attack_type"])
            all_files.extend(batch["filename"])

    y = np.array(all_labels)
    p = np.array(all_probs)
    preds = (p >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds),
        "recall": recall_score(y, preds),
        "f1": f1_score(y, preds),
        "roc_auc": roc_auc_score(y, p),
        "n_test": len(y),
    }
    print("Overall test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    cm = confusion_matrix(y, preds)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["authentic", "tampered"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["authentic", "tampered"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix (test)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "figures", "confusion_matrix.png"), dpi=140)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, label=f"AUC={metrics['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (test)"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "figures", "roc_curve.png"), dpi=140)
    plt.close(fig)

    df = pd.DataFrame({"label": y, "pred": preds, "prob": p, "attack": all_attacks, "file": all_files})
    per_attack = {}
    for attack, g in df[df.label == 1].groupby("attack"):
        per_attack[attack] = {
            "n": int(len(g)),
            "recall": float((g.pred == 1).mean()),
            "mean_prob": float(g.prob.mean()),
        }
    print("\nPer-attack-type recall (tampered class only):")
    for a, s in per_attack.items():
        print(f"  {a:12s} n={s['n']:4d}  recall={s['recall']:.3f}  mean_pred_prob={s['mean_prob']:.3f}")

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"overall": metrics, "per_attack_type": per_attack}, f, indent=2)
    print(f"\nSaved metrics.json and figures/ under {args.out}/")


if __name__ == "__main__":
    main()
