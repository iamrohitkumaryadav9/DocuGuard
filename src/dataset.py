"""
dataset.py
----------
PyTorch Dataset for DocuGuard. Each item yields:
  - a 224x224 RGB tensor (ImageNet-normalized) for the frozen backbone branch
  - a 96x96 single-channel ELA tensor for the trainable forensic branch
  - the label (0=authentic, 1=tampered)
  - the ground-truth tamper bbox (or None), used only for localization eval
"""
import os
import ast
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from features import compute_ela

# Trained from scratch (see model.py for why) -> simple [-1, 1] normalization
# rather than ImageNet statistics, which are meaningless for a model that
# never saw ImageNet.
rgb_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

ela_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


class DocuGuardDataset(Dataset):
    def __init__(self, manifest_csv: str, raw_dir: str, split: str):
        df = pd.read_csv(manifest_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.raw_dir = raw_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.raw_dir, row["filename"])
        img = Image.open(path).convert("RGB")

        rgb_t = rgb_transform(img)
        ela_img = compute_ela(img, quality=90)
        ela_t = ela_transform(ela_img)

        label = int(row["label"])
        if pd.notna(row["bbox_x0"]) and row["bbox_x0"] != "":
            bbox = (int(row["bbox_x0"]), int(row["bbox_y0"]), int(row["bbox_x1"]), int(row["bbox_y1"]))
        else:
            bbox = (-1, -1, -1, -1)

        return {
            "rgb": rgb_t,
            "ela": ela_t,
            "label": torch.tensor(label, dtype=torch.float32),
            "bbox": torch.tensor(bbox, dtype=torch.float32),
            "filename": row["filename"],
            "attack_type": row["attack_type"],
        }
