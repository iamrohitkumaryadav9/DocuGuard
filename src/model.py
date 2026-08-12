"""
model.py
--------
Two-branch fusion model, trained end-to-end FROM SCRATCH (no pretrained
ImageNet weights). This was a deliberate pivot: the sandbox this project
was originally built in has no outbound access to torchvision's weight
CDN, so depending on downloaded pretrained weights would make the project
impossible for anyone else to reproduce in a similarly locked-down
environment (a CI runner, an air-gapped machine, etc). Training from
scratch on a synthetic, narrow-domain dataset (a single fixed ID-card
template) is a reasonable trade here -- the visual domain is far more
constrained than general natural images, so a compact CNN can still learn
it well without transfer learning, and the whole pipeline stays runnable
offline with zero external downloads.

  RGB branch  : small conv stack over the 160x160 RGB image -- learns
                appearance-level cues (layout, field regions).
  ELA branch  : small conv stack over the 96x96 Error-Level-Analysis map --
                learns the forensic double-compression artifact tampering
                leaves behind.
  Fusion head : concatenate both embeddings -> small MLP -> single logit.
"""
import torch
import torch.nn as nn


def _conv_block(in_ch, out_ch, pool=True):
    layers = [
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class RGBBranch(nn.Module):
    """Trained from scratch over 160x160 RGB input."""

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            _conv_block(3, 24),     # 160 -> 80
            _conv_block(24, 48),    # 80 -> 40
            _conv_block(48, 96),    # 40 -> 20
            _conv_block(96, 128),   # 20 -> 10
            _conv_block(128, 128, pool=False),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, out_dim)
        self.last_conv_features = None

    def forward(self, x):
        feat = self.net(x)
        self.last_conv_features = feat
        pooled = self.pool(feat).flatten(1)
        return self.fc(pooled)


class ELABranch(nn.Module):
    """Small CNN over the 1-channel 96x96 ELA map."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            _conv_block(1, 16),   # 96 -> 48
            _conv_block(16, 32),  # 48 -> 24
            _conv_block(32, 64),  # 24 -> 12
            _conv_block(64, 64, pool=False),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, out_dim)
        self.last_conv_features = None  # used by Grad-CAM

    def forward(self, x):
        feat = self.net(x)
        self.last_conv_features = feat
        pooled = self.pool(feat).flatten(1)
        return self.fc(pooled)


class DocuGuardFusionModel(nn.Module):
    def __init__(self, rgb_embed_dim: int = 256, ela_embed_dim: int = 128):
        super().__init__()
        self.rgb_branch = RGBBranch(out_dim=rgb_embed_dim)
        self.ela_branch = ELABranch(out_dim=ela_embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(rgb_embed_dim + ela_embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(128, 1),
        )

    def forward(self, rgb: torch.Tensor, ela: torch.Tensor) -> torch.Tensor:
        rgb_embedding = self.rgb_branch(rgb)
        ela_embedding = self.ela_branch(ela)
        fused = torch.cat([rgb_embedding, ela_embedding], dim=1)
        return self.classifier(fused).squeeze(1)  # logits
