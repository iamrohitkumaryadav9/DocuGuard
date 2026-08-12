"""
gradcam.py
----------
Grad-CAM for the ELA branch's last conv layer -- shows WHERE in the
Error-Level-Analysis map the model is focusing when it predicts "tampered",
and lets us sanity-check that against the ground-truth tamper bounding box
we recorded during dataset generation (see build_dataset.py).

Produces:
  - an overlay image per example (original | ELA map | Grad-CAM heatmap)
  - a quantitative localization score: fraction of Grad-CAM energy that
    falls inside the ground-truth tamper bbox, averaged over N tampered
    test examples (a simple stand-in for IoU when there's no predicted box)
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from model import DocuGuardFusionModel
from dataset import rgb_transform, ela_transform
from features import compute_ela


def gradcam_for_ela_branch(model: DocuGuardFusionModel, rgb: torch.Tensor, ela: torch.Tensor):
    """rgb/ela: single-example batches, shape [1, C, H, W]. Returns a HxW heatmap in [0,1]."""
    model.eval()
    ela = ela.clone().requires_grad_(False)

    logits = model(rgb, ela)  # populates model.ela_branch.last_conv_features
    feat = model.ela_branch.last_conv_features
    feat.retain_grad()

    prob = torch.sigmoid(logits)
    model.zero_grad()
    logits.backward()

    grads = feat.grad[0]          # [C, h, w]
    activations = feat[0].detach()  # [C, h, w]
    weights = grads.mean(dim=(1, 2))  # global-average-pooled gradient per channel

    cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
    for c in range(activations.shape[0]):
        cam += weights[c] * activations[c]
    cam = F.relu(cam)
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().numpy(), float(prob.item())


def overlay_heatmap(base_img: Image.Image, cam: np.ndarray, out_path: str):
    cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize(base_img.size, Image.BILINEAR))
    heat = cm.jet(cam_resized / 255.0)[:, :, :3]
    base = np.asarray(base_img.convert("RGB")).astype(np.float32) / 255.0
    overlay = 0.55 * base + 0.45 * heat
    Image.fromarray((overlay * 255).astype(np.uint8)).save(out_path)


def bbox_energy_fraction(cam: np.ndarray, bbox, cam_input_size=96, card_size=(640, 400)):
    """What fraction of total Grad-CAM energy falls inside the GT tamper bbox."""
    x0, y0, x1, y1 = bbox
    W, H = card_size
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    cam_full = np.asarray(cam_img).astype(np.float32)
    total = cam_full.sum() + 1e-8
    inside = cam_full[max(0, y0):min(H, y1), max(0, x0):min(W, x1)].sum()
    return float(inside / total)


def run_examples(manifest_csv="data/manifest.csv", raw_dir="data/raw",
                  ckpt="models/docuguard_best.pt", out_dir="reports/figures", n_examples=6):
    import pandas as pd
    os.makedirs(out_dir, exist_ok=True)
    model = DocuGuardFusionModel()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))

    df = pd.read_csv(manifest_csv)
    tampered_test = df[(df.split == "test") & (df.label == 1)].sample(n_examples, random_state=1)

    fractions = []
    for i, row in tampered_test.iterrows():
        img = Image.open(os.path.join(raw_dir, row["filename"])).convert("RGB")
        rgb_t = rgb_transform(img).unsqueeze(0)
        ela_img = compute_ela(img, quality=90)
        ela_t = ela_transform(ela_img).unsqueeze(0)

        cam, prob = gradcam_for_ela_branch(model, rgb_t, ela_t)
        bbox = (int(row.bbox_x0), int(row.bbox_y0), int(row.bbox_x1), int(row.bbox_y1))
        frac = bbox_energy_fraction(cam, bbox)
        fractions.append(frac)

        out_path = os.path.join(out_dir, f"gradcam_{row.filename.replace('.jpg','')}.png")
        overlay_heatmap(img, cam, out_path)
        print(f"{row.filename:40s} attack={row.attack_type:10s} pred_prob={prob:.3f} "
              f"bbox_energy_fraction={frac:.3f} -> {out_path}")

    mean_frac = float(np.mean(fractions))
    print(f"\nMean Grad-CAM energy inside ground-truth tamper bbox: {mean_frac:.3f} "
          f"(bbox covers ~{(( (bbox[2]-bbox[0])*(bbox[3]-bbox[1]) )/(640*400)):.3f} of the card area on average)")
    return mean_frac


if __name__ == "__main__":
    run_examples()
