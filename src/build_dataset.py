"""
build_dataset.py
-----------------
Generates the full DocuGuard dataset: N authentic synthetic ID cards and N
tampered counterparts (one random attack applied per authentic card, so the
dataset is balanced 50/50). Writes JPEGs (quality=92, matching a realistic
"scanned upload" compression) plus a manifest.csv with:

    filename, label (0=authentic,1=tampered), attack_type, bbox_x0, bbox_y0, bbox_x1, bbox_y1, split

Usage:
    python src/build_dataset.py --n 1500 --out data
"""
import argparse
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))

from synth_id_generator import IDRecord, render_card
from tamper import apply_random_tamper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500, help="number of authentic cards (tampered count matches)")
    ap.add_argument("--out", type=str, default="data")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    random.seed(args.seed)
    raw_dir = os.path.join(args.out, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    rows = []
    for i in range(args.n):
        seed = args.seed * 10_000 + i
        rec = IDRecord.random(seed=seed)
        img, boxes, avatar_box = render_card(rec)

        auth_name = f"auth_{seed}.jpg"
        img.save(os.path.join(raw_dir, auth_name), quality=97)
        rows.append({
            "filename": auth_name, "label": 0, "attack_type": "none",
            "bbox_x0": "", "bbox_y0": "", "bbox_x1": "", "bbox_y1": "",
        })

        tampered_img, bbox, attack_name = apply_random_tamper(img, boxes, seed=seed)
        tamp_name = f"tamp_{seed}_{attack_name}.jpg"
        tampered_img.save(os.path.join(raw_dir, tamp_name), quality=97)
        rows.append({
            "filename": tamp_name, "label": 1, "attack_type": attack_name,
            "bbox_x0": bbox[0], "bbox_y0": bbox[1], "bbox_x1": bbox[2], "bbox_y1": bbox[3],
        })

        if (i + 1) % 200 == 0:
            print(f"  generated {i + 1}/{args.n} pairs...")

    random.shuffle(rows)
    n_total = len(rows)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.15)
    for idx, row in enumerate(rows):
        if idx < n_train:
            row["split"] = "train"
        elif idx < n_train + n_val:
            row["split"] = "val"
        else:
            row["split"] = "test"

    manifest_path = os.path.join(args.out, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "label", "attack_type", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1", "split"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. {n_total} images ({args.n} authentic + {args.n} tampered) -> {raw_dir}")
    print(f"Manifest -> {manifest_path}")
    print(f"Split sizes: train={n_train}, val={n_val}, test={n_total - n_train - n_val}")


if __name__ == "__main__":
    main()
