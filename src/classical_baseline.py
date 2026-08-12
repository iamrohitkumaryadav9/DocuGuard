"""
classical_baseline.py
----------------------
A non-deep-learning baseline: hand-crafted, template-region-aware ELA/noise
features (see features.py) fed into a Random Forest. This exists to give an
honest point of comparison for the CNN -- and to be upfront in the README
about what global image statistics alone CAN'T catch (see the ablation
note), versus what a model with spatial awareness (the CNN) picks up.

Usage:
    python src/classical_baseline.py
"""
import json
import os
import time

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from tqdm import tqdm

from features import extract_feature_vector, FEATURE_NAMES


def build_feature_table(manifest_csv, raw_dir):
    df = pd.read_csv(manifest_csv)
    feats = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="extracting features"):
        img = Image.open(os.path.join(raw_dir, row["filename"])).convert("RGB")
        feats.append(extract_feature_vector(img))
    X = np.array(feats)
    y = df["label"].values
    split = df["split"].values
    return X, y, split, df


def main():
    t0 = time.time()
    X, y, split, df = build_feature_table("data/manifest.csv", "data/raw")

    X_train, y_train = X[split == "train"], y[split == "train"]
    X_val, y_val = X[split == "val"], y[split == "val"]
    X_test, y_test = X[split == "test"], y[split == "test"]

    clf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5,
                                  random_state=0, n_jobs=-1, class_weight="balanced")
    clf.fit(X_train, y_train)

    results = {}
    for name, Xs, ys in [("val", X_val, y_val), ("test", X_test, y_test)]:
        probs = clf.predict_proba(Xs)[:, 1]
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(ys, preds)
        auc = roc_auc_score(ys, probs)
        results[name] = {"accuracy": acc, "auc": auc}
        print(f"[{name}] accuracy={acc:.4f}  auc={auc:.4f}")
        print(classification_report(ys, preds, target_names=["authentic", "tampered"]))

    importances = sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda t: -t[1])
    print("\nTop feature importances:")
    for name, imp in importances[:10]:
        print(f"  {name:25s} {imp:.4f}")

    os.makedirs("reports", exist_ok=True)
    with open("reports/classical_baseline_results.json", "w") as f:
        json.dump({
            "results": results,
            "feature_importances": [{"name": n, "importance": float(i)} for n, i in importances],
            "elapsed_sec": time.time() - t0,
        }, f, indent=2)
    print(f"\nDone in {time.time()-t0:.1f}s. Results -> reports/classical_baseline_results.json")


if __name__ == "__main__":
    main()
