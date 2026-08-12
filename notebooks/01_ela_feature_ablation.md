# Ablation note: global vs. template-ROI-aware ELA features

This documents a real finding from building the classical baseline, kept
here (rather than silently fixed and forgotten) because it's the whole
reason the project uses template-ROI-aware features instead of naive
whole-image statistics, and because it's a useful, honest data point on why
a spatially-aware CNN ends up mattering.

## Setup

For 80 authentic/tampered synthetic card pairs, compute two feature sets:

1. **Whole-image global ELA statistics**: mean, std, max, p95/p99 percentile,
   and coarse-grid (4×4 or 16×10 cell) statistics of the whole-image Error
   Level Analysis map.
2. **Template-ROI-aware statistics**: the same kind of stats, but computed
   only inside the ten known, fixed template regions (photo box, each of the
   7 text-field rows, signature, barcode strip) — see `TEMPLATE_ROIS` in
   `src/features.py`.

Separability reported as `|mean(authentic) - mean(tampered)| / (std_a + std_t)`
— larger is more separable.

## Result

**Global whole-image statistics: effectively no signal.**

```
ela_mean               sep=0.013
ela_std                sep=0.006
ela_max                sep=0.006
ela_p95                sep=0.005
ela_cell_std (4x4)     sep=0.038
ela_cell_max_dev       sep=0.005
noise_var (whole img)  sep=0.289   <- best of the global stats, still weak
```

**Template-ROI-aware statistics: real, usable signal.**

```
field_5_noise_var         sep=0.240
field_2_noise_var         sep=0.145
max_anomaly_noise_zscore  sep=0.118
field_6_noise_var         sep=0.111
...
```

## Why

Every synthetic card — tampered or not — already contains naturally
high-frequency content (crisp text glyphs, barcode bars, the avatar outline).
Those regions produce a locally elevated ELA/noise response on *every* card,
tampered or not. A global "biggest anomalous cell" statistic can't tell the
difference between "this cell is anomalous because it's the barcode" and
"this cell is anomalous because it's a tampered field" — both look like
outliers relative to the flat background. Scoping statistics to only the
specific ten regions where content (and therefore tampering) can actually
occur removes that confound and recovers a real, if still modest, separation.

## Earlier bug this also caught

An earlier version of the tamper simulator applied a "smooth the paste seam"
Gaussian blur to the **entire tampered image** instead of just the pasted
patch. That produced an artificially huge, misleading separation (e.g.
`avatar_noise_var sep=0.696`, on a region the tamper attacks never even
touch) — the model would have been learning "is this image globally
blurrier" rather than anything about localized forgery. Fixed in
`tamper.py`'s `_local_seam_blur`, which is scoped to the tampered bbox only.
This is worth remembering as a general lesson for synthetic-data pipelines:
any post-processing applied inconsistently between the "real" and
"fake" generation paths can leak a shortcut the model will happily learn
instead of the thing you actually wanted it to learn.
