import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from synth_id_generator import IDRecord, render_card
from features import compute_ela, extract_feature_vector, FEATURE_NAMES, roi_feature_vector


def test_compute_ela_returns_grayscale_same_size():
    rec = IDRecord.random(seed=11)
    img, _, _ = render_card(rec)
    ela = compute_ela(img)
    assert ela.mode == "L"
    assert ela.size == img.size


def test_feature_vector_length_matches_names():
    rec = IDRecord.random(seed=12)
    img, _, _ = render_card(rec)
    vec = extract_feature_vector(img)
    assert vec.shape[0] == len(FEATURE_NAMES)
    assert np.all(np.isfinite(vec))


def test_roi_feature_vector_no_nans():
    rec = IDRecord.random(seed=13)
    img, _, _ = render_card(rec)
    vec = roi_feature_vector(img)
    assert not np.any(np.isnan(vec))
