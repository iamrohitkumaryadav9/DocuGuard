import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from synth_id_generator import IDRecord, render_card, CARD_W, CARD_H
from tamper import apply_random_tamper, ATTACKS


def test_record_is_deterministic_for_same_seed():
    a = IDRecord.random(seed=42)
    b = IDRecord.random(seed=42)
    assert a == b


def test_record_differs_across_seeds():
    a = IDRecord.random(seed=1)
    b = IDRecord.random(seed=2)
    assert a != b


def test_render_card_output_shape():
    rec = IDRecord.random(seed=5)
    img, boxes, avatar_box = render_card(rec)
    assert img.size == (CARD_W, CARD_H)
    assert len(boxes) == 7
    assert all(len(v) == 4 for v in boxes.values())


def test_all_attack_types_produce_valid_bbox():
    rec = IDRecord.random(seed=8)
    img, boxes, _ = render_card(rec)
    for _ in range(20):
        tampered, bbox, name = apply_random_tamper(img, boxes, seed=_)
        assert name in ATTACKS
        assert tampered.size == img.size
        x0, y0, x1, y1 = bbox
        assert 0 <= x0 < x1 <= CARD_W
        assert 0 <= y0 < y1 <= CARD_H
