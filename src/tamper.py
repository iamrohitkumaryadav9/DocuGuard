"""
tamper.py
---------
Simulates realistic document-forgery operations on synthetic ID cards and
records ground-truth tamper bounding boxes (used later to check whether
Grad-CAM localizes tampering in roughly the right place).

Implemented attack types (mirrors real-world ID fraud patterns):
  1. splice        - paste a field region from a DIFFERENT synthetic card
                      (e.g. swapped photo or swapped ID-number strip)
  2. copy_move      - duplicate a region of the SAME card onto another spot
                      (e.g. covering an original ID number with a copy of
                      the background, then stamping fabricated digits)
  3. retype         - erase a text field and re-render it with a different
                      font/anti-aliasing/rotation, simulating a crudely
                      edited field
  4. recompress     - locally re-JPEG-compresses a patch at a different
                      quality than the rest of the image, leaving a
                      double-compression artifact that Error Level Analysis
                      is specifically designed to expose

Each function returns (tampered_image, tamper_bbox).
"""
import io
import random
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont, ImageFilter

from synth_id_generator import IDRecord, render_card, _find_font


BBox = Tuple[int, int, int, int]


def splice_field(img: Image.Image, field_boxes: dict, other_seed: int) -> Tuple[Image.Image, BBox]:
    other_rec = IDRecord.random(other_seed)
    other_img, other_boxes, _ = render_card(other_rec)
    label = random.choice(list(field_boxes.keys()))
    box = field_boxes[label]
    pad = 4
    src_box = (max(0, box[0] - pad), max(0, box[1] - pad), box[2] + pad, box[3] + pad)
    patch = other_img.crop(src_box)
    out = img.copy()
    out.paste(patch, src_box[:2])
    return out, src_box


def copy_move(img: Image.Image, field_boxes: dict) -> Tuple[Image.Image, BBox]:
    label = random.choice(list(field_boxes.keys()))
    box = field_boxes[label]
    w, h = box[2] - box[0], box[3] - box[1]
    src_x = max(0, box[0] - w - 10)
    src_box = (src_x, box[1], src_x + w, box[3])
    patch = img.crop(src_box).resize((box[2] - box[0], box[3] - box[1]))
    out = img.copy()
    dst_box = (box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2)
    out.paste(patch, (box[0], box[1]))
    return out, dst_box


def retype_field(img: Image.Image, field_boxes: dict) -> Tuple[Image.Image, BBox]:
    label = random.choice(list(field_boxes.keys()))
    box = field_boxes[label]
    out = img.copy()
    draw = ImageDraw.Draw(out)
    pad = 3
    erase_box = (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad)
    bg_sample = out.crop((max(0, box[0] - 30), box[1], max(0, box[0] - 25), box[3]))
    avg = tuple(int(x) for x in bg_sample.resize((1, 1)).getpixel((0, 0)))
    draw.rectangle(erase_box, fill=avg)
    fake_font = _find_font(random.choice([15, 16, 19, 20]), bold=random.random() > 0.5)
    fake_text = "".join(random.choices("0123456789ABCDEFXYZ", k=random.randint(5, 10)))
    draw.text((box[0], box[1] - 1), fake_text, font=fake_font, fill=(10, 10, 10))
    return out, erase_box


def recompress_patch(img: Image.Image, field_boxes: dict) -> Tuple[Image.Image, BBox]:
    label = random.choice(list(field_boxes.keys()))
    box = field_boxes[label]
    pad = 10
    patch_box = (max(0, box[0] - pad), max(0, box[1] - pad), box[2] + pad, box[3] + pad)
    patch = img.crop(patch_box)
    buf = io.BytesIO()
    patch.save(buf, format="JPEG", quality=random.randint(25, 45))
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")
    out = img.copy()
    out.paste(recompressed, patch_box[:2])
    return out, patch_box


ATTACKS = {
    "splice": splice_field,
    "copy_move": copy_move,
    "retype": retype_field,
    "recompress": recompress_patch,
}


def inject_compression_artifact(img: Image.Image, bbox: BBox, quality: int) -> Image.Image:
    """
    Simulates the double-JPEG-compression signature real forgeries leave
    behind: the tampered patch is independently JPEG-compressed (as if it
    were copied from another already-compressed image, or saved out of an
    editor) *before* the full composite gets its own final save. This is
    what makes the region detectable under Error Level Analysis even when
    it is a pixel-perfect content match otherwise.
    """
    pad = 6
    x0, y0, x1, y1 = bbox
    box = (max(0, x0 - pad), max(0, y0 - pad), min(img.width, x1 + pad), min(img.height, y1 + pad))
    patch = img.crop(box)
    buf = io.BytesIO()
    patch.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")
    out = img.copy()
    out.paste(recompressed, box[:2])
    return out


def _local_seam_blur(img: Image.Image, bbox: BBox, radius: float) -> Image.Image:
    """
    Blurs ONLY the tampered patch (feathered edges) to mask the paste seam --
    deliberately scoped to the bbox so it doesn't leak a "whole image is
    blurrier" shortcut into regions that were never touched. A classifier
    that picked up on global blur instead of the local artifact would be
    learning a generation-pipeline quirk, not real forgery forensics.
    """
    pad = 8
    x0, y0, x1, y1 = bbox
    box = (max(0, x0 - pad), max(0, y0 - pad), min(img.width, x1 + pad), min(img.height, y1 + pad))
    patch = img.crop(box).filter(ImageFilter.GaussianBlur(radius=radius))
    out = img.copy()
    out.paste(patch, box[:2])
    return out


def apply_random_tamper(img: Image.Image, field_boxes: dict, seed: int):
    rnd = random.Random(seed)
    name = rnd.choice(list(ATTACKS.keys()))
    fn = ATTACKS[name]
    if name == "splice":
        tampered, bbox = fn(img, field_boxes, other_seed=seed + 99991)
    else:
        tampered, bbox = fn(img, field_boxes)
    # mild LOCAL post-processing so the paste seam isn't a giveaway to the
    # naked eye -- scoped to the patch only, see _local_seam_blur docstring
    if rnd.random() > 0.5:
        tampered = _local_seam_blur(tampered, bbox, radius=0.3)
    # bake in a double-compression artifact over the tampered region --
    # this is what a real edited/spliced/re-saved region looks like under ELA
    patch_quality = rnd.randint(35, 65)
    tampered = inject_compression_artifact(tampered, bbox, quality=patch_quality)
    return tampered, bbox, name


if __name__ == "__main__":
    import os
    rec = IDRecord.random(seed=7)
    img, boxes, _ = render_card(rec)
    os.makedirs("data/samples", exist_ok=True)
    img.save("data/samples/authentic_7.png")
    for name, fn in ATTACKS.items():
        if name == "splice":
            out, bbox = fn(img, boxes, other_seed=123)
        else:
            out, bbox = fn(img, boxes)
        out.save(f"data/samples/tampered_{name}.png")
        print(name, bbox)
