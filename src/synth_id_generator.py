"""
synth_id_generator.py
----------------------
Generates fully synthetic identity-card images (no real people, no real PII).

Why synthetic? Real KYC/identity documents are sensitive PII and cannot be
legally scraped or redistributed for training data. Companies building
document-fraud detection (HyperVerge included) rely heavily on programmatically
generated synthetic documents to train and stress-test forgery detectors
without touching real user data. This module reproduces that approach.

Each generated card is a layered PIL composition:
  - background panel with a subtle guilloche-style security pattern
  - a generic avatar silhouette standing in for a photo (no real faces used)
  - text fields (name, DOB, ID number, nationality, sex, expiry)
  - a barcode-style strip and a synthetic signature stroke

Fonts: falls back to PIL's default bitmap font if no TrueType font is found,
so this runs anywhere without extra font files.
"""
import os
import random
import string
import json
from dataclasses import dataclass, asdict
from typing import Tuple, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

CARD_W, CARD_H = 640, 400
SUPERSAMPLE = 2  # render at 2x then downsample -> anti-aliased, photo-like edges
                 # instead of razor-sharp vector edges (which would otherwise
                 # dominate ELA/noise statistics regardless of tampering)

FIRST_NAMES = ["ROHIT", "AISHA", "DAVID", "MARIA", "WEI", "FATIMA", "JOHN", "PRIYA",
               "CARLOS", "YUKI", "OMAR", "ELENA", "SAM", "NINA", "LIAM", "ZARA"]
LAST_NAMES = ["KUMAR", "SHARMA", "SMITH", "GARCIA", "CHEN", "KHAN", "PATEL", "MULLER",
              "SILVA", "TANAKA", "IBRAHIM", "PETROV", "LEE", "SANTOS", "SINGH", "MARTIN"]
NATIONALITIES = ["INDIAN", "AMERICAN", "BRAZILIAN", "GERMAN", "JAPANESE", "KENYAN",
                  "FRENCH", "MEXICAN", "VIETNAMESE", "EGYPTIAN"]
SEXES = ["M", "F"]

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


def _find_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if bold and "Bold" not in path:
            continue
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _rand_id_number() -> str:
    return "".join(random.choices(string.digits, k=4)) + "-" + \
           "".join(random.choices(string.digits, k=4)) + "-" + \
           "".join(random.choices(string.digits, k=4))


def _rand_date(start_year=1965, end_year=2005) -> str:
    y = random.randint(start_year, end_year)
    m = random.randint(1, 12)
    d = random.randint(1, 28)
    return f"{d:02d}/{m:02d}/{y}"


@dataclass
class IDRecord:
    first_name: str
    last_name: str
    dob: str
    id_number: str
    nationality: str
    sex: str
    expiry: str
    bg_hue: int
    seed: int

    @staticmethod
    def random(seed: int) -> "IDRecord":
        rnd = random.Random(seed)
        return IDRecord(
            first_name=rnd.choice(FIRST_NAMES),
            last_name=rnd.choice(LAST_NAMES),
            dob=_rand_date_seeded(rnd),
            id_number=_rand_id_seeded(rnd),
            nationality=rnd.choice(NATIONALITIES),
            sex=rnd.choice(SEXES),
            expiry=f"{rnd.randint(1,12):02d}/{rnd.randint(28,35)}",
            bg_hue=rnd.randint(0, 359),
            seed=seed,
        )


def _rand_date_seeded(rnd: random.Random) -> str:
    y = rnd.randint(1965, 2005)
    m = rnd.randint(1, 12)
    d = rnd.randint(1, 28)
    return f"{d:02d}/{m:02d}/{y}"


def _rand_id_seeded(rnd: random.Random) -> str:
    return "".join(rnd.choices(string.digits, k=4)) + "-" + \
           "".join(rnd.choices(string.digits, k=4)) + "-" + \
           "".join(rnd.choices(string.digits, k=4))


def _hsv_to_rgb_pastel(h: int) -> Tuple[int, int, int]:
    img = Image.new("HSV", (1, 1), (int(h / 360 * 255), 60, 245))
    return img.convert("RGB").getpixel((0, 0))


def _draw_guilloche(draw: ImageDraw.ImageDraw, w: int, h: int, base_color, rnd: random.Random):
    """Draws overlapping sine-wave line patterns, mimicking a security background."""
    for i in range(14):
        amp = rnd.randint(6, 22)
        freq = rnd.uniform(0.02, 0.06)
        phase = rnd.uniform(0, 6.28)
        y0 = rnd.randint(0, h)
        pts = []
        for x in range(0, w, 4):
            import math
            y = y0 + amp * math.sin(freq * x + phase)
            pts.append((x, y))
        shade = tuple(max(0, min(255, c - 25)) for c in base_color)
        draw.line(pts, fill=shade, width=1)


def _draw_avatar(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], rnd: random.Random):
    """A generic, non-identifying avatar silhouette (no real face data)."""
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=(222, 226, 232), outline=(150, 150, 150))
    cx = (x0 + x1) // 2
    head_r = int((x1 - x0) * 0.28)
    head_cy = y0 + int((y1 - y0) * 0.38)
    skin = (int(200 + rnd.randint(-20, 20)), int(170 + rnd.randint(-15, 15)), int(150 + rnd.randint(-15, 15)))
    draw.ellipse([cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r], fill=skin)
    shoulder_w = int((x1 - x0) * 0.75)
    shoulder_top = head_cy + int(head_r * 1.15)
    draw.pieslice([cx - shoulder_w // 2, shoulder_top, cx + shoulder_w // 2, y1 + 40],
                  start=180, end=360, fill=(90, 110, 140))


def _draw_barcode(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], rnd: random.Random):
    x0, y0, x1, y1 = box
    x = x0
    while x < x1:
        bw = rnd.randint(1, 4)
        if rnd.random() > 0.45:
            draw.rectangle([x, y0, min(x + bw, x1), y1], fill=(20, 20, 20))
        x += bw + rnd.randint(1, 3)


def _draw_signature(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], rnd: random.Random):
    import math
    x0, y0, x1, y1 = box
    pts = []
    n = 40
    for i in range(n):
        t = i / n
        x = x0 + t * (x1 - x0)
        y = (y0 + y1) / 2 + math.sin(t * rnd.uniform(6, 10) + rnd.uniform(0, 3)) * (y1 - y0) * 0.35
        pts.append((x, y))
    draw.line(pts, fill=(30, 30, 90), width=2)


def render_card(record: IDRecord) -> Image.Image:
    """
    Renders at SUPERSAMPLE x resolution then downsamples with Lanczos
    resampling. This turns razor-sharp vector edges (text, barcode bars)
    into naturally anti-aliased ones -- much closer to a photographed /
    scanned document, and it keeps those inherent hard edges from swamping
    the forensic (ELA / noise) signal that tampering is supposed to leave.
    A small amount of sensor-style noise is added at the end to mimic a
    real phone-camera capture.
    """
    S = SUPERSAMPLE
    rnd = random.Random(record.seed)
    bg = _hsv_to_rgb_pastel(record.bg_hue)
    W, H = CARD_W * S, CARD_H * S
    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)
    _draw_guilloche(draw, W, H, bg, rnd)

    draw.rectangle([0, 0, W - 1, H - 1], outline=(60, 60, 60), width=3 * S)
    title_font = _find_font(22 * S, bold=True)
    label_font = _find_font(14 * S)
    value_font = _find_font(17 * S, bold=True)

    draw.rectangle([0, 0, W, 46 * S], fill=(35, 55, 95))
    draw.text((16 * S, 12 * S), f"IDENTITY CARD  •  DEMO-{record.nationality[:3]}", font=title_font, fill=(255, 255, 255))

    avatar_box = (24 * S, 66 * S, 184 * S, 266 * S)
    _draw_avatar(draw, avatar_box, rnd)

    fields = [
        ("SURNAME", record.last_name),
        ("GIVEN NAME", record.first_name),
        ("DATE OF BIRTH", record.dob),
        ("SEX", record.sex),
        ("NATIONALITY", record.nationality),
        ("ID NUMBER", record.id_number),
        ("EXPIRY", record.expiry),
    ]
    fx, fy = 200 * S, 66 * S
    field_boxes = {}
    for label, value in fields:
        draw.text((fx, fy), label, font=label_font, fill=(70, 70, 70))
        vy = fy + 17 * S
        draw.text((fx, vy), value, font=value_font, fill=(15, 15, 15))
        tb = draw.textbbox((fx, vy), value, font=value_font)
        field_boxes[label] = tb
        fy += 40 * S

    _draw_signature(draw, (24 * S, 280 * S, 184 * S, 320 * S), rnd)
    _draw_barcode(draw, (200 * S, H - 34 * S, W - 20 * S, H - 14 * S), rnd)

    img = img.resize((CARD_W, CARD_H), Image.LANCZOS)
    field_boxes = {k: tuple(v // S for v in box) for k, box in field_boxes.items()}
    avatar_box = tuple(v // S for v in avatar_box)

    # simulate phone-camera sensor noise from a real document capture
    arr = np.asarray(img).astype(np.int16)
    noise = np.random.default_rng(record.seed).normal(0, 2.2, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    return img, field_boxes, avatar_box


if __name__ == "__main__":
    rec = IDRecord.random(seed=42)
    img, boxes, avatar_box = render_card(rec)
    os.makedirs("data/samples", exist_ok=True)
    img.save("data/samples/sample_authentic.png")
    print(asdict(rec))
    print("field boxes:", boxes)
