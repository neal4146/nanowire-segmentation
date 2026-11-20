"""Side-by-side previews of the hand-drawn outlines and the converted instance masks.

Left: the original labelled image. Right: the raw image with each converted crystal in its own colour,
and regions flagged for review (probable background "pockets") in red. Run from the Materials folder:
    python 03_Code/starter_code/preview_labels.py
"""
import glob
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.segmentation import find_boundaries

LABELS = "starter_code/outputs/labels"
OUT = "starter_code/outputs/label_previews"


def find(pattern):
    hits = [p for p in glob.glob("02_Data/Manual labelling/*") if os.path.basename(p).lower() == pattern.lower()]
    return hits[0]


def labelled_path(name):
    if name.startswith("NbS3"):
        k = name.split()[-1]
        return [p for p in glob.glob("02_Data/Manual labelling/*")
                if os.path.splitext(os.path.basename(p))[0] == f"NbS3 Labeled {k}"][0]
    return find(f"{name} Labeled.tif")


def main():
    os.makedirs(OUT, exist_ok=True)
    names = json.load(open(f"{LABELS}/summary.json", encoding="utf-8"))
    rng = np.random.default_rng(0)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except OSError:
        font = None
    for name in names:
        m = np.array(Image.open(f"{LABELS}/{name}_instances.png")).astype(np.int32)
        flags = json.load(open(f"{LABELS}/{name}_qc_flags.json", encoding="utf-8"))
        raw = [p for p in glob.glob("02_Data/Manual labelling/*")
               if os.path.splitext(os.path.basename(p))[0] == name][0]

        # Masks are aligned with the labelled copy, minus the info bar at the bottom.
        labelled = Image.open(labelled_path(name)).convert("RGB")
        h, w = m.shape
        g = np.array(Image.open(raw), dtype=np.float32)
        lo, hi = np.percentile(g, 1), np.percentile(g, 99)
        g = (np.clip((g - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)
        g = np.array(Image.fromarray(g).resize(labelled.size), dtype=np.float32)[:h, :w]
        rgb = np.stack([g] * 3, -1)

        cols = rng.integers(60, 255, (m.max() + 1, 3))
        flagged = np.array([flags.get(str(i), False) for i in range(m.max() + 1)])
        good = (m > 0) & ~flagged[m]
        bad = (m > 0) & flagged[m]
        rgb[good] = 0.55 * rgb[good] + 0.45 * cols[m[good]]
        rgb[bad] = 0.4 * rgb[bad] + 0.6 * np.array([255, 0, 0])
        rgb[find_boundaries(m, mode="inner")] = 255
        right = Image.fromarray(rgb.astype(np.uint8))

        s = 1100 / right.width
        size = (int(right.width * s), int(right.height * s))
        right = right.resize(size)
        left = labelled.crop((0, 0, w, h)).resize(size)

        out = Image.new("RGB", (size[0] * 2 + 20, size[1] + 50), "white")
        out.paste(left, (0, 50))
        out.paste(right, (size[0] + 20, 50))
        d = ImageDraw.Draw(out)
        n_good, n_bad = int((~flagged[1:]).sum()), int(flagged[1:].sum())
        d.text((10, 12), "Hand-drawn outlines (original)", fill="black", font=font)
        d.text((size[0] + 30, 12), f"Converted: {n_good} crystals in colour, {n_bad} flagged in red",
               fill="black", font=font)
        out.save(f"{OUT}/{name}_preview.jpg", quality=88)
        print(f"{name}: {n_good} crystals, {n_bad} flagged")


if __name__ == "__main__":
    main()
