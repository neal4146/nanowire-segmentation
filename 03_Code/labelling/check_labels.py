"""Check every labels_*.csv and rebuild the review images.

    ..\\..\\..\\labelling_env\\Scripts\\python check_labels.py

For each labels file: wire count, lines with more than 2 points, lines shorter
than 8 px (stray clicks or tiny stubs), near-duplicate lines (endpoints within
6 px). Then writes review/review_<photo>_<quarter>.png: the raw quarter next to
the same quarter with your lines in yellow, at 2x. Only your own lines are drawn.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import label_wires as lw

HERE = Path(__file__).resolve().parent
SCALE = 2


def check(csv: Path) -> pd.DataFrame:
    d = pd.read_csv(csv)
    segs = {w: x[["row", "col"]].values for w, x in d.groupby("wire_id")}
    multi = [w for w, p in segs.items() if len(p) > 2]
    lens = {w: float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))) for w, p in segs.items()}
    print(f"{csv.name}: {len(segs)} wires" + (f", multi-point lines {multi}" if multi else ""))
    for w, L in lens.items():
        if L < 8:
            print(f"  short  wire {w}: {L:.1f} px from {segs[w][0].round(0)} to {segs[w][-1].round(0)}")
    ks = list(segs)
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            pa, pb = segs[a], segs[b]
            m = min(np.linalg.norm(pa[0] - pb[0]) + np.linalg.norm(pa[-1] - pb[-1]),
                    np.linalg.norm(pa[0] - pb[-1]) + np.linalg.norm(pa[-1] - pb[0]))
            if m < 6:
                print(f"  duplicate? wires {a} and {b}")
    return d


def review(csv: Path, d: pd.DataFrame) -> None:
    meta = json.loads(csv.with_suffix(".json").read_text(encoding="utf-8"))
    photo = re.sub(r"(\d)x_", r"\1×_", meta["photo"])
    img, _ = lw.load_crop(photo, meta["quarter"])
    lo, hi = np.percentile(img, [1, 99])
    g = (np.clip((img - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    base = Image.fromarray(g).convert("RGB").resize((g.shape[1] * SCALE, g.shape[0] * SCALE), Image.LANCZOS)
    ov = base.copy()
    dr = ImageDraw.Draw(ov)
    for _, x in d.groupby("wire_id"):
        dr.line([(c * SCALE, r * SCALE) for r, c in x[["row", "col"]].values], fill=(255, 220, 0), width=2)
    W, H = base.size
    out = Image.new("RGB", (W * 2 + 10, H), (0, 0, 0))
    out.paste(base, (0, 0))
    out.paste(ov, (W + 10, 0))
    (HERE / "review").mkdir(exist_ok=True)
    out.save(HERE / "review" / f"review_{csv.stem[len('labels_'):]}.png")


def main() -> None:
    total = 0
    files = sorted(HERE.glob("labels_*.csv"))
    for csv in files:
        d = check(csv)
        total += d.wire_id.nunique()
        review(csv, d)
    print(f"{len(files)} quarters, {total} wires; review images in {HERE / 'review'}")


if __name__ == "__main__":
    main()
