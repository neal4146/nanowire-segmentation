"""Score every method on a NEW labelled photo (labels drawn as lines with
Materials_ThangPham/03_Code/labelling/label_wires.py).

The two 80k labelled photos are used only to choose settings (and train the
forests), exactly as saved in results/compare/results.json. Each method is then
run on the full 50k photo the way production runs it, cropped to the labelled
quarter, and scored there. So this is a test on a photo none of the methods has
seen, from the 60-min growth run that Nikhil's pipeline actually measures.

    venv/Scripts/python compare/external.py                      # newest labels file
    venv/Scripts/python compare/external.py --labels path/to/labels_....csv

Each saved direction (settings from photo 1, settings from photo 2) is applied
and the two scores are averaged. Nikhil's production forest (trained on both
80k photos, p > 0.70) is added as 'rf_v3_production'.

Ground truth from the lines:
  wires       every drawn line is one wire (crossings stay whole, unlike outlines)
  pixel mask  lines widened to 8 px (~44 nm, the outer wire width measured at
              200k by width_check.py, the same convention as the 80k outlines)
  count, median length and junctions come straight from the lines; width is not
  labelled, so width error is blank.
Writes results/compare/external_<photo>_<quarter>.csv and overlays.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from skimage.draw import line as draw_line

sys.path.insert(0, str(Path(__file__).resolve().parent))

from score import OUT, Frame, extract_params, save_overlay, score_mask   # noqa: E402
from ridge_nanowire_images2 import load_frame_50k                       # noqa: E402  (path set by score)
from wire_classifier import PROD_THR, load_classifier, wire_probability  # noqa: E402
from methods import METHODS                                             # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MATERIALS = ROOT.parent if (ROOT.parent / "03_Code").is_dir() else ROOT.parent / "Materials_ThangPham"     # folders as reorganised on 21 Oct
LABEL_DIR = MATERIALS / "03_Code" / "labelling"
PHOTO_DIRS = [ROOT / "data" / "real" / "NanowireImages2",
              *sorted((MATERIALS / "02_Data" / "SEM images" / "TiS3 on sapphire").glob("SF*"))]
GT_HALF_WIDTH = 4.0      # px: lines widened to ~8-9 px (~44 nm at 5.4 nm/px)
JUNCTION_TOL = 3.0       # px: a line ending this close to another counts as a junction


def read_lines(csv_path: Path) -> list[np.ndarray]:
    wires: dict[int, list] = {}
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            wires.setdefault(int(row["wire_id"]), []).append((float(row["row"]), float(row["col"])))
    return [np.array(v) for _, v in sorted(wires.items())]


def _segments(lines):
    for i, L in enumerate(lines):
        for a, b in zip(L[:-1], L[1:]):
            yield i, a, b


def _seg_dist(p, a, b) -> float:
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
    return float(np.hypot(*(p - (a + t * ab))))


def _intersect(a, b, c, d):
    """Intersection point of segments ab and cd, or None."""
    r, s = b - a, d - c
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 1e-9:
        return None
    t = ((c - a)[0] * s[1] - (c - a)[1] * s[0]) / den
    u = ((c - a)[0] * r[1] - (c - a)[1] * r[0]) / den
    return a + t * r if 0 <= t <= 1 and 0 <= u <= 1 else None


def junctions(lines) -> np.ndarray:
    """Crossings and T-junctions between different drawn wires, (row, col)."""
    segs = list(_segments(lines))
    pts = []
    for x in range(len(segs)):
        for y in range(x + 1, len(segs)):
            i, a, b = segs[x]
            j, c, d = segs[y]
            if i == j:
                continue
            p = _intersect(a, b, c, d)
            if p is None:
                for e, (s0, s1) in ((a, (c, d)), (b, (c, d)), (c, (a, b)), (d, (a, b))):
                    if _seg_dist(e, s0, s1) <= JUNCTION_TOL:
                        p = e
                        break
            if p is not None:
                pts.append(p)
    if not pts:
        return np.zeros((0, 2))
    pts = np.array(pts)                     # merge duplicates within the tolerance
    keep = []
    for p in pts:
        if all(np.hypot(*(p - q)) > JUNCTION_TOL for q in keep):
            keep.append(p)
    return np.array(keep)


def lines_to_frame(lines, shape, nm_per_px, name, img01) -> tuple[Frame, dict]:
    inst = np.zeros(shape, np.int32)
    for i, L in enumerate(lines, start=1):
        for a, b in zip(L[:-1], L[1:]):
            rr, cc = draw_line(int(round(a[0])), int(round(a[1])), int(round(b[0])), int(round(b[1])))
            ok = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
            inst[rr[ok], cc[ok]] = i
    gt = ndi.distance_transform_edt(inst == 0) <= GT_HALF_WIDTH
    lengths = [np.sum(np.hypot(*np.diff(L, axis=0).T)) * nm_per_px for L in lines]
    jx = junctions(lines)
    ref = {"_instances": inst, "count": len(lines),
           "length_median_nm": float(np.median(lengths)) if lengths else None,
           "width_median_nm": None, "junctions": len(jx), "_junction_xy": jx}
    return Frame(name, img01, gt, inst, nm_per_px), ref


def find_photo(name: str) -> Path:
    for d in PHOTO_DIRS:
        p = d / f"{name}.tif"
        if p.exists():
            return p
    raise SystemExit(f"photo {name}.tif not found in {PHOTO_DIRS}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labels", type=Path, default=None)
    args = ap.parse_args()
    csv_path = args.labels or max(LABEL_DIR.glob("labels_*.csv"), key=lambda p: p.stat().st_mtime,
                                  default=None)
    if csv_path is None:
        raise SystemExit(f"no labels_*.csv in {LABEL_DIR}; draw some with label_wires.py first")
    meta = json.loads(csv_path.with_suffix(".json").read_text(encoding="utf-8"))
    lines = read_lines(csv_path)
    print(f"{csv_path.name}: {len(lines)} labelled wires on {meta['photo']} ({meta['quarter']})")

    gray, fmeta = load_frame_50k(find_photo(meta["photo"]))
    full01 = gray / 255.0
    r0, c0 = meta["crop_origin_row_col"]
    h, w = meta["crop_shape_row_col"]
    crop = (slice(r0, r0 + h), slice(c0, c0 + w))
    nm = fmeta["pixel_size_m"] * 1e9 / fmeta["rescale_to_50k"]
    tag = f"{meta['photo'].replace('×', 'x')}_{meta['quarter']}"
    frame, ref = lines_to_frame(lines, (h, w), nm, tag, full01[crop])
    print(f"  reference: {ref['count']} wires, median length {ref['length_median_nm']:.0f} nm, "
          f"{ref['junctions']} junctions")

    saved = json.load(open(OUT / "results.json"))["results"]
    p = extract_params()
    from score import load_gt
    train_frames = {f.name: f for f in load_gt()}
    rows = []

    def add(name, protocol, masks):
        scores = [score_mask(m, frame, p, ref, measures_width=False) for m in masks]
        row = {"method": name, "protocol": protocol}
        for k in ("pixel_f1", "wire_f1", "wire_recall", "wire_precision", "cov_pred_over_gt",
                  "count_err_pct", "length_median_nm_err_pct", "junction_recall"):
            vals = [s[k] for s in scores if s.get(k) is not None]
            row[k] = float(np.mean(vals)) if vals else None
        rows.append(row)
        save_overlay(masks[0], frame, OUT / "external_overlays" / f"{tag}_{name}_{protocol}.png")
        print(f"  {name:16s} {protocol:8s} pixel F1 {row['pixel_f1']:.3f}  wire F1 {row['wire_f1']:.3f}  "
              f"count {row['count_err_pct']:+.0f}%", flush=True)

    clf = load_classifier()
    add("rf_v3_production", "author", [(wire_probability(clf, full01) > PROD_THR)[crop]])

    for m in METHODS:
        if m.name not in saved:
            continue
        fitted = {}
        for protocol, res in saved[m.name].items():
            masks = []
            for d in res["directions"]:
                if d["train"] not in fitted:
                    fn = m.fit(train_frames[d["train"]])
                    fitted[d["train"]] = fn(full01)
                masks.append(m.mask(full01, fitted[d["train"]], d["thr"], d["params"])[crop])
            add(m.name, protocol, masks)

    df = pd.DataFrame(rows).sort_values("wire_f1", ascending=False)
    out = OUT / f"external_{tag}.csv"
    df.to_csv(out, index=False, float_format="%.4f")
    print("\n" + df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
