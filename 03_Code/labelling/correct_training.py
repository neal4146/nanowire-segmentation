"""Correct the automatic labels of a TRAINING photo in napari (contender E).

    ..\\..\\..\\labelling_env\\Scripts\\python correct_training.py --list
    ..\\..\\..\\labelling_env\\Scripts\\python correct_training.py --photo SF251211N_50000x_27 --quarter tl

Opens one quarter of a training photo (50k-equivalent scale, as the networks
see it) with the automatic training label on top in green: what A and the
forest both call wire. Toggle the green layer (eye icon) to see the photo
underneath. Then, with the line tool, as for the test labels:

  'add wires'  (yellow, selected at start)  one line per wire the green MISSES:
               faint wires, short rods, bright rods, the far wire of a fused
               pair. Leave alone what is already green.
  'not wire'   (red)  a line over green that is NOT a wire: a blob, a smear,
               the gap between two wires that the green fused into one.

--complete: you draw EVERY wire in the quarter (green or not), exactly as for
the test labels. Your lines then replace the automatic label in that quarter:
lines = wire, everything else = background, so the green's false wires are
fixed too. A wire you skip becomes background, so it must really be all of
them. The red layer becomes 'unsure': a line over anything you cannot decide,
which is then ignored in training. A quarter saved once with --complete stays
complete.

Close the window to save (Ctrl+S opens napari's own dialog: cancel it).
Lines go to train_fixes/<photo>_<quarter>.csv (+ .json); running the same
command again continues where you stopped. Then rebuild E's training data:

    cd ..\\..\\..\\Ridge-Filtering-Nanowires-main
    venv\\Scripts\\python bakeoff\\build_e_data.py

Test photos (bakeoff/test_photos.txt) are refused: they must never train.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2] / "Ridge-Filtering-Nanowires-main"
TRAIN = REPO / "results" / "bakeoff" / "train"
TEST_LIST = REPO / "bakeoff" / "test_photos.txt"
FIX_DIR = HERE / "train_fixes"
LAYERS = {"add wires": "yellow", "not wire": "red"}
# suggested set: 50k photos (same scale as the test labels), all three growth runs,
# more from the 15-min run whose faint wires A misses most
SUGGESTED = [("SF251211N_50000x_27", "tl"), ("SF251211N_50000x_31", "br"),
             ("SF251211N_50000x_5", "tr"), ("SF251211M_50000x_48", "tr"),
             ("SF251211M_50000x_65", "bl"), ("SF251211M_50000x_70", "tl"),
             ("SF251213M_50000x_20", "br"), ("SF251213M_50000x_35", "tl")]


def case_of(photo: str) -> str:
    return photo.strip().replace("×", "x")


def test_cases() -> set[str]:
    out = set()
    for line in TEST_LIST.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line:
            out.add(case_of(line))
    return out


def quarter_box(shape, quarter: str) -> tuple[int, int, int, int]:
    H, W = shape
    h, w = H // 2, W // 2
    r0 = 0 if quarter[0] == "t" else h
    c0 = 0 if quarter[1] == "l" else w
    return r0, c0, (h if r0 == 0 else H - h), (w if c0 == 0 else W - w)


def out_paths(case: str, quarter: str) -> tuple[Path, Path]:
    stem = f"{case}_{quarter}"
    return FIX_DIR / f"{stem}.csv", FIX_DIR / f"{stem}.json"


def load_existing(case: str, quarter: str) -> dict[str, list[np.ndarray]]:
    csv_path, _ = out_paths(case, quarter)
    out: dict[str, dict[int, list]] = {k: {} for k in LAYERS}
    if csv_path.exists():
        with open(csv_path) as fh:
            for row in csv.DictReader(fh):
                out[row["layer"]].setdefault(int(row["wire_id"]), []).append(
                    (float(row["row"]), float(row["col"])))
    return {k: [np.array(v) for _, v in sorted(d.items())] for k, d in out.items()}


def is_complete(case: str, quarter: str) -> bool:
    meta_path = out_paths(case, quarter)[1]
    return meta_path.exists() and bool(json.loads(meta_path.read_text(encoding="utf-8")).get("complete"))


def save(layers: dict, case: str, quarter: str, box, complete: bool) -> dict[str, int]:
    FIX_DIR.mkdir(exist_ok=True)
    csv_path, meta_path = out_paths(case, quarter)
    counts = {}
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["layer", "wire_id", "vertex", "row", "col"])
        for name, shapes in layers.items():
            n = 0
            for data, kind in zip(shapes.data, shapes.shape_type):
                if kind not in ("line", "path") or len(data) < 2:
                    continue
                for v, (r, c) in enumerate(np.asarray(data)[:, -2:]):
                    w.writerow([name, n, v, round(float(r), 2), round(float(c), 2)])
                n += 1
            counts[name] = n
    r0, c0, h, w_ = box
    meta_path.write_text(json.dumps({
        "case": case, "quarter": quarter, "complete": complete,
        "mode": ("complete: every wire drawn ('add wires'); 'not wire' = unsure, ignored"
                 if complete else "add-only: 'add wires' = missed wires, 'not wire' = false green"),
        "crop_origin_row_col": [r0, c0],
        "crop_shape_row_col": [h, w_], "scale": "50k-equivalent training image (results/bakeoff/train)",
        "coordinates": "row, col in the crop (0, 0 = top-left of the quarter)",
        "counts": counts}, indent=2), encoding="utf-8")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--photo", help="training photo, e.g. SF251211N_50000x_27")
    ap.add_argument("--quarter", default="tl", choices=["tl", "tr", "bl", "br"])
    ap.add_argument("--complete", action="store_true",
                    help="draw EVERY wire; the lines replace the automatic label in this quarter")
    ap.add_argument("--list", action="store_true", help="show the suggested quarters and progress")
    args = ap.parse_args()

    if args.list or not args.photo:
        print("suggested training quarters (saved = has a train_fixes file):")
        for case, q in SUGGESTED:
            st = ("complete" if is_complete(case, q) else "add-only") \
                if out_paths(case, q)[0].exists() else ""
            print(f"  {st:8s}  --photo {case} --quarter {q}")
        return

    case = case_of(re.sub(r"(\d)×_", r"\1x_", args.photo))
    if case in test_cases():
        raise SystemExit(f"{case} is a TEST photo: it must never be used for training")
    img_path, lab_path = TRAIN / "images" / f"{case}.png", TRAIN / "labels" / f"{case}.png"
    if not img_path.exists():
        raise SystemExit(f"{case} is not in the training set ({img_path} missing; "
                         f"run bakeoff/pseudo_labels.py first)")
    img = np.array(Image.open(img_path)).astype(np.float32)
    lab = np.array(Image.open(lab_path))
    box = quarter_box(img.shape, args.quarter)
    r0, c0, h, w = box
    crop = (slice(r0, r0 + h), slice(c0, c0 + w))

    complete = args.complete or is_complete(case, args.quarter)
    existing = load_existing(case, args.quarter)
    if args.complete and not is_complete(case, args.quarter) and out_paths(case, args.quarter)[0].exists():
        print("note: this quarter was saved add-only; closing now saves it as COMPLETE, so every "
              "wire must be drawn (the lines you already have are kept)")

    import napari
    lo, hi = np.percentile(img[crop], [1, 99])
    title = (f"TRAINING {case} ({args.quarter}) COMPLETE - yellow: draw EVERY wire, "
             f"red: unsure (ignored); close to save" if complete else
             f"TRAINING {case} ({args.quarter}) - yellow: add missed wires, red: not a wire; close to save")
    shown = ({"add wires": "ALL wires (yellow)", "not wire": "unsure (red)"} if complete
             else {k: k for k in LAYERS})
    viewer = napari.Viewer(title=title)
    viewer.add_image(img[crop], name="photo", contrast_limits=(float(lo), float(hi)))
    viewer.add_image((lab[crop] == 1).astype(np.float32), name="auto label (green = wire)",
                     colormap="green", blending="additive", opacity=0.3 if complete else 0.45,
                     contrast_limits=(0, 1))
    layers = {}
    for name in ["not wire", "add wires"]:                  # 'add wires' ends up selected
        col, data = LAYERS[name], existing[name]
        sh = viewer.add_shapes(data, shape_type=["path" if len(d) > 2 else "line" for d in data] or "line",
                               name=shown[name], edge_width=1.5, edge_color=col)
        sh.current_edge_color, sh.current_edge_width = col, 1.5
        sh.mode = "add_line"
        layers[name] = sh
    print(f"{case} {args.quarter} ({'COMPLETE' if complete else 'add-only'}): "
          f"{len(existing['add wires'])} wire lines and {len(existing['not wire'])} red lines loaded. "
          f"Close the window to save.")
    napari.run()
    counts = save(layers, case, args.quarter, box, complete)
    print(f"saved {counts['add wires']} wire lines and {counts['not wire']} red lines "
          f"({'complete' if complete else 'add-only'}) to {out_paths(case, args.quarter)[0]}")


if __name__ == "__main__":
    main()
