"""Label TiS3 wires as lines in napari: two clicks per wire.

    ..\\..\\..\\labelling_env\\Scripts\\python label_wires.py              # default photo + quarter
    ..\\..\\..\\labelling_env\\Scripts\\python label_wires.py --quarter tr # another quarter

Opens one quarter (768 x 512 px) of a 50,000x photo from any TiS3 growth run
(--photo SF251211N_50000x_22 for the 15-min run, for example), with a 'wires'
layer already in line mode. Every wire you draw is one line from one visible
end to the other. Your lines are saved to labels_<photo>_<quarter>.csv (+ .json)
when you press Ctrl+S and when you close the window. Run the same command again
to continue where you stopped.

For the model bake-off, label only photos listed in
Ridge-Filtering-Nanowires-main/bakeoff/test_photos.txt, and draw every line
yourself: lines pre-drawn by a method would favour that method in the test.
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
RUNS = HERE.parents[1] / "02_Data" / "SEM images" / "TiS3 on sapphire"
DEFAULT_PHOTO = "SF251213M_50000×_37"
NM_PER_PX = 5.396                     # FEI metadata for every 50,000x photo, all runs
DATA_ROWS = 1024                      # rows above the information bar
QUARTERS = {"tl": (0, 0), "tr": (0, 768), "bl": (512, 0), "br": (512, 768)}


def find_photo(photo: str) -> Path:
    if "_50000×_" not in photo:
        raise SystemExit(f"{photo}: only 50,000x photos can be labelled (5.4 nm/px)")
    hits = sorted(RUNS.glob(f"*/{photo}.tif"))
    if not hits:
        raise SystemExit(f"{photo}.tif not found in any run folder under {RUNS}")
    return hits[0]


def load_crop(photo: str, quarter: str) -> tuple[np.ndarray, tuple[int, int]]:
    a = np.array(Image.open(find_photo(photo))).astype(np.float32)[:DATA_ROWS]
    r0, c0 = QUARTERS[quarter]
    return a[r0:r0 + 512, c0:c0 + 768], (r0, c0)


def out_paths(photo: str, quarter: str) -> tuple[Path, Path]:
    stem = f"labels_{photo.replace('×', 'x')}_{quarter}"
    return HERE / f"{stem}.csv", HERE / f"{stem}.json"


def save(shapes, photo: str, quarter: str, origin: tuple[int, int]) -> int:
    csv_path, meta_path = out_paths(photo, quarter)
    n = 0
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["wire_id", "vertex", "row", "col"])
        for data, kind in zip(shapes.data, shapes.shape_type):
            if kind not in ("line", "path") or len(data) < 2:
                continue
            for v, (r, c) in enumerate(np.asarray(data)[:, -2:]):
                w.writerow([n, v, round(float(r), 2), round(float(c), 2)])
            n += 1
    meta_path.write_text(json.dumps({
        "photo": photo, "quarter": quarter, "crop_origin_row_col": list(origin),
        "crop_shape_row_col": [512, 768], "nm_per_px": NM_PER_PX,
        "coordinates": "row, col in the crop (0, 0 = top-left of the quarter)",
        "n_wires": n}, indent=2), encoding="utf-8")
    return n


def load_existing(photo: str, quarter: str) -> list[np.ndarray]:
    csv_path, _ = out_paths(photo, quarter)
    if not csv_path.exists():
        return []
    wires: dict[int, list] = {}
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            wires.setdefault(int(row["wire_id"]), []).append((float(row["row"]), float(row["col"])))
    return [np.array(v) for _, v in sorted(wires.items())]


def main() -> None:
    import napari

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--photo", default=DEFAULT_PHOTO)
    ap.add_argument("--quarter", default="tl", choices=sorted(QUARTERS))
    args = ap.parse_args()
    args.photo = re.sub(r"(\d)x_", r"\1×_", args.photo)     # accept a plain x for ×

    img, origin = load_crop(args.photo, args.quarter)
    lo, hi = np.percentile(img, [1, 99])
    viewer = napari.Viewer(title=f"{args.photo} ({args.quarter}) - draw one line per wire, "
                                 f"Ctrl+S saves, closing saves")
    viewer.add_image(img, name="photo", contrast_limits=(float(lo), float(hi)))
    existing = load_existing(args.photo, args.quarter)
    shapes = viewer.add_shapes(existing, shape_type=["path" if len(w) > 2 else "line" for w in existing] or "line",
                               name="wires", edge_width=1.5, edge_color="yellow")
    shapes.current_edge_color = "yellow"
    shapes.current_edge_width = 1.5
    shapes.mode = "add_line"

    @viewer.bind_key("Control-S", overwrite=True)
    def _save(_viewer):
        n = save(shapes, args.photo, args.quarter, origin)
        _viewer.status = f"saved {n} wires"
        print(f"saved {n} wires")

    print(f"{len(existing)} wires loaded. Draw lines; Ctrl+S to save; close the window when done.")
    napari.run()
    n = save(shapes, args.photo, args.quarter, origin)
    print(f"saved {n} wires to {out_paths(args.photo, args.quarter)[0]}")


if __name__ == "__main__":
    main()
