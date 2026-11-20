"""A dev set in the test's style: the 5 complete-mode hand-fixed TRAINING quarters.

    venv/Scripts/python bakeoff/dev_cv.py prepare     # images + folds -> results/bakeoff/devcv/
    bash bakeoff/dev_cv.sh                            # ~2 h: E's recipe trained twice

Why: every choice after E_thin (how masks become wires, thresholds, filters)
needs photos that are labelled like the test quarters (one straight line per
wire, every wire drawn) but that no test photo is part of and that the model
did not train on. The 5 complete-mode quarters drawn with correct_training.py
--complete are exactly that once their photos are held out of training.

Two folds, fixed before any model was trained (one 30-min and one 60-min
photo on each side where possible; 299 vs 397 lines):
  fold A holds out SF251211M_50000x_48 (tr), SF251213M_50000x_20 (br)
  fold B holds out SF251211M_50000x_65 (bl), SF251211M_50000x_70 (tl), SF251213M_50000x_35 (tl)
Round 2 (E2) adds SF251211N_50000x_59 (br, 15 min) to fold A and
SF251213M_50000x_38 (bl, 60 min) to fold B: 7 quarters, 797 lines.
Each fold's model trains on E's data minus the held-out photos AND their
copy-paste copies. The copy-paste sprites come only from the two hand-outlined
80k photos (copy_paste.py), so nothing of a held-out photo leaks in.

Caveat: the hand fixes were drawn with the automatic (A/v3 agreement) label
shown in green, so the dev labels may lean slightly towards what A finds.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

from common import (LABEL_DIR, OUT, find_photo, img_to_u8, load_photo, read_json, save_u8,
                    test_photos, write_json)

DEV = OUT / "devcv"
FIX_DIR = LABEL_DIR / "train_fixes"
FOLDS = {"A": ["SF251211M_50000x_48", "SF251213M_50000x_20", "SF251211N_50000x_59"],
         "B": ["SF251211M_50000x_65", "SF251211M_50000x_70", "SF251213M_50000x_35", "SF251213M_50000x_38"]}
# round 1 (E_cv, R_cv) used the first 5 only; 59 br (A) and 38 bl (B) were drawn
# after round 1 and added for E2 (README "Round 2"). Round-1 manifest: devcv/manifest_r1.json.


def fix_meta() -> dict:
    """case -> metadata of its complete-mode hand-fix quarter."""
    out = {}
    for f in sorted(FIX_DIR.glob("*.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        if m.get("complete"):
            out[m["case"]] = {**m, "csv": str(f.with_suffix(".csv"))}
    return out


def read_fix_lines(csv_path) -> list[np.ndarray]:
    """The 'add wires' lines of a hand-fix quarter (the 'not wire' layer = unsure, left out)."""
    wires: dict[int, list] = {}
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            if row["layer"] == "add wires":
                wires.setdefault(int(row["wire_id"]), []).append((float(row["row"]), float(row["col"])))
    return [np.array(v) for _, v in sorted(wires.items())]


def unsure_lines(csv_path) -> list[np.ndarray]:
    wires: dict[int, list] = {}
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            if row["layer"] == "not wire":
                wires.setdefault(int(row["wire_id"]), []).append((float(row["row"]), float(row["col"])))
    return [np.array(v) for _, v in sorted(wires.items())]


def fold_of(case: str) -> str:
    return next(k for k, v in FOLDS.items() if case in v)


def photo_of(case: str) -> str:
    man = read_json(OUT / "train_E" / "manifest.json")
    return next(c["photo"] for c in man["cases"] if c["case"] == case)


def prepare() -> None:
    meta = fix_meta()
    cases = [c for v in FOLDS.values() for c in v]
    if set(cases) != set(meta):
        raise SystemExit(f"complete-mode quarters {sorted(meta)} != folds {sorted(cases)}")
    tests = set(test_photos())
    (DEV / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases:
        photo = photo_of(case)
        if photo in tests:
            raise SystemExit(f"{photo} is a test photo")
        img01, m = load_photo(find_photo(photo))
        save_u8(DEV / "images" / f"{case}_0000.png", img_to_u8(img01))
        rows.append({"case": case, "photo": photo, "fold": fold_of(case),
                     "quarter": meta[case]["quarter"],
                     "crop_origin_row_col": meta[case]["crop_origin_row_col"],
                     "crop_shape_row_col": meta[case]["crop_shape_row_col"],
                     "n_lines": len(read_fix_lines(meta[case]["csv"])), "csv": meta[case]["csv"],
                     "shape": list(img01.shape), "rescale_to_50k": m["rescale_to_50k"]})
        print(f"  {case} ({meta[case]['quarter']}), fold {rows[-1]['fold']}: {rows[-1]['n_lines']} lines")
    from common import COPYPASTE_E
    cp = read_json(COPYPASTE_E / "manifest.json")["cases"]
    excl = {k: v + sorted(c["case"] for c in cp if c["from"] in v) for k, v in FOLDS.items()}
    write_json(DEV / "manifest.json", {"folds": FOLDS, "exclude": excl, "cases": rows})
    for k, v in excl.items():
        (DEV / f"exclude_{k}.txt").write_text(" ".join(v), encoding="utf-8")
    print(f"wrote {DEV}")


if __name__ == "__main__":
    {"prepare": prepare}[sys.argv[1]]()
