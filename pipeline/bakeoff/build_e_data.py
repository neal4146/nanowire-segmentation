"""Training data for contender E: B/C's data with the three fixes for what the
agreement pseudo-labels teach wrongly (faint wires, short rods and bright rods
labelled background, because neither A nor the forest sees them).

    venv/Scripts/python bakeoff/build_e_data.py              # all three fixes (~3 min)
    venv/Scripts/python bakeoff/build_e_data.py --no-paste   # fixes 1 and 3 only

  1  faint ignore   pseudo_labels.faint_candidates: line-shaped things above
                    the photo's own noise level, 6+ px long, that A is not
                    sensitive enough to see. Their agreed background becomes
                    IGNORE, so the network is no longer told they are not wires.
  2  faint paste    copy_paste.py --faint: real hand-drawn wires pasted at
                    2-3.5 noise units of contrast (40% cut to 6-20 px), labelled
                    WIRE. Written to results/bakeoff/copypaste_E/ together with
                    C's pairs and crossings.
  3  hand fixes     lines drawn with 03_Code/labelling/correct_training.py on
                    training photos: 'add wires' -> a 3 px WIRE core with an
                    IGNORE band out to 6 px (a line says where a wire runs, not
                    how wide it is); 'not wire' -> IGNORE 4 px either side.
                    A quarter saved with --complete (every wire drawn) replaces
                    the automatic label there: background except the lines, so
                    the pseudo-labels' false wires are fixed too. Automatic wire
                    within 12 px of a line is IGNORE (wide rods, line placement);
                    red lines mean 'unsure' -> IGNORE. Check image per complete
                    quarter: previews/<photo>_<quarter>_complete.png.
                    Those photos are marked 'corrected' and drawn --corrected-copies
                    times as often in training (train_unet.py).

Reads results/bakeoff/train/ (run pseudo_labels.py first) and writes
results/bakeoff/train_E/ in the same layout (images/, labels/, previews/,
manifest.json). B, C and D's data are not touched. Test photos are refused.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from common import (BG, COPYPASTE_E, IGNORE, TRAIN, TRAIN_E, TRAIN_FIX_DIR, WIRE, case_name,
                    read_json, read_png, save_u8, test_photos, write_json)

CORE = 3        # px: thickness of the WIRE line drawn for an added wire
BAND = 13       # px: thickness of the IGNORE band around it (6 px either side)
ERASE = 9       # px: thickness of the IGNORE line drawn for 'not wire'
NEAR = 25       # px: complete quarters: automatic wire within 12 px of a line -> IGNORE, beyond -> BG


def read_fixes(case: str) -> list[tuple[dict, dict[str, list[np.ndarray]]]]:
    """[(meta, {'add wires': [lines], 'not wire': [lines]})] for every fixed quarter."""
    out = []
    for csv_path in sorted(TRAIN_FIX_DIR.glob(f"{case}_*.csv")):
        meta = read_json(csv_path.with_suffix(".json"))
        if meta["case"] != case:
            continue                        # e.g. SF..._5 must not pick up SF..._59
        lines: dict[str, dict[int, list]] = {"add wires": {}, "not wire": {}}
        with open(csv_path) as fh:
            for row in csv.DictReader(fh):
                lines[row["layer"]].setdefault(int(row["wire_id"]), []).append(
                    (float(row["row"]), float(row["col"])))
        out.append((meta, {k: [np.array(v) for _, v in sorted(d.items())] for k, d in lines.items()}))
    return out


def draw(mask_shape, lines, origin, thickness) -> np.ndarray:
    m = np.zeros(mask_shape, np.uint8)
    r0, c0 = origin
    for ln in lines:
        pts = np.round(ln[:, ::-1] + [c0, r0]).astype(np.int32)    # (x, y) in the full photo
        cv2.polylines(m, [pts], False, 1, thickness=thickness, lineType=cv2.LINE_8)
    return m.astype(bool)


def quarter_slices(meta) -> tuple[slice, slice]:
    (r0, c0), (h, w) = meta["crop_origin_row_col"], meta["crop_shape_row_col"]
    return slice(r0, r0 + h), slice(c0, c0 + w)


def apply_fixes(lab: np.ndarray, fixes) -> tuple[np.ndarray, int, int]:
    lab = lab.copy()
    n_add = n_not = 0
    for meta, lines in fixes:
        origin = meta["crop_origin_row_col"]
        if meta.get("complete"):
            # every wire is drawn: the automatic label is replaced by background,
            # except automatic wire close to a line (its width is not drawn)
            box = quarter_slices(meta)
            near = draw(lab.shape, lines["add wires"], origin, NEAR)[box]
            q = np.full(lab[box].shape, BG, lab.dtype)
            q[(lab[box] == WIRE) & near] = IGNORE
            lab[box] = q
        if lines["not wire"]:
            lab[draw(lab.shape, lines["not wire"], origin, ERASE)] = IGNORE
        if lines["add wires"]:
            band = draw(lab.shape, lines["add wires"], origin, BAND)
            lab[band & (lab == BG)] = IGNORE
            lab[draw(lab.shape, lines["add wires"], origin, CORE)] = WIRE
        n_add += len(lines["add wires"])
        n_not += len(lines["not wire"])
    return lab, n_add, n_not


def preview(img: np.ndarray, base: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """Gray photo; green = wire, magenta = ignored before, orange = newly ignored,
    yellow = wire added by hand."""
    rgb = np.stack([img] * 3, axis=2).astype(np.float32)
    old = base == IGNORE
    new = (lab == IGNORE) & ~old
    rgb[old] = 0.6 * rgb[old] + 0.4 * np.array([255, 0, 255])
    rgb[new] = 0.4 * rgb[new] + 0.6 * np.array([255, 160, 0])
    rgb[lab == WIRE] = (40, 210, 40)
    rgb[(lab == WIRE) & (base != WIRE)] = (255, 230, 0)
    return rgb.astype(np.uint8)


def complete_check(img: np.ndarray, base: np.ndarray, lab: np.ndarray, meta) -> np.ndarray:
    """One complete quarter at 2x: photo | yellow = your wires, magenta = ignored,
    RED = automatic wire that is now background. Every red bit should be a non-wire;
    a red wire means it was skipped."""
    box = quarter_slices(meta)
    g, b, L = img[box], base[box], lab[box]
    rgb = np.stack([g] * 3, axis=2).astype(np.float32)
    rgb[L == IGNORE] = 0.6 * rgb[L == IGNORE] + 0.4 * np.array([255, 0, 255])
    rgb[L == WIRE] = (255, 230, 0)
    gone = (b == WIRE) & (L == BG)
    rgb[gone] = (255, 0, 0)
    pair = np.concatenate([np.stack([g] * 3, axis=2), np.full((g.shape[0], 10, 3), 255),
                           rgb.astype(np.uint8)], axis=1).astype(np.uint8)
    return cv2.resize(pair, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-paste", action="store_true", help="skip fix 2 (copypaste_E/)")
    ap.add_argument("--faint", type=int, default=15, help="faint wires pasted per photo")
    ap.add_argument("--events", type=int, default=30, help="pair/crossing attempts per photo (as C)")
    ap.add_argument("--force", action="store_true", help="recompute the cached faint masks")
    args = ap.parse_args()

    from PIL import Image
    from pseudo_labels import faint_candidates

    tests = {case_name(p) for p in test_photos()}
    manifest = read_json(TRAIN / "manifest.json")
    for d in ("images", "labels", "previews", "faint"):
        (TRAIN_E / d).mkdir(parents=True, exist_ok=True)
    fixed_cases = {read_json(p)["case"] for p in TRAIN_FIX_DIR.glob("*.json")} \
        if TRAIN_FIX_DIR.exists() else set()
    bad = fixed_cases & tests
    if bad:
        raise SystemExit(f"train_fixes/ has TEST photos {sorted(bad)}: delete those files")
    unknown = fixed_cases - {c["case"] for c in manifest["cases"]}
    if unknown:
        print(f"warning: fixes for photos not in the training set, ignored: {sorted(unknown)}")

    cases = []
    for c in manifest["cases"]:
        case = c["case"]
        if case in tests:
            raise SystemExit(f"{case} is a test photo but is in train/manifest.json")
        img = read_png(TRAIN / "images" / f"{case}.png")
        base = read_png(TRAIN / "labels" / f"{case}.png")
        shutil.copyfile(TRAIN / "images" / f"{case}.png", TRAIN_E / "images" / f"{case}.png")
        if c["source"] == "hand":
            save_u8(TRAIN_E / "labels" / f"{case}.png", base)
            cases.append(dict(c))
            continue
        t0 = time.perf_counter()
        fpath = TRAIN_E / "faint" / f"{case}.png"
        if fpath.exists() and not args.force:
            faint = read_png(fpath) > 0
        else:
            faint = faint_candidates(img / 255.0, base == BG)
            save_u8(fpath, faint.astype(np.uint8) * 255)
        lab = base.copy()
        lab[faint & (lab == BG)] = IGNORE
        fixes = read_fixes(case)
        lab, n_add, n_not = apply_fixes(lab, fixes)
        save_u8(TRAIN_E / "labels" / f"{case}.png", lab)
        Image.fromarray(preview(img, base, lab)).save(TRAIN_E / "previews" / f"{case}.png")
        complete = [m["quarter"] for m, _ in fixes if m.get("complete")]
        for m, _ in fixes:
            if m.get("complete"):
                Image.fromarray(complete_check(img, base, lab, m)).save(
                    TRAIN_E / "previews" / f"{case}_{m['quarter']}_complete.png")
        src = "corrected" if fixes else c["source"]
        cases.append({**c, "source": src, "wire": float((lab == WIRE).mean()),
                      "ignore": float((lab == IGNORE).mean()),
                      "ignore_before": c["ignore"], "fixed_quarters": [m["quarter"] for m, _ in fixes],
                      "complete_quarters": complete, "added_wires": n_add, "not_wire_lines": n_not})
        note = (f"  + {len(fixes)} hand-fixed quarter(s) ({len(complete)} complete): "
                f"{n_add} wire lines, {n_not} red lines") if fixes else ""
        print(f"  {case:24s} ignore {c['ignore'] * 100:5.1f}% -> {cases[-1]['ignore'] * 100:5.1f}%"
              f"  ({time.perf_counter() - t0:3.0f} s){note}", flush=True)

    pseudo = [c for c in cases if c["source"] in ("pseudo", "corrected")]
    write_json(TRAIN_E / "manifest.json", {
        "settings": {**manifest["settings"], "built_from": "results/bakeoff/train",
                     "fix1_faint_ignore": "pseudo_labels.faint_candidates defaults",
                     "fix3_hand_fixes": {"dir": str(TRAIN_FIX_DIR), "core_px": CORE,
                                         "band_px": BAND, "erase_px": ERASE, "near_px": NEAR}},
        "test_photos_excluded": manifest.get("test_photos_excluded", []),
        "cases": cases})
    n_fix = sum(c["source"] == "corrected" for c in cases)
    print(f"\n{len(pseudo)} pseudo-labelled photos, {n_fix} with hand fixes; mean ignore "
          f"{np.mean([c['ignore_before'] for c in pseudo]) * 100:.1f}% -> "
          f"{np.mean([c['ignore'] for c in pseudo]) * 100:.1f}%. Wrote {TRAIN_E}")

    if not args.no_paste:
        here = Path(__file__).resolve().parent
        cmd = [sys.executable, str(here / "copy_paste.py"), "--src", str(TRAIN_E),
               "--dst", str(COPYPASTE_E), "--faint", str(args.faint), "--events", str(args.events)]
        print("\nfix 2: " + " ".join(cmd[1:]), flush=True)
        subprocess.run(cmd, check=True, cwd=here)


if __name__ == "__main__":
    main()
