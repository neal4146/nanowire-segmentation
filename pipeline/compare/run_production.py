"""Run the race winner (path_open_clean, one fixed setting) through the full
production pipeline on every photo in data/real/NanowireImages2, and compare it
photo by photo with Nikhil's v3 output.

    venv/Scripts/python compare/run_production.py

Steps:
  1. Score the fixed setting (methods.PATH_OPEN_CLEAN author_thr/params) on the
     two labelled photos, cross-frame, so its numbers are known.
  2. For each photo: load_frame_50k (same scale chain as production), mask, and
     write <name>_50k_pseudo.png + overlay + ridge_coverage.json in the format
     ridge_nanowire_images2.py uses, into results/path_open_clean/
     (Nikhil's v3 masks in results/ridge_nanowire_images2/ are not touched).
  3. Run scripts/ridge/wire_extract.py unchanged on those masks, writing
     results/wire_extract_path_open_clean/.
  4. Compare with v3 (results/wire_extract/extract_summary.json and
     results/ridge_nanowire_images2/): coverage, wire counts, median length and
     width, junctions, direction-family shares, plus how much the two masks and
     the two sets of wires agree with each other. Writes
     results/compare/production_vs_v3.csv.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from score import (OUT, chains_to_instances, evaluate, extract_params,  # noqa: E402
                   gt_reference, load_gt)
from ridge_nanowire_images2 import load_frame_50k                      # noqa: E402  (path set by score)
import wire_extract                                                     # noqa: E402
from methods import PATH_OPEN_CLEAN                                     # noqa: E402
from wire_metrics import centreline_scores                              # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "real" / "NanowireImages2"
MASKS = ROOT / "results" / "path_open_clean"
EXTRACT = ROOT / "results" / "wire_extract_path_open_clean"
V3_MASKS = ROOT / "results" / "ridge_nanowire_images2"
V3_EXTRACT = ROOT / "results" / "wire_extract"


def score_fixed_setting() -> dict:
    frames = load_gt()
    p = extract_params()
    refs = {f.name: gt_reference(f, p) for f in frames}
    res = evaluate(PATH_OPEN_CLEAN, frames, p, refs, protocols=("author",), overlays=False)
    m = res["author"]["mean"]
    print(f"fixed setting on the labelled photos (cross-frame): pixel F1 {m['pixel_f1']:.3f}, "
          f"wire F1 {m['wire_f1']:.3f}, count err {m['count_abs_err_pct']:.0f}%, "
          f"marked/labelled {m['cov_pred_over_gt']:.2f}")
    return m


def make_masks() -> None:
    MASKS.mkdir(parents=True, exist_ok=True)
    per_frame = {}
    for path in sorted(SRC.glob("*.tif")):
        gray, meta = load_frame_50k(path)
        img01 = gray / 255.0
        resp = PATH_OPEN_CLEAN.response(img01)
        mask = PATH_OPEN_CLEAN.mask(img01, resp, PATH_OPEN_CLEAN.author_thr,
                                    PATH_OPEN_CLEAN.author_params)
        Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(
            MASKS / f"{path.stem}_50k_pseudo.png")
        rgb = np.stack([np.clip(gray, 0, 255).astype(np.uint8)] * 3, axis=2)
        rgb[mask] = (0, 220, 255)
        Image.fromarray(rgb).save(MASKS / f"{path.stem}_50k_overlay.png")
        meta.update(coverage=float(mask.mean()), shape_50k=list(gray.shape), overlay_scale=1.0)
        per_frame[path.stem] = meta
        print(f"  {path.stem:26s} coverage {meta['coverage'] * 100:5.2f}%"
              f"{'' if meta['scale_in_validated_band'] else '  [outside validated band]'}",
              flush=True)
    with open(MASKS / "ridge_coverage.json", "w") as fh:
        json.dump({"source": str(SRC.relative_to(ROOT)),
                   "pipeline": "path_open_clean (compare/methods.py), fixed setting",
                   "setting": {"thr": PATH_OPEN_CLEAN.author_thr, **PATH_OPEN_CLEAN.author_params},
                   "per_frame": per_frame}, fh, indent=2)


def run_wire_extract() -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / "ridge" / "wire_extract.py"),
           "--masks-dir", str(MASKS), "--out", str(EXTRACT)]
    subprocess.run(cmd, check=True, cwd=ROOT)


def _mask(folder: Path, name: str) -> np.ndarray:
    return np.array(Image.open(folder / f"{name}_50k_pseudo.png")) > 127


def _chains(mask: np.ndarray, name: str, p) -> np.ndarray:
    meta = json.load(open(MASKS / "ridge_coverage.json"))["per_frame"][name]
    nm = meta["pixel_size_m"] * 1e9 / meta["rescale_to_50k"]
    records, _, _ = wire_extract.extract_from_mask(mask, nm, p)
    return chains_to_instances(records, mask.shape)


def compare() -> pd.DataFrame:
    new = json.load(open(EXTRACT / "extract_summary.json"))["per_frame"]
    old = json.load(open(V3_EXTRACT / "extract_summary.json"))["per_frame"]
    cov_new = json.load(open(MASKS / "ridge_coverage.json"))["per_frame"]
    cov_old = json.load(open(V3_MASKS / "ridge_coverage.json"))["per_frame"]
    p = extract_params()
    rows = []
    for name in sorted(set(new) & set(old)):
        a, b = old[name], new[name]
        ma, mb = _mask(V3_MASKS, name), _mask(MASKS, name)
        dice = 2 * (ma & mb).sum() / max(ma.sum() + mb.sum(), 1)
        w = centreline_scores(_chains(ma, name, p), _chains(mb, name, p))
        row = {"photo": name,
               "coverage_v3": cov_old[name]["coverage"], "coverage_new": cov_new[name]["coverage"],
               "wires_v3": a["wire_count_all"], "wires_new": b["wire_count_all"],
               "wires_excl_border_v3": a["wire_count_excl_border"],
               "wires_excl_border_new": b["wire_count_excl_border"],
               "length_med_nm_v3": a["length_nm_median_all"], "length_med_nm_new": b["length_nm_median_all"],
               "width_med_nm_v3": a["width_mean_nm_median"], "width_med_nm_new": b["width_mean_nm_median"],
               "junctions_v3": a["junction_count"], "junctions_new": b["junction_count"],
               "mask_dice": dice, "wire_agreement_f1": w["wire_f1"]}
        for fam in ("0.0", "60.0", "123.5"):
            row[f"family_{fam}_v3"] = a["family_fractions_all"].get(fam)
            row[f"family_{fam}_new"] = b["family_fractions_all"].get(fam)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    print("1. scoring the fixed setting")
    score_fixed_setting()
    print("2. masks for every photo")
    make_masks()
    print("3. wire_extract (unchanged) on the new masks")
    run_wire_extract()
    print("4. comparison with v3")
    df = compare()
    df.to_csv(OUT / "production_vs_v3.csv", index=False, float_format="%.4f")
    show = df[["photo", "coverage_v3", "coverage_new", "wires_v3", "wires_new",
               "length_med_nm_v3", "length_med_nm_new", "width_med_nm_v3", "width_med_nm_new",
               "junctions_v3", "junctions_new", "mask_dice", "wire_agreement_f1"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    tot = df[["wires_v3", "wires_new", "junctions_v3", "junctions_new"]].sum()
    print(f"\ntotal wires v3 {tot.wires_v3} vs new {tot.wires_new} "
          f"({100 * (tot.wires_new / tot.wires_v3 - 1):+.0f}%); median length "
          f"v3 {df.length_med_nm_v3.median():.0f} nm vs new {df.length_med_nm_new.median():.0f} nm")
    print(f"wrote {OUT / 'production_vs_v3.csv'}")


if __name__ == "__main__":
    main()
