"""Quick check before the real test labels exist: a U-Net (contender D) trained
without one hand-labelled photo, scored on that photo with compare/'s scorer,
then the same with the photos swapped.

    venv/Scripts/python bakeoff/holdout_check.py prepare    # held-out images to predict
    (train + predict with venv_dl, see holdout_check.sh)
    venv/Scripts/python bakeoff/holdout_check.py score

Same answer key, same wire step and same 160 px tile bootstrap as compare/, so
the numbers sit next to compare/RESULTS.md: path_open_clean at its fixed
setting scored pixel F1 0.627 and whole-wire F1 0.544 there (mean of the two
photos). Here it is rescored on the same masks, plus the forest trained on the
other photo (v3, p > 0.70).

Not fully unseen: the pseudo-labels on the other 41 photos came from
path_open_clean (tuned on both hand photos) and the forest (trained on both).
So D's score is slightly optimistic. The real test is score_bakeoff.py on
the new line labels.

Writes results/bakeoff/holdout/scores.csv and overlays/.
"""
from __future__ import annotations

import shutil
import sys

import numpy as np
import pandas as pd

from common import OUT, TRAIN, read_png

HOLD = OUT / "holdout"
CASES = {"TiS3_80000x_1": "HAND_TiS3_80000x_1", "TiS3_80000x_2": "HAND_TiS3_80000x_2"}
N_BOOT = 2000


def prepare() -> None:
    (HOLD / "images").mkdir(parents=True, exist_ok=True)
    for case in CASES.values():
        shutil.copyfile(TRAIN / "images" / f"{case}.png", HOLD / "images" / f"{case}_0000.png")
    print(f"wrote {HOLD / 'images'}")


def score() -> None:
    from bootstrap import pooled, tile_counts
    from methods import RF_V3
    from pseudo_labels import path_open_clean_fixed
    from score import extract_params, gt_reference, load_gt, save_overlay, score_mask
    from wire_classifier import PROD_THR

    frames = {f.name: f for f in load_gt()}
    p = extract_params()
    rows, counts = [], {}
    for k, (name, case) in enumerate(CASES.items(), start=1):
        frame = frames[name]
        other = frames[[n for n in frames if n != name][0]]
        ref = gt_reference(frame, p)
        d = read_png(HOLD / "pred" / f"D_holdout{k}" / f"{case}.png") > 127
        if d.shape != frame.gt.shape:
            raise SystemExit(f"D mask {d.shape} vs photo {frame.gt.shape}")
        masks = {"A_path_open_clean": path_open_clean_fixed(frame.img01),
                 "v3_forest_other_photo": RF_V3.fit(other)(frame.img01) > PROD_THR,
                 "D_unet": d}
        for m_name, m in masks.items():
            s = score_mask(m, frame, p, ref)
            rows.append({"photo": name, "contender": m_name,
                         **{c: s.get(c) for c in ("pixel_recall", "pixel_precision", "pixel_f1",
                                                  "wire_recall", "wire_precision", "wire_f1",
                                                  "cov_pred_over_gt", "count_err_pct",
                                                  "length_median_nm_err_pct",
                                                  "width_median_nm_err_pct", "junction_recall")}})
            counts.setdefault(m_name, {})[name] = tile_counts(m, frame, p, ref)
            save_overlay(m, frame, HOLD / "overlays" / f"{name}_{m_name}.png")
            print(f"  {name} {m_name:22s} pixel F1 {s['pixel_f1']:.3f}  wire F1 {s['wire_f1']:.3f}",
                  flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(HOLD / "scores.csv", index=False, float_format="%.4f")
    rng = np.random.default_rng(0)
    nt = {n: len(c["pix_tp"]) for n, c in counts["D_unet"].items()}
    draws = [{n: rng.integers(0, k, k) for n, k in nt.items()} for _ in range(N_BOOT)]
    boot = {m: np.array([pooled(c, dr) for dr in draws]) for m, c in counts.items()}

    print("\nmean of the two held-out photos (pooled F1 and 95% interval of the gain over A in brackets)")
    print(f"{'contender':22s} {'pixel F1':>8s} {'wire F1':>8s} {'marked/lab':>10s} {'count err':>9s} "
          f"{'length err':>10s} {'width err':>9s}  {'wire F1 vs A':>26s}")
    for m in counts:
        sub = df[df.contender == m]
        pix, wf = pooled(counts[m])
        if m == "A_path_open_clean":
            vs = "-"
        else:
            dd = boot[m][:, 1] - boot["A_path_open_clean"][:, 1]
            vs = (f"{dd.mean():+.3f} [{np.percentile(dd, 2.5):+.3f}, {np.percentile(dd, 97.5):+.3f}]"
                  f" P>A {np.mean(dd > 0):.2f}")
        print(f"{m:22s} {sub.pixel_f1.mean():8.3f} {sub.wire_f1.mean():8.3f} "
              f"{sub.cov_pred_over_gt.mean():10.2f} {sub.count_err_pct.abs().mean():8.0f}% "
              f"{sub.length_median_nm_err_pct.abs().mean():9.0f}% "
              f"{sub.width_median_nm_err_pct.abs().mean():8.0f}%  {vs:>26s}")
    print(f"\nwrote {HOLD / 'scores.csv'} and {HOLD / 'overlays'}")


if __name__ == "__main__":
    {"prepare": prepare, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "score"]()
