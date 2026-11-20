"""Score every bake-off contender on the line-labelled test quarters.

    venv/Scripts/python bakeoff/score_bakeoff.py

For every labels_*.csv drawn with label_wires.py (Materials_ThangPham/03_Code/labelling/):
  - the photo must be listed in bakeoff/test_photos.txt and absent from the
    training manifest; a labelled photo that fails this is reported and left out
  - ground truth from the lines, built exactly as compare/external.py builds it
    (one wire per line, crossings stay whole; pixel mask = lines widened to 8 px)
  - contenders, each run on the whole photo and cropped to the quarter:
      A_path_open_clean  path_open_clean at its fixed production setting: the bar to beat
      v3_production      Nikhil's production forest, p > 0.70 (reference)
      <name>             every results/bakeoff/pred/<name>/ folder (predict.py)
  - scores from compare/score.score_mask: pixel F1, whole-wire F1, wire count
    and median length error, junction recall

Pooled over all quarters (micro average, as compare/bootstrap.py does), with 95%
intervals from resampling 160 px tiles within each quarter. Every contender
gets the same draws, so differences are paired: d_wire_f1_vs_A and
p_better_wire_vs_A say how much and how often each beats A.
A contender wins only if its d_wire_f1_vs_A interval sits above zero.

Writes results/bakeoff/scores_per_quarter.csv, summary.csv and overlays/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (LABEL_DIR, OUT, PRED, TRAIN, TRAIN_E, case_name, find_photo, load_photo, photo_name,
                    read_json, read_png, run_of, test_photos)

N_BOOT = 2000
SEED = 0
REF = "A_path_open_clean"


def label_sets() -> list[tuple]:
    sets = []
    for csv_path in sorted(LABEL_DIR.glob("labels_*.csv")):
        meta = read_json(csv_path.with_suffix(".json"))
        sets.append((csv_path, meta))
    return sets


def main() -> None:
    from bootstrap import pooled, tile_counts
    from external import lines_to_frame, read_lines
    from pseudo_labels import path_open_clean_fixed
    from score import extract_params, save_overlay, score_mask
    from wire_classifier import PROD_THR, load_classifier, wire_probability

    tests = set(test_photos())
    trained_on = {c["photo"] for t in (TRAIN, TRAIN_E) if (t / "manifest.json").exists()
                  for c in read_json(t / "manifest.json")["cases"]}
    pred_dirs = sorted(d for d in PRED.iterdir() if d.is_dir()) if PRED.exists() else []
    p = extract_params()
    clf = None
    rows, counts, frames = [], {}, {}

    for csv_path, meta in label_sets():
        photo = photo_name(meta["photo"])
        tag = f"{case_name(photo)}_{meta['quarter']}"
        if photo not in tests or photo in trained_on:
            why = "not in test_photos.txt" if photo not in tests else "used in training"
            print(f"SKIPPED {csv_path.name}: {photo} is {why}")
            continue
        lines = read_lines(csv_path)
        img01, fmeta = load_photo(find_photo(photo))
        if abs(fmeta["rescale_to_50k"] - 1) > 1e-3:
            raise SystemExit(f"{photo} is not a 50,000x photo; the line labels are in native pixels")
        r0, c0 = meta["crop_origin_row_col"]
        h, w = meta["crop_shape_row_col"]
        crop = (slice(r0, r0 + h), slice(c0, c0 + w))
        nm = fmeta["pixel_size_m"] * 1e9 / fmeta["rescale_to_50k"]
        frame, ref = lines_to_frame(lines, (h, w), nm, tag, img01[crop])
        frames[tag] = frame
        print(f"{tag}: {len(lines)} labelled wires ({run_of(photo)})", flush=True)

        if clf is None:
            clf = load_classifier()
        masks = {REF: path_open_clean_fixed(img01)[crop],
                 "v3_production": (wire_probability(clf, img01) > PROD_THR)[crop]}
        for d in pred_dirs:
            f = d / f"{case_name(photo)}.png"
            if f.exists():
                m = read_png(f) > 127
                if m.shape != img01.shape:
                    raise SystemExit(f"{f}: shape {m.shape}, photo {img01.shape}")
                masks[d.name] = m[crop]

        for name, m in masks.items():
            s = score_mask(m, frame, p, ref, measures_width=False)
            rows.append({"quarter": tag, "photo": photo, "run": run_of(photo), "contender": name,
                         **{k: s.get(k) for k in ("pixel_f1", "wire_f1", "wire_recall",
                                                  "wire_precision", "cov_pred_over_gt",
                                                  "count_err_pct", "length_median_nm_err_pct",
                                                  "junction_recall")},
                         "n_wires_gt": ref["count"], "n_wires_pred": s["n_wires_pred"]})
            counts.setdefault(name, {})[tag] = tile_counts(m, frame, p, ref)
            save_overlay(m, frame, OUT / "overlays" / f"{tag}_{name}.png")
            print(f"  {name:22s} pixel F1 {s['pixel_f1']:.3f}  wire F1 {s['wire_f1']:.3f}  "
                  f"count {s['count_err_pct']:+.0f}%", flush=True)

    if not rows:
        raise SystemExit(f"no usable labels_*.csv in {LABEL_DIR}")
    per_q = pd.DataFrame(rows)
    per_q.to_csv(OUT / "scores_per_quarter.csv", index=False, float_format="%.4f")

    quarters = sorted(frames)
    complete = [n for n, per in counts.items() if set(per) == set(quarters)]
    for n in set(counts) - set(complete):
        print(f"note: {n} lacks predictions for some quarters, left out of the summary")
    rng = np.random.default_rng(SEED)
    ntiles = {q: len(counts[REF][q]["pix_tp"]) for q in quarters}
    draws = [{q: rng.integers(0, k, k) for q, k in ntiles.items()} for _ in range(N_BOOT)]
    boot = {n: np.array([pooled(counts[n], d) for d in draws]) for n in complete}

    summary = []
    for n in complete:
        pix, wf = pooled(counts[n])
        b, sub = boot[n], per_q[per_q.contender == n]
        row = {"contender": n, "quarters": len(quarters),
               "pixel_f1": pix, "pixel_f1_lo": np.percentile(b[:, 0], 2.5),
               "pixel_f1_hi": np.percentile(b[:, 0], 97.5),
               "wire_f1": wf, "wire_f1_lo": np.percentile(b[:, 1], 2.5),
               "wire_f1_hi": np.percentile(b[:, 1], 97.5),
               "count_abs_err_pct": sub.count_err_pct.abs().mean(),
               "length_abs_err_pct": sub.length_median_nm_err_pct.abs().mean(),
               "junction_recall": sub.junction_recall.mean(),
               "cov_pred_over_gt": sub.cov_pred_over_gt.mean()}
        if n != REF:
            for j, metric in enumerate(("pixel", "wire")):
                d = b[:, j] - boot[REF][:, j]
                row[f"d_{metric}_f1_vs_A"] = float(np.mean(d))
                row[f"d_{metric}_f1_vs_A_lo"] = float(np.percentile(d, 2.5))
                row[f"d_{metric}_f1_vs_A_hi"] = float(np.percentile(d, 97.5))
                row[f"p_better_{metric}_vs_A"] = float(np.mean(d > 0))
        for run in sorted(per_q.run.unique()):
            qs = [q for q in quarters if run_of(frames[q].name.rsplit("_", 1)[0]) == run]
            if qs:
                row[f"wire_f1_{run.replace(' ', '')}"] = pooled({q: counts[n][q] for q in qs})[1]
        summary.append(row)
    df = pd.DataFrame(summary).sort_values("wire_f1", ascending=False)
    df.to_csv(OUT / "summary.csv", index=False, float_format="%.4f")

    print(f"\n{len(quarters)} test quarters, pooled; 95% intervals from {N_BOOT} tile resamples")
    print(f"{'contender':22s} {'pixel F1':>8s} {'wire F1 [95%]':>22s} {'vs A [95%]':>24s} "
          f"{'P(>A)':>6s} {'count err':>9s} {'length err':>10s}")
    for _, r in df.iterrows():
        vs = ("-" if r.contender == REF else
              f"{r.d_wire_f1_vs_A:+.3f} [{r.d_wire_f1_vs_A_lo:+.3f}, {r.d_wire_f1_vs_A_hi:+.3f}]")
        pb = "-" if r.contender == REF else f"{r.p_better_wire_vs_A:.2f}"
        print(f"{r.contender:22s} {r.pixel_f1:8.3f} "
              f"{f'{r.wire_f1:.3f} [{r.wire_f1_lo:.3f}, {r.wire_f1_hi:.3f}]':>22s} {vs:>24s} "
              f"{pb:>6s} {r.count_abs_err_pct:8.0f}% {r.length_abs_err_pct:9.0f}%")
    print(f"\nwrote {OUT / 'summary.csv'} and {OUT / 'scores_per_quarter.csv'}")


if __name__ == "__main__":
    main()
