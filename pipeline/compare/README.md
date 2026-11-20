# compare/: one scorer for every wire-segmentation method

Puts every method (ridge filters, the v3 random forest, later the U-Net) through
the same ground truth, the same cross-frame protocol and the same three scores,
so the numbers can be compared directly.

```bash
venv/Scripts/python compare/score.py                    # sanity check only (~10 s)
venv/Scripts/python compare/run_all.py                  # every method (~20 min, RF trains once)
venv/Scripts/python compare/run_all.py sato_dark_v1 rf_v3   # selected methods
venv/Scripts/python compare/bootstrap.py                # 95% intervals for the last run (~15 min)
venv/Scripts/python compare/width_check.py              # true wire width on 200k photos (~2 min)
venv/Scripts/python compare/external.py                 # every method on a newly labelled photo
venv/Scripts/python compare/run_production.py           # the winner through the full pipeline vs v3 (~5 min)
```

Run from the repo root. The run stops if the sanity check fails: v1 sato
(dark, sigmas 1-3, thr 0.10) must reproduce the README's
recall 0.517 / precision 0.484 / F1 0.500.

Latest results and what they mean: [RESULTS.md](RESULTS.md).

| File | What it does |
|---|---|
| `score.py` | `load_gt()`, `score_pixels()`, `score_mask()`, `evaluate()` (cross-frame), `sanity_check()` |
| `methods.py` | The methods. To add a filter, write `img01 -> map` and append a `Method` to `METHODS` |
| `run_all.py` | Runs `METHODS`, writes `results/compare/results.csv`, `results.json` and `overlays/` |
| `bootstrap.py` | Tile bootstrap: resamples 160 px tiles of the test frames (same draws for every method) to get 95% intervals and the probability that one method beats another. Writes `results/compare/bootstrap.csv` |
| `run_production.py` | Runs `path_open_clean` at one fixed setting on every photo, the same way production does (masks in `results/path_open_clean/`, then Nikhil's unchanged `wire_extract` into `results/wire_extract_path_open_clean/`), and compares with v3 photo by photo |
| `width_check.py` | Measures hand-picked wires on 200k photos (edge-to-edge and outer width), then compares with Steger and v3 widths at 50k. Profile pictures go in `results/compare/width_check/` |
| `external.py` | Scores every method on a photo labelled with `Materials_ThangPham/labelling/label_wires.py`, using only settings learned on the two 80k photos |
| `steger.py` | Steger line detector: sub-pixel centre lines, plus a width per centre point from the nearest edge on each side |
| `line_fit.py` | Sato + line fitting along the three growth directions (the review's E1b), ported unchanged. `check_port()` proves it matches the original |
| `wire_metrics.py` | Whole-wire centreline matching, copied from the review's `starter_code/metrics.py` |

## Protocols

Settings are always chosen on one labelled frame and scored on the other,
then the frames swap. Only test-frame scores are reported, averaged over both
directions. Learned methods (`rf_v3`) are also trained on one frame only.

- `author`: the author's fixed setting. This is how the main README's numbers were made.
- `tuned`: best pixel F1 on the training frame. This can reward over-marking, so check `cov_pred_over_gt`.
- `matched`: training-frame coverage equal to its GT coverage. The two frames'
  GT coverage differs (12.8% vs 9.7%), so this choice transfers unevenly.
- `wire`: best whole-wire F1 on the training frame. Use it when whole wires are
  what matters. It uses a coarser threshold grid, because every candidate runs the wire step.

`sato_line_fit`, `path_open`, `path_open_clean`, `steger` and `rf_pathopen` tune several settings at once. The chosen
settings are listed in `params_1to2` / `params_2to1`. Line fitting draws 5-px lines,
so its width error is left blank.

## Columns in results.csv

| Column | Meaning |
|---|---|
| `pixel_recall/precision/f1` | `tune_pseudo_mask.score()`, unchanged |
| `cov_pred_over_gt` | marked pixels ÷ GT pixels (above 1 means over-marking) |
| `wire_recall/precision/f1` | Whole wires matched (centreline within 4 px, mutual coverage ≥ 50%) against the wires `wire_extract` builds from the GT mask. A perfect mask scores 1.0 |
| `wire_f1_hand` | The same match against the raw hand-drawn instances. Ceiling is about 0.56 (`gt_mask` row), because outlines are cut into fragments where one wire was drawn over another |
| `*_abs_err_pct` | Wire count, median length, median width and junction count from `wire_extract` on the predicted mask, compared with the same step on the GT mask (mean of \|error\| over both directions) |
| `junction_recall` | Share of GT junctions with a detected junction within 5 px (the `--close-ab` proxy) |
| `runtime_s` | Wall time per frame for map + threshold. Noisy on a busy laptop, so treat it as rough |

Caveats: there are only two labelled frames, so trust a win only if it holds
in both directions (`results.json` has each direction) and `bootstrap.py`
gives it a high probability of being ahead. Width errors are measured against
the outline width, which `width_check.py` shows is the wire's outer width
(~45 nm, including the dark edges). The edge-to-edge width is ~29 nm.
