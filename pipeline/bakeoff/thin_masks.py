"""Thinner masks for contender E: choose how E's probability map becomes a mask
on the hand-labelled photos held out of E's hold-out runs, then apply that one
choice to E's test predictions.

    bash bakeoff/holdout_E.sh                          # ~2 h: two hold-out models
    venv/Scripts/python bakeoff/thin_masks.py choose   # pick on the held-out photos
    venv/Scripts/python bakeoff/thin_masks.py apply    # writes results/bakeoff/pred/E_thin/
    venv/Scripts/python bakeoff/score_bakeoff.py

Why: on the dense test photo E's masks are wider than A's, so close parallel
wires fuse into one and crossings blur (wire F1 and junction recall drop).

Candidates, fixed before any of them was scored:
  thr_t        prob > t, t in 0.5 0.6 0.7 0.8 0.9   (thr_0.5 = E as scored)
  ridge_t_r    the probability's ridge line: pixels with prob > t that are the
               maximum ACROSS the wire (non-maximum suppression along the
               Hessian's normal), widened by a disk of radius r. Splits wires
               that the plain threshold fuses. t in 0.3 0.5, r in 1 2 3
Chosen by whole-wire F1 pooled over the two held-out photos (ties: pixel F1),
with compare/'s scorer as in holdout_check.py. The test photos choose nothing;
E_thin is the only extra variant scored on them.

Caveat: two 80,000x photos choose the setting; the test photos are 50,000x.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy import ndimage as ndi

from common import OUT, PRED, read_png, save_u8

HOLD = OUT / "holdout"
CASES = {"TiS3_80000x_1": "HAND_TiS3_80000x_1", "TiS3_80000x_2": "HAND_TiS3_80000x_2"}
CHOICE = HOLD / "thin_choice.json"
SIGMA = 1.0         # px: smoothing of the probability before the across-wire maximum
SIGMA_DIR = 2.0     # px: smoothing for the across-wire direction
CANDIDATES = ([("thr", t, 0) for t in (0.5, 0.6, 0.7, 0.8, 0.9)]
              + [("ridge", t, r) for t in (0.3, 0.5) for r in (1, 2, 3)])


def name_of(kind: str, t: float, r: int) -> str:
    return f"thr_{t:.1f}" if kind == "thr" else f"ridge_{t:.1f}_{r}"


def disk(r: int) -> np.ndarray:
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r + r


def ridge(prob: np.ndarray, t: float, r: int) -> np.ndarray:
    """prob > t AND a local maximum across the wire, widened by a disk of radius r."""
    p = ndi.gaussian_filter(prob, SIGMA)
    # direction from a smoother Hessian: on a saturated (flat-topped) wire the
    # SIGMA=1 curvature is ~0 and its direction is noise
    a = ndi.gaussian_filter(prob, SIGMA_DIR, order=(2, 0))   # d2/drow2
    c = ndi.gaussian_filter(prob, SIGMA_DIR, order=(0, 2))   # d2/dcol2
    b = ndi.gaussian_filter(prob, SIGMA_DIR, order=(1, 1))
    root = np.sqrt(((a - c) / 2) ** 2 + b ** 2)
    l1, l2 = (a + c) / 2 - root, (a + c) / 2 + root          # l1 = most negative curvature
    v1 = np.stack([b, l1 - a])                               # its eigenvector (row, col)
    v2 = np.stack([l1 - c, b])                               # same, other form (b ~ 0)
    v = np.where(np.linalg.norm(v1, axis=0) >= np.linalg.norm(v2, axis=0), v1, v2)
    v = v / np.maximum(np.linalg.norm(v, axis=0), 1e-9)      # unit normal across the ridge
    rr, cc = np.mgrid[0:prob.shape[0], 0:prob.shape[1]].astype(np.float32)
    plus = ndi.map_coordinates(p, [rr + v[0], cc + v[1]], order=1, mode="nearest")
    minus = ndi.map_coordinates(p, [rr - v[0], cc - v[1]], order=1, mode="nearest")
    # a ridge: bent down across the wire, more than along it (not a wire's flank)
    line = (p >= plus) & (p >= minus) & (prob > t) & (l1 < 0) & (-l1 > np.abs(l2))
    return ndi.binary_dilation(line, disk(r)) if r else line


def make_mask(prob: np.ndarray, kind: str, t: float, r: int) -> np.ndarray:
    return prob > t if kind == "thr" else ridge(prob, t, r)


def choose() -> None:
    import pandas as pd
    from bootstrap import pooled, tile_counts
    from pseudo_labels import path_open_clean_fixed
    from score import extract_params, gt_reference, load_gt, score_mask

    frames = {f.name: f for f in load_gt()}
    p = extract_params()
    rows, counts = [], {}
    for k, (name, case) in enumerate(CASES.items(), start=1):
        frame = frames[name]
        ref = gt_reference(frame, p)
        f = HOLD / "pred" / f"E_holdout{k}" / f"{case}_prob.png"
        if not f.exists():
            raise SystemExit(f"{f} missing: run bash bakeoff/holdout_E.sh first")
        prob = read_png(f).astype(np.float32) / 255
        if prob.shape != frame.gt.shape:
            raise SystemExit(f"{f}: shape {prob.shape}, photo {frame.gt.shape}")
        masks = {"A_path_open_clean": path_open_clean_fixed(frame.img01)}
        masks.update({name_of(*cand): make_mask(prob, *cand) for cand in CANDIDATES})
        for m_name, m in masks.items():
            s = score_mask(m, frame, p, ref)
            rows.append({"photo": name, "variant": m_name,
                         **{c: s.get(c) for c in ("pixel_f1", "wire_f1", "wire_precision",
                                                  "wire_recall", "count_err_pct",
                                                  "junction_recall", "cov_pred_over_gt")}})
            counts.setdefault(m_name, {})[name] = tile_counts(m, frame, p, ref)
        print(f"{name}: scored {len(masks)} masks", flush=True)

    df = pd.DataFrame(rows)
    summ = []
    for m_name, per in counts.items():
        pix, wf = pooled(per)
        sub = df[df.variant == m_name]
        summ.append({"variant": m_name, "pixel_f1": pix, "wire_f1": wf,
                     **{f"wire_f1_{n}": sub[sub.photo == n].wire_f1.iloc[0] for n in CASES},
                     "count_abs_err_pct": sub.count_err_pct.abs().mean(),
                     "junction_recall": sub.junction_recall.mean(),
                     "marked_over_labelled": sub.cov_pred_over_gt.mean()})
    s = pd.DataFrame(summ).sort_values(["wire_f1", "pixel_f1"], ascending=False)
    s.to_csv(HOLD / "thin_scores.csv", index=False, float_format="%.4f")
    print(s.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    cands = s[s.variant != "A_path_open_clean"]
    best = cands.iloc[0]
    kind, t, r = next(c for c in CANDIDATES if name_of(*c) == best.variant)
    base = cands[cands.variant == "thr_0.5"].iloc[0]
    CHOICE.write_text(json.dumps({
        "variant": best.variant, "kind": kind, "t": t, "r": r,
        "held_out_wire_f1": float(best.wire_f1), "held_out_wire_f1_thr_0.5": float(base.wire_f1),
        "chosen_on": list(CASES), "candidates": [name_of(*c) for c in CANDIDATES]}, indent=2),
        encoding="utf-8")
    print(f"\nchosen: {best.variant} (held-out wire F1 {best.wire_f1:.3f} vs {base.wire_f1:.3f} "
          f"for E as scored). Wrote {CHOICE}")


def apply() -> None:
    ch = json.loads(CHOICE.read_text(encoding="utf-8"))
    src, dst = PRED / "E", PRED / "E_thin"
    dst.mkdir(parents=True, exist_ok=True)
    probs = sorted(src.glob("*_prob.png"))
    if not probs:
        raise SystemExit(f"no probability maps in {src}: run predict.py for E first")
    for f in probs:
        prob = read_png(f).astype(np.float32) / 255
        m = make_mask(prob, ch["kind"], ch["t"], ch["r"])
        case = f.name[:-len("_prob.png")]
        save_u8(dst / f"{case}.png", m * 255)
        print(f"  {case}: wire {m.mean() * 100:.1f}% of pixels (E: {np.mean(prob > 0.5) * 100:.1f}%)")
    (dst / "variant.json").write_text(json.dumps(ch, indent=2), encoding="utf-8")
    print(f"wrote {dst} with {ch['variant']}, chosen on the held-out training photos")


if __name__ == "__main__":
    {"choose": choose, "apply": apply}[sys.argv[1] if len(sys.argv) > 1 else "choose"]()
