"""One scorer for every wire-segmentation method on the two hand-labelled TiS3 frames.

Every method (ridge filter, random forest, later the U-Net) goes through the
same steps, so their numbers can be put side by side:

  1. Ground truth: gt_labels.parse_labeled (filled outlines), rescaled to the
     50k-equivalent grid exactly as tune_pseudo_mask.py and wire_classifier.py do.
  2. Cross-frame: choose the threshold (and, for learned methods, train) on one
     labelled frame, score on the other, then swap. Only test-frame scores are
     reported; the two directions are averaged.
  3. Three kinds of score on each test frame:
       pixel        recall / precision / F1 -- tune_pseudo_mask.score(), unchanged
       wire         whole-wire matching of centrelines (wire_metrics.centreline_scores)
                    between the chains wire_extract builds from the predicted mask
                    and the chains it builds from the GT mask (same reference as
                    --validate-gt; a perfect mask scores 1.0).
                    wire_*_hand matches against the hand-drawn instances instead.
                    Its ceiling is only ~0.56 (the gt_mask row of run_all.py):
                    where one wire was drawn over another, the lower outline is
                    cut into short pieces that count as separate hand wires.
       measurement  wire_extract.extract_from_mask on the predicted mask and on the
                    GT mask with the same parameters: count, median length, median
                    width, junction count, junction recall (the --close-ab proxy)

Four protocols per method (settings are always chosen on the training frame):
  author   the author's fixed setting (sato thr 0.10, RF p > 0.70, ...). Learned
           methods are still trained on the other frame. This is how the README
           numbers were made, so it doubles as the reproduction check.
  tuned    best pixel F1 on the training frame, from a grid of coverage
           quantiles (the same tuning effort for every method). Pixel F1 can
           reward over-marking: check cov_pred_over_gt.
  matched  predicted coverage closest to the training frame's GT coverage, the
           README's "matched coverage". The two frames' GT coverage differs
           (12.8% vs 9.7%), so the match transfers unevenly to the test frame.
  wire     best wire F1 on the training frame (a coarser grid, since every
           candidate runs the wire step). The protocol to use when whole wires
           are what matters, and the one the review's line fitting was built for.

Methods with several settings (line fitting) search threshold x params; the
author protocol uses author_thr + author_params.

`python compare/score.py` runs the sanity check: v1 sato (dark, sigmas 1-3,
thr 0.10) must reproduce the README's recall 0.517 / precision 0.484 / F1 0.500.
"""
from __future__ import annotations

import itertools
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ridge"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import wire_extract                                      # noqa: E402
from gt_labels import FAMILIES, gt_frames, load_raw, parse_labeled   # noqa: E402
from ridge_nanowire_images2 import read_pixel_size       # noqa: E402
from tune_pseudo_mask import score as _nikhil_score, to_50k          # noqa: E402
from wire_metrics import centreline_scores               # noqa: E402

OUT = ROOT / "results" / "compare"

COVERAGE_GRID = np.round(np.arange(0.04, 0.3001, 0.005), 3)  # threshold search grid
WIRE_TOL_PX = 4         # centreline match distance at 50k-equivalent (~22 nm)
WIRE_COVER = 0.5        # mutual centreline coverage for a whole-wire match
README_V1 = {"recall": 0.517, "precision": 0.484, "f1": 0.500}

Response = Callable[[np.ndarray], np.ndarray]


# --------------------------------------------------------------------------
# ground truth
# --------------------------------------------------------------------------

@dataclass
class Frame:
    name: str
    img01: np.ndarray       # frame at 50k-equivalent scale, [0, 1]
    gt: np.ndarray          # filled GT wire mask (union over families)
    instances: np.ndarray   # hand-drawn wire instances, ids unique across families
    nm_per_px: float


def load_gt() -> list[Frame]:
    """Both labelled frames and their filled GT at the 50k-equivalent scale."""
    frames = []
    for fr in gt_frames():
        s = fr["scale_to_50k"]
        img01 = to_50k(load_raw(fr["raw"]), s) / 255.0
        parsed = parse_labeled(fr["labeled"])
        gt = to_50k(parsed["union"].astype(np.uint8), s, nearest=True).astype(bool)

        # per-family instance labels -> one image; where families cross, the
        # later family keeps the pixel and the earlier wire keeps its id elsewhere
        inst = np.zeros(gt.shape, np.int32)
        offset = 0
        for fam in FAMILIES:
            lab = to_50k(parsed["labels"][fam].astype(np.uint16), s,
                         nearest=True).astype(np.int32)
            inst[lab > 0] = lab[lab > 0] + offset
            offset += parsed["counts"][fam]

        px_m, _ = read_pixel_size(fr["raw"])
        frames.append(Frame(fr["name"], img01, gt, inst, px_m * 1e9 / s))
    if len(frames) != 2:
        raise SystemExit("need both GT frames in data/real/OneDrive_1_7-17-2026")
    return frames


def score_pixels(mask: np.ndarray, gt: np.ndarray) -> dict:
    """Pixel recall / precision / F1, computed by tune_pseudo_mask.score itself."""
    return _nikhil_score(mask, gt)


# --------------------------------------------------------------------------
# methods
# --------------------------------------------------------------------------

@dataclass
class Method:
    """A segmentation method: a wire-likeness map, a threshold, optional cleanup.

    Give `response` for a fixed filter, or `train` for a learned method
    (train(frame) returns a response function fitted on that frame only).
    Methods with more than a threshold give `make_mask(img01, resp, thr, **params)`
    and a `params` grid ({name: [values]}) searched together with the threshold.
    """
    name: str
    response: Response | None = None
    train: Callable[[Frame], Response] | None = None
    post: Callable[[np.ndarray], np.ndarray] | None = None
    make_mask: Callable[..., np.ndarray] | None = None
    params: dict[str, Sequence] | None = None
    author_thr: float | None = None
    author_params: dict | None = None
    thresholds: Sequence[float] | None = None    # None -> coverage-quantile grid
    grid_map: Callable[[object], np.ndarray] | None = None   # map the quantile grid is read from
    measures_width: bool = True                  # False: width is drawn, not measured
    note: str = ""

    def fit(self, frame: Frame) -> Response:
        return self.train(frame) if self.train else self.response

    def mask(self, img01: np.ndarray, resp: np.ndarray, thr: float,
             params: dict | None = None) -> np.ndarray:
        if self.make_mask:
            return self.make_mask(img01, resp, thr, **(params or {}))
        m = resp > thr
        return self.post(m) if self.post else m

    def candidates(self, resp: np.ndarray, coarse: bool = False) -> list[tuple[float, dict]]:
        """(threshold, params) pairs to search. coarse: every 4th quantile step."""
        if self.thresholds is not None:
            thrs = np.asarray(self.thresholds, float)
        else:
            base = self.grid_map(resp) if self.grid_map else resp
            thrs = np.unique(np.quantile(base, 1.0 - COVERAGE_GRID))
            if coarse:
                thrs = thrs[::4]
        names = list(self.params or {})
        combos = [dict(zip(names, c)) for c in
                  itertools.product(*[self.params[n] for n in names])] or [{}]
        return [(float(t), kw) for t in thrs for kw in combos]


# --------------------------------------------------------------------------
# wire and measurement scores
# --------------------------------------------------------------------------

def extract_params():
    """wire_extract's default parameters (the production settings)."""
    return wire_extract.parse_args([])


def _measurements(summary: dict, render: dict) -> dict:
    wa = summary["wires_all"]
    return {"count": wa["count"],
            "length_median_nm": wa["length_nm"].get("median"),
            "width_median_nm": wa["width_mean_nm"].get("median"),
            "junctions": summary["junctions"]["count"],
            "_junction_xy": np.asarray(render["junction_centroids"], float).reshape(-1, 2)}


def gt_reference(frame: Frame, p) -> dict:
    """Extraction on the GT mask: the wire and measurement reference for this frame."""
    records, summary, render = wire_extract.extract_from_mask(frame.gt, frame.nm_per_px, p)
    ref = _measurements(summary, render)
    ref["_instances"] = chains_to_instances(records, frame.gt.shape)
    return ref


def chains_to_instances(records: list, shape: tuple) -> np.ndarray:
    """Label image with one id per extracted wire chain (its skeleton path)."""
    inst = np.zeros(shape, np.int32)
    for i, rec in enumerate(records, start=1):
        P = np.asarray(rec["_pixels"], int).reshape(-1, 2)
        inst[P[:, 0], P[:, 1]] = i
    return inst


def _pct(pred, ref):
    if pred is None or not ref:
        return None
    return 100.0 * (pred - ref) / ref


def _wire(gt_inst: np.ndarray, pred_inst: np.ndarray) -> dict:
    return centreline_scores(gt_inst, pred_inst, tol=WIRE_TOL_PX, cover=WIRE_COVER)


def wire_f1(mask: np.ndarray, frame: Frame, p, ref: dict) -> float:
    """Whole-wire F1 of a mask against the GT-mask chains (used for wire tuning)."""
    records, _, _ = wire_extract.extract_from_mask(mask, frame.nm_per_px, p)
    return _wire(ref["_instances"], chains_to_instances(records, mask.shape))["wire_f1"]


def score_mask(mask: np.ndarray, frame: Frame, p, ref: dict,
               measures_width: bool = True) -> dict:
    """Pixel, wire and measurement scores of one predicted mask on one frame."""
    out = {f"pixel_{k}": v for k, v in score_pixels(mask, frame.gt).items()
           if k in ("recall", "precision", "f1")}
    out["cov_pred_over_gt"] = float(mask.mean() / max(frame.gt.mean(), 1e-9))

    records, summary, render = wire_extract.extract_from_mask(mask, frame.nm_per_px, p)
    pred_inst = chains_to_instances(records, mask.shape)
    for suffix, gt_inst in (("", ref["_instances"]), ("_hand", frame.instances)):
        w = _wire(gt_inst, pred_inst)
        out.update({f"wire_recall{suffix}": w["wire_recall"],
                    f"wire_precision{suffix}": w["wire_precision"],
                    f"wire_f1{suffix}": w["wire_f1"],
                    f"centreline_completeness{suffix}": w["completeness"],
                    f"centreline_correctness{suffix}": w["correctness"],
                    f"n_wires_gt{suffix}": w["n_gt"]})
    out["n_wires_pred"] = len(records)

    m = _measurements(summary, render)
    if not measures_width:
        m["width_median_nm"] = None
    for k in ("count", "length_median_nm", "width_median_nm", "junctions"):
        out[k] = m[k]
        out[f"{k}_gt"] = ref[k]
        out[f"{k}_err_pct"] = _pct(m[k], ref[k])

    ref_j, det = ref["_junction_xy"], m["_junction_xy"]
    if len(ref_j) and len(det):
        d = np.sqrt(((ref_j[:, None] - det[None]) ** 2).sum(-1)).min(1)
        out["junction_recall"] = float((d <= p.junction_match_px).mean())
    else:
        out["junction_recall"] = None
    return out


def save_overlay(mask: np.ndarray, frame: Frame, path: Path) -> None:
    """Gray frame; green = correct wire, orange = false alarm, blue = missed."""
    g = (frame.img01 * 255).clip(0, 255).astype(np.uint8)
    rgb = np.stack([g] * 3, axis=2)
    rgb[mask & frame.gt] = (40, 200, 40)
    rgb[mask & ~frame.gt] = (255, 140, 0)
    rgb[~mask & frame.gt] = (40, 110, 255)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


# --------------------------------------------------------------------------
# cross-frame evaluation
# --------------------------------------------------------------------------

def _directions(frames: list[Frame]):
    a, b = frames
    return [(a, b), (b, a)]


def _pick(method: Method, train: Frame, resp: np.ndarray, protocol: str,
          masks: dict, p=None, ref: dict | None = None) -> tuple[float, dict, float]:
    """(threshold, params, training-frame score) for tuned / matched / wire.

    masks caches training-frame masks per candidate across protocols.
    """
    best = None
    for t, kw in method.candidates(resp, coarse=(protocol == "wire")):
        key = (t, tuple(sorted(kw.items())))
        if key not in masks:
            masks[key] = method.mask(train.img01, resp, t, kw)
        m = masks[key]
        if protocol == "tuned":
            sc = score_pixels(m, train.gt)["f1"]
            obj = sc
        elif protocol == "matched":
            sc = score_pixels(m, train.gt)["f1"]
            obj = -abs(m.mean() - train.gt.mean())
        else:
            sc = obj = wire_f1(m, train, p, ref)
        if best is None or obj > best[0]:
            best = (obj, t, kw, sc)
    return best[1], best[2], best[3]


def evaluate(method: Method, frames: list[Frame], p=None, refs: dict | None = None,
             protocols=("author", "tuned", "matched", "wire"), full: bool = True,
             overlays: bool = True) -> dict:
    """Score one method in both cross-frame directions under each protocol.

    full=False keeps only the pixel scores (fast; used by the sanity check).
    Returns {protocol: {"directions": [...], "mean": {...}}}.
    """
    if full and (p is None or refs is None):
        raise ValueError("full scoring needs extract params and GT references")
    fitted, resp, masks = {}, {}, {}

    def response(train: Frame, frame: Frame):
        """(map, seconds) of the method fitted on `train`, applied to `frame`."""
        if train.name not in fitted:
            t0 = time.perf_counter()
            fitted[train.name] = (method.fit(train), time.perf_counter() - t0)
        if (train.name, frame.name) not in resp:
            t0 = time.perf_counter()
            r = fitted[train.name][0](frame.img01)
            resp[(train.name, frame.name)] = (r, time.perf_counter() - t0)
        return resp[(train.name, frame.name)]

    out = {}
    for protocol in protocols:
        if protocol == "author" and method.author_thr is None:
            continue
        if protocol == "wire" and not full:
            continue
        dirs = []
        for train, test in _directions(frames):
            if protocol == "author":
                thr, kw, train_score = method.author_thr, dict(method.author_params or {}), None
            else:
                r_tr, _ = response(train, train)
                thr, kw, train_score = _pick(method, train, r_tr, protocol,
                                             masks.setdefault(train.name, {}),
                                             p, refs[train.name] if refs else None)

            r_te, resp_s = response(train, test)
            t0 = time.perf_counter()
            mask = method.mask(test.img01, r_te, thr, kw)
            run_s = resp_s + time.perf_counter() - t0

            row = {"train": train.name, "test": test.name, "thr": float(thr),
                   "params": kw, "train_score": train_score, "runtime_s": run_s,
                   "fit_s": fitted[train.name][1]}
            if full:
                row.update(score_mask(mask, test, p, refs[test.name], method.measures_width))
                if overlays:
                    save_overlay(mask, test, OUT / "overlays" /
                                 f"{method.name}_{protocol}_{test.name}.png")
            else:
                row.update({f"pixel_{k}": v for k, v in score_pixels(mask, test.gt).items()
                            if k in ("recall", "precision", "f1")})
            dirs.append(row)
        out[protocol] = {"directions": dirs, "mean": _mean(dirs)}
    return out


def _mean(dirs: list[dict]) -> dict:
    """Average numeric scores over directions; errors become mean |error|."""
    out = {}
    for k, v0 in dirs[0].items():
        vals = [d[k] for d in dirs]
        if not all(isinstance(v, (int, float, np.integer, np.floating)) for v in vals):
            continue
        if k.endswith("_err_pct"):
            out[k.replace("_err_pct", "_abs_err_pct")] = float(np.mean(np.abs(vals)))
        else:
            out[k] = float(np.mean(vals))
    return out


# --------------------------------------------------------------------------
# sanity check
# --------------------------------------------------------------------------

def sanity_check(frames: list[Frame] | None = None, tol: float = 0.001) -> dict:
    """v1 sato under the author protocol must reproduce the README row."""
    from methods import SATO_V1
    frames = frames or load_gt()
    res = evaluate(SATO_V1, frames, protocols=("author",), full=False)["author"]
    m = res["mean"]
    print("sanity check: v1 sato (dark, sigmas 1-3, thr 0.10), mean of both GT frames")
    for d in res["directions"]:
        print(f"  {d['test']:15s} recall {d['pixel_recall']:.4f}  "
              f"precision {d['pixel_precision']:.4f}  F1 {d['pixel_f1']:.4f}")
    ok = True
    for k, target in README_V1.items():
        got = m[f"pixel_{k}"]
        good = abs(got - target) <= tol
        ok &= good
        print(f"  {k:9s} {got:.4f}  README {target:.3f}  {'OK' if good else 'MISMATCH'}")
    if not ok:
        raise SystemExit("scorer does NOT reproduce the README v1 row -- do not trust it")
    print("  -> scorer agrees with Nikhil's numbers")
    return m


if __name__ == "__main__":
    sanity_check()
