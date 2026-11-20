"""Build the bake-off training set: agreement pseudo-labels on every in-band
TiS3 photo, plus the two hand-labelled photos.

    venv/Scripts/python bakeoff/pseudo_labels.py              # every photo (~1 min each)
    venv/Scripts/python bakeoff/pseudo_labels.py --previews   # + overlays to check by eye

For every photo inside the validated scale band (50k-80k), except the test
photos (bakeoff/test_photos.txt) and the raw copies of the hand-labelled photos:
  A  path_open_clean at its fixed production setting (compare/methods.py)
  P  wire probability from Nikhil's production forest (results/wire_classifier/rf_v3.joblib)
  wire        A and P > --wire-p                           both say wire
  background  P <= --bg-p and more than --ring px from A   both say background
  ignore      everything else: the two disagree, or a wire edge
Both cut-offs default to the forest's own production threshold (0.70). The
forest was trained on balanced samples, so its probabilities run high: the
median background pixel sits near 0.3, and a stricter background cut-off
(0.3) ignored ~40% of every photo, mostly the mottled background the
networks most need to learn as background.
Training on agreement only keeps each filter's systematic mistakes
(path_open_clean fusing close wires, the forest's short fragments and round
particles) out of the labels; the networks never see a target there.

The two hand-labelled 80k photos get their filled hand outlines (0/1, no
ignore), read with compare/score.load_gt: the same answer key compare/ uses.
Their raw copies in the 30-min folder (SF251211M_80000x_62 and _66) are found
by pixel hash and skipped.

Writes results/bakeoff/train/images/<case>.png (8-bit, 50k-equivalent),
labels/<case>.png (0 background, 1 wire, 2 ignore) and manifest.json.
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from common import (BG, IGNORE, TRAIN, WIRE, case_name, img_to_u8, in_band_photos,
                    load_photo, native_hash, read_png, run_of, save_u8, test_photos,
                    write_json)

WIRE_P = 0.70    # forest probability above which it says wire (its production threshold)
BG_P = 0.70      # ... and at or below which it says background
RING = 1         # px around path_open_clean's band that never counts as background


def path_open_clean_fixed(img01: np.ndarray) -> np.ndarray:
    """methods.PATH_OPEN_CLEAN at its production setting, computing only the map
    that setting reads (sato 1-2, 25 px opening; no particle removal or thinning)."""
    from methods import PATH_OPEN_CLEAN, line_open
    from tune_pseudo_mask import ridge_response
    kw = PATH_OPEN_CLEAN.author_params
    assert not kw["particles"] and not kw["erode"], "update path_open_clean_fixed"
    sig = {"1-3": (1, 2, 3), "1-2": (1, 2)}[kw["sig"]]
    resp = {(kw["sig"], kw["length"]): line_open(ridge_response(img01, "sato", sig), kw["length"])}
    return PATH_OPEN_CLEAN.mask(img01, resp, PATH_OPEN_CLEAN.author_thr, kw)


def agreement_labels(a: np.ndarray, prob: np.ndarray, wire_p: float, bg_p: float,
                     ring: int) -> np.ndarray:
    lab = np.full(a.shape, IGNORE, np.uint8)
    near = cv2.dilate(a.astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring + 1,) * 2)) > 0
    lab[~near & (prob <= bg_p)] = BG
    lab[a & (prob > wire_p)] = WIRE
    return lab


def faint_candidates(img01: np.ndarray, bg: np.ndarray, q: float = 99.5, q_strong: float = 99.9,
                     length: int = 7, min_len: float = 6.0, long_len: float = 12.0,
                     min_elong: float = 2.5, grow: int = 2) -> np.ndarray:
    """Line-shaped things A is not sensitive enough to see (contender E).

    A needs a 25 px straight dark run, so short rods (the test labels have
    5-8 px ones), faint wires and the bodies of bright rods all end up as
    agreed background. Here: dark (sato 1-2) and bright (sato 1-3) ridge maps,
    each opened with only a `length` px line in the growth directions, cut at
    this photo's own noise level (the q-th percentile over its background
    pixels). A component counts if it is at least min_elong times longer than
    wide and either long_len px long, or min_len px long with one pixel above
    the q_strong percentile (which drops the short noise flecks). Returned
    grown by `grow` px; the caller turns the background under it into ignore.
    """
    import scipy.ndimage as ndi
    from methods import line_open
    from skimage.filters import sato

    def opened(black: bool, sig) -> np.ndarray:
        r = sato(img01, sigmas=sig, black_ridges=black)
        return line_open((r - r.min()) / (r.max() - r.min() + 1e-12), length)

    od, ob = opened(True, (1, 2)), opened(False, (1, 2, 3))
    td = np.percentile(od[bg], [q, q_strong])
    tb = np.percentile(ob[bg], [q, q_strong])
    weak = (od > td[0]) | (ob > tb[0])
    strong = (od > td[1]) | (ob > tb[1])
    lab, n = ndi.label(weak, structure=np.ones((3, 3), int))
    keep = np.zeros(n + 1, bool)
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        inside = lab[sl] == i
        r, c = np.nonzero(inside)
        if len(r) < 4:
            continue
        pts = np.stack([r, c], 1).astype(float)
        pts -= pts.mean(0)
        _, vec = np.linalg.eigh(np.cov(pts.T) + 1e-6 * np.eye(2))
        proj = pts @ vec[:, 1]
        ln = proj.max() - proj.min() + 1
        if ln / max(len(r) / ln, 1.0) < min_elong:
            continue
        keep[i] = ln >= long_len or (ln >= min_len and strong[sl][inside].any())
    out = keep[lab]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2)
    return cv2.dilate(out.astype(np.uint8), k) > 0


def preview(img_u8: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """Gray photo; green = wire label, magenta tint = ignored."""
    rgb = np.stack([img_u8] * 3, axis=2).astype(np.float32)
    ign = lab == IGNORE
    rgb[ign] = 0.55 * rgb[ign] + 0.45 * np.array([255, 0, 255])
    rgb[lab == WIRE] = (40, 210, 40)
    return rgb.astype(np.uint8)


def stats(lab: np.ndarray) -> dict:
    return {"wire": float((lab == WIRE).mean()), "ignore": float((lab == IGNORE).mean())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wire-p", type=float, default=WIRE_P)
    ap.add_argument("--bg-p", type=float, default=BG_P)
    ap.add_argument("--ring", type=int, default=RING)
    ap.add_argument("--force", action="store_true", help="redo photos that already have labels")
    ap.add_argument("--previews", action="store_true", help="also write overlay PNGs")
    ap.add_argument("--only", default="", help="only photos whose name contains this")
    args = ap.parse_args()

    from PIL import Image
    from score import load_gt
    from wire_classifier import load_classifier, wire_probability

    tests = set(test_photos())
    cases = []
    if args.previews:
        (TRAIN / "previews").mkdir(parents=True, exist_ok=True)

    # the hand-labelled photos, and the hashes of their raw files
    from gt_labels import gt_frames
    hand_hashes = {native_hash(fr["raw"]) for fr in gt_frames()}
    for fr in load_gt():
        case = f"HAND_{fr.name}"
        img = img_to_u8(fr.img01)
        lab = fr.gt.astype(np.uint8)            # 1 = wire, 0 = background
        save_u8(TRAIN / "images" / f"{case}.png", img)
        save_u8(TRAIN / "labels" / f"{case}.png", lab)
        if args.previews:
            Image.fromarray(preview(img, lab)).save(TRAIN / "previews" / f"{case}.png")
        cases.append({"case": case, "photo": fr.name, "run": "30 min", "source": "hand",
                      "shape": list(lab.shape), **stats(lab)})
        print(f"  {case:30s} hand labels   wire {cases[-1]['wire'] * 100:5.1f}%", flush=True)

    clf = None
    photos = [p for p in in_band_photos() if args.only in p.stem]
    skipped_test, skipped_hand = [], []
    for i, path in enumerate(photos, 1):
        if path.stem in tests:
            skipped_test.append(path.stem)
            continue
        if native_hash(path) in hand_hashes:
            skipped_hand.append(path.stem)
            continue
        case = case_name(path.stem)
        img_path, lab_path = TRAIN / "images" / f"{case}.png", TRAIN / "labels" / f"{case}.png"
        t0 = time.perf_counter()
        if lab_path.exists() and not args.force:
            lab = read_png(lab_path)
            img = read_png(img_path)
            note = "kept"
        else:
            img01, meta = load_photo(path)
            if clf is None:
                clf = load_classifier()
            a = path_open_clean_fixed(img01)
            prob = wire_probability(clf, img01)
            lab = agreement_labels(a, prob, args.wire_p, args.bg_p, args.ring)
            img = img_to_u8(img01)
            save_u8(img_path, img)
            save_u8(lab_path, lab)
            note = f"{time.perf_counter() - t0:4.0f} s"
        if args.previews:
            Image.fromarray(preview(img, lab)).save(TRAIN / "previews" / f"{case}.png")
        cases.append({"case": case, "photo": path.stem, "run": run_of(path.stem),
                      "source": "pseudo", "shape": list(lab.shape), **stats(lab)})
        print(f"  [{i:2d}/{len(photos)}] {case:24s} wire {cases[-1]['wire'] * 100:5.1f}%  "
              f"ignore {cases[-1]['ignore'] * 100:5.1f}%  ({note})", flush=True)

    if args.only:
        print("filtered run: manifest.json not rewritten")
        return
    pseudo = [c for c in cases if c["source"] == "pseudo"]
    write_json(TRAIN / "manifest.json", {
        "settings": {"wire_p": args.wire_p, "bg_p": args.bg_p, "ring_px": args.ring,
                     "pseudo_from": "path_open_clean (fixed production setting) AND rf_v3",
                     "scale": "50k-equivalent, 5.396 nm/px",
                     "labels": {"background": BG, "wire": WIRE, "ignore": IGNORE}},
        "test_photos_excluded": sorted(skipped_test),
        "hand_label_duplicates_skipped": sorted(skipped_hand),
        "cases": cases})
    w = np.mean([c["wire"] for c in pseudo]) if pseudo else 0
    ig = np.mean([c["ignore"] for c in pseudo]) if pseudo else 0
    print(f"\n{len(pseudo)} pseudo-labelled photos (mean wire {w * 100:.1f}%, ignore {ig * 100:.1f}%), "
          f"{len(cases) - len(pseudo)} hand-labelled; kept out: {len(skipped_test)} test photos, "
          f"{len(skipped_hand)} raw copies of the hand-labelled photos")
    print(f"wrote {TRAIN}")


if __name__ == "__main__":
    main()
