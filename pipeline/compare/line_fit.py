"""Sato + direction-constrained line fitting (the review's experiment E1b).

Ported unchanged from the review's starter_code (baseline_ridge.ridge_response,
baseline_ridge.ridge_hough_instances, measure.structure_tensor_orientations,
measure.dominant_directions, common.normalize) so compare/ runs without that
folder. check_port() in this file shows the port gives identical output to the original.

How it works: two-polarity Sato map, hysteresis threshold, skeleton, then a
probabilistic Hough transform restricted to the three growth directions
(estimated from the image's structure tensor). Collinear segments are merged,
and every merged segment is drawn as a straight line.

Scale: E1b ran at native 80k resolution. Here it runs on the 50k-equivalent
grid like every other method, so its pixel lengths are scaled by 0.625
(line_len 15/25 -> 10/16 px, gap 4/8 -> 3/5 px). Sato sigmas stay 1-3 px, the
same as the other Sato rows, so the race isolates the line-fitting step. Lines
are drawn 5 px wide (~27 nm, the dark-body width), so this method's width
"measurement" is the drawn width, not a measurement of the wires.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, morphology, segmentation
from skimage.draw import line as draw_line
from skimage.transform import probabilistic_hough_line

SIGMAS = (1, 2, 3)
WIDTH_PX = 5            # drawn line width at 50k-equivalent (~27 nm)

# the ported code keeps remove_small_objects(min_size=...) as written; same result
warnings.filterwarnings("ignore", message="Parameter `min_size` is deprecated",
                        category=FutureWarning)


def normalize(a, lo=1, hi=99):
    p1, p2 = np.percentile(a, [lo, hi])
    return np.clip((a - p1) / (p2 - p1 + 1e-9), 0, 1).astype(np.float32)


def dominant_directions(angles_deg, weights=None, spacing=60.0):
    """Fit three directions theta, theta+60, theta+120 (mod 180). Returns theta in [0,60) and
    R in [0,1] (1 = all orientations exactly on the three directions, ~0 = random)."""
    a = np.radians(np.asarray(angles_deg, float) * 360.0 / spacing)
    w = np.ones_like(a) if weights is None else np.asarray(weights, float)
    c, s = (w * np.cos(a)).sum(), (w * np.sin(a)).sum()
    return (np.degrees(np.arctan2(s, c)) * spacing / 360.0) % spacing, float(np.hypot(c, s) / w.sum())


def structure_tensor_orientations(img, sigma=1.5, rho=5.0):
    """Label-free local line orientation (same convention) and coherence in [0,1]."""
    g = ndi.gaussian_filter(img.astype(float), sigma)
    gy, gx = np.gradient(g)
    Jxx, Jyy, Jxy = (ndi.gaussian_filter(v, rho) for v in (gx * gx, gy * gy, gx * gy))
    grad_ang = 0.5 * np.degrees(np.arctan2(2 * Jxy, Jxx - Jyy))
    coh = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2) / (Jxx + Jyy + 1e-12)
    return (grad_ang + 90) % 180, coh


def ridge_response(img, sigmas, polarity="both"):
    out = []
    if polarity in ("both", "bright"):
        r = filters.sato(img, sigmas=sigmas, black_ridges=False)
        out.append(r / (np.percentile(r, 99.5) + 1e-9))
    if polarity in ("both", "dark"):
        r = filters.sato(img, sigmas=sigmas, black_ridges=True)
        out.append(r / (np.percentile(r, 99.5) + 1e-9))
    return np.max(out, 0)


def ridge_hough_instances(img, R, low, high, line_len=20, gap=6, width=5, tol_deg=4):
    """Improved classical baseline for thin wires: threshold the ridge map, then fit straight segments
    only along the three epitaxial directions (estimated from the image), and merge collinear pieces."""
    la, coh = structure_tensor_orientations(img)
    m = coh > np.percentile(coh, 80)
    th0, _ = dominant_directions(la[m], weights=coh[m])
    mask = morphology.remove_small_objects(filters.apply_hysteresis_threshold(R, low, high), 30)
    sk = morphology.skeletonize(mask)
    merged = []
    for k in range(3):
        d = (th0 + 60 * k) % 180                                   # line direction (deg, y down)
        u = np.array([np.cos(np.radians(d)), np.sin(np.radians(d))])
        n = np.array([-u[1], u[0]])
        normal = ((d - 90 + 90) % 180) - 90                        # Hough normal angle in [-90, 90)
        thetas = np.radians(np.linspace(normal - tol_deg, normal + tol_deg, 9))
        segs = np.array([[x0, y0, x1, y1] for (x0, y0), (x1, y1) in
                         probabilistic_hough_line(sk, threshold=10, line_length=line_len, line_gap=gap, theta=thetas, rng=0)], float)
        if len(segs) == 0:
            continue
        p, q = segs[:, :2], segs[:, 2:]
        s = np.sort(np.stack([p @ u, q @ u], 1), 1)                # interval along the direction
        o = ((p + q) / 2) @ n                                        # lateral offset
        # sweep: group segments with similar lateral offset, then merge overlapping/nearby intervals
        order = np.argsort(o)
        groups, cur = [], [order[0]]
        for i in order[1:]:
            if o[i] - np.mean(o[cur]) <= 2.5:
                cur.append(i)
            else:
                groups.append(cur); cur = [i]
        groups.append(cur)
        for g in groups:
            og = float(np.mean(o[g]))
            iv = sorted((s[i, 0], s[i, 1]) for i in g)
            a0, a1 = iv[0]
            for b0, b1 in iv[1:]:
                if b0 <= a1 + gap:
                    a1 = max(a1, b1)
                else:
                    merged.append((k, og * n + a0 * u, og * n + a1 * u)); a0, a1 = b0, b1
            merged.append((k, og * n + a0 * u, og * n + a1 * u))
    inst = np.zeros(R.shape, np.int32)
    for i, (k, a, b) in enumerate(sorted(merged, key=lambda s: -np.linalg.norm(s[2] - s[1])), start=1):
        rr, cc = draw_line(int(round(a[1])), int(round(a[0])), int(round(b[1])), int(round(b[0])))
        ok = (rr >= 0) & (rr < R.shape[0]) & (cc >= 0) & (cc < R.shape[1])
        inst[rr[ok], cc[ok]] = i                    # 1-px centrelines (shorter segments drawn on top)
    return segmentation.expand_labels(inst, distance=width // 2)


# --------------------------------------------------------------------------
# adapters for compare/score.py
# --------------------------------------------------------------------------

def line_fit_response(img01: np.ndarray) -> np.ndarray:
    """Stacked (dark, bright) Sato maps of the percentile-normalised frame,
    each scaled by its 99.5th percentile exactly as ridge_response does."""
    img = normalize(img01)
    return np.stack([ridge_response(img, SIGMAS, "dark"),
                     ridge_response(img, SIGMAS, "bright")])


def line_fit_mask(img01: np.ndarray, resp: np.ndarray, thr: float, pol: str = "bright",
                  high: float = 0.5, line_len: int = 16, gap: int = 5) -> np.ndarray:
    """Hysteresis (low = thr) -> skeleton -> direction-constrained Hough -> drawn lines."""
    R = {"dark": resp[0], "bright": resp[1], "both": resp.max(0)}[pol]
    return ridge_hough_instances(normalize(img01), R, thr, high, line_len, gap,
                                 width=WIDTH_PX) > 0


def check_port(starter_dir: str) -> None:
    """Port == original on both TiS3 frames, and the original still reproduces
    the review's E1b test scores (outputs/E1b_ridge_hough_TiS3.json)."""
    import json
    import os
    import sys
    sys.path.insert(0, starter_dir)
    import baseline_ridge as orig
    from common import LABEL_SOURCES, load_gray
    from metrics import centreline_scores

    keys = [k for k in LABEL_SOURCES if k.startswith("TiS3")]
    saved = json.load(open(os.path.join(starter_dir, "outputs", "E1b_ridge_hough_TiS3.json"),
                           encoding="utf-8"))
    for tune, test in [(keys[0], keys[1]), (keys[1], keys[0])]:
        rec = saved[f"tune={tune} -> test={test}"]
        prm = rec["params"]
        img = normalize(load_gray(os.path.join(starter_dir, "..", LABEL_SOURCES[test])))
        R_o = orig.ridge_response(img, (1, 2, 3), prm["polarity"])
        R_p = ridge_response(img, (1, 2, 3), prm["polarity"])
        a = orig.ridge_hough_instances(img, R_o, prm["low"], prm["high"], prm["line_len"], prm["gap"])
        b = ridge_hough_instances(img, R_p, prm["low"], prm["high"], prm["line_len"], prm["gap"])
        s = centreline_scores(orig.gt_for(test, img.shape), b)
        print(f"  test {test}: port identical to original: {np.array_equal(a, b)}  "
              f"wire F1 {s['wire_f1']:.3f} (review saved {rec['test']['wire_f1']:.3f})")
