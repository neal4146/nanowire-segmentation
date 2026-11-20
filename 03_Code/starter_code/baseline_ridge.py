"""Experiment E1: classical ridge-filter baseline, scored against the converted hand labels.

Pipeline: two-polarity multi-scale Sato ridge filter -> hysteresis threshold -> skeleton ->
split at junctions -> grow each branch back into the mask (one instance per branch).
Parameters are tuned by grid search on one image per material and then applied unchanged to the
other images of that material. Run from the Materials folder:  python 03_Code/starter_code/baseline_ridge.py
"""
import os, sys, json, itertools
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import filters, morphology, measure, segmentation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LABEL_SOURCES, load_gray, normalize
from metrics import mask_instance_scores, centreline_scores

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(HERE, "outputs", "labels")


def ridge_response(img, sigmas, polarity="both"):
    out = []
    if polarity in ("both", "bright"):
        r = filters.sato(img, sigmas=sigmas, black_ridges=False)
        out.append(r / (np.percentile(r, 99.5) + 1e-9))
    if polarity in ("both", "dark"):
        r = filters.sato(img, sigmas=sigmas, black_ridges=True)
        out.append(r / (np.percentile(r, 99.5) + 1e-9))
    return np.max(out, 0)


def ridge_instances(R, low, high, min_size=60, min_len=15, grow=6):
    mask = filters.apply_hysteresis_threshold(R, low, high)
    mask = morphology.remove_small_objects(mask, min_size)
    sk = morphology.skeletonize(mask)
    nb = ndi.convolve(sk.astype(int), np.ones((3, 3), int), mode="constant") - sk
    branches = sk & ~morphology.binary_dilation(sk & (nb >= 3), morphology.disk(1))
    lab = measure.label(branches, connectivity=2)
    sizes = np.bincount(lab.ravel())
    lab[sizes[lab] < min_len] = 0
    inst = segmentation.expand_labels(lab, distance=grow) * mask
    return segmentation.relabel_sequential(inst)[0]


def ridge_hough_instances(img, R, low, high, line_len=20, gap=6, width=5, tol_deg=4):
    """Improved classical baseline for thin wires: threshold the ridge map, then fit straight segments
    only along the three epitaxial directions (estimated from the image), and merge collinear pieces."""
    from skimage.transform import probabilistic_hough_line
    from measure import structure_tensor_orientations, dominant_directions
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
    from skimage.draw import line as draw_line
    for i, (k, a, b) in enumerate(sorted(merged, key=lambda s: -np.linalg.norm(s[2] - s[1])), start=1):
        rr, cc = draw_line(int(round(a[1])), int(round(a[0])), int(round(b[1])), int(round(b[0])))
        ok = (rr >= 0) & (rr < R.shape[0]) & (cc >= 0) & (cc < R.shape[1])
        inst[rr[ok], cc[ok]] = i                    # 1-px centrelines (shorter segments drawn on top)
    return segmentation.expand_labels(inst, distance=width // 2)


def gt_for(key, shape):
    g = np.array(Image.open(os.path.join(LAB, key + "_instances.png"))).astype(np.int64)
    if g.shape != shape:
        g = np.array(Image.fromarray(g.astype(np.int32)).resize((shape[1], shape[0]), Image.NEAREST)).astype(np.int64)
    return g


def score(key, inst, gt):
    return centreline_scores(gt, inst) if key.startswith("TiS3") else mask_instance_scores(gt, inst)


if __name__ == "__main__":
    groups = {"TiS3": [k for k in LABEL_SOURCES if k.startswith("TiS3")],
              "NbS3": [k for k in LABEL_SOURCES if k.startswith("NbS3")]}
    grids = {"TiS3": dict(sig=[(1, 2), (1, 2, 3), (2, 3)], low=[0.08, 0.12, 0.18], high=[0.3, 0.45], pol=["both", "bright", "dark"]),
             "NbS3": dict(sig=[(2, 4), (2, 4, 8), (4, 8)], low=[0.1, 0.15, 0.25], high=[0.35, 0.5], pol=["both", "bright"])}
    tune_on = {"TiS3": "TiS3_80000×_1", "NbS3": "NbS3 4"}
    results = {}
    for mat, keys in groups.items():
        imgs = {k: normalize(load_gray(os.path.join(HERE, "..", "..", LABEL_SOURCES[k]))) for k in keys}
        if "NbS3 2" in imgs:   # its label image was down-scaled to 76 %; evaluate at label resolution
            a = imgs["NbS3 2"]
            imgs["NbS3 2"] = np.array(Image.fromarray((a * 255).astype(np.uint8)).resize((2331, 1554), Image.BILINEAR)) / 255.0
        gts = {k: gt_for(k, imgs[k].shape) for k in keys}
        G, best = grids[mat], None
        for sig, pol in itertools.product(G["sig"], G["pol"]):
            R = ridge_response(imgs[tune_on[mat]], sig, pol)
            for lo, hi in itertools.product(G["low"], G["high"]):
                s = score(tune_on[mat], ridge_instances(R, lo, hi), gts[tune_on[mat]])
                f = s["wire_f1"] if mat == "TiS3" else s["f1"]
                if best is None or f > best[0]:
                    best = (f, sig, pol, lo, hi)
        f, sig, pol, lo, hi = best
        print(mat, "best (tuned on %s): sigmas=%s polarity=%s low=%.2f high=%.2f tune-F1=%.3f" % (tune_on[mat], sig, pol, lo, hi, f))
        results[mat] = {"params": dict(sigmas=sig, polarity=pol, low=lo, high=hi), "per_image": {}}
        for k in keys:
            for p in (["both", "bright", "dark"] if mat == "TiS3" else [pol]):
                inst = ridge_instances(ridge_response(imgs[k], sig, p), lo, hi)
                s = {a: (round(b, 3) if isinstance(b, float) else b) for a, b in score(k, inst, gts[k]).items()}
                name = k + " [" + p + "]" + (" (tuning image)" if k == tune_on[mat] else "")
                results[mat]["per_image"][name] = s
                print("  ", name, s)
    out = os.path.join(HERE, "outputs", "E1_ridge_baseline.json")
    json.dump(results, open(out, "w"), indent=1, default=str)
    print("saved", out)
