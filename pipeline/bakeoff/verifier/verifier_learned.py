"""Learned piece verifier for E_thin + LSD pieces (self-contained).

    from verifier_learned import verify
    kept = verify(pieces, prob_crop, img_crop)

pieces    : extract_plus.extract(mask, nm, p, {"mode": "lsd", "min_len_px": 0}) output
            (dicts with 'pixels' (N, 2) row/col, 'length_px', optionally 'orientation_deg')
prob_crop : E's probability map (float in [0, 1]) of the same crop
img_crop  : the photo (img01, float in [0, 1], 50k-equivalent scale) of the same crop

Every piece gets a feature vector (geometry, probability along the piece and in
3-6 px side bands, image contrast vs a 6-12 px band, growth-family angle,
nearest longer piece, parallel duplicates, end contacts, border, density); a
small gradient-boosting model (verifier_learned.joblib next to this file,
fitted on the dev quarters: 5 for E_cv, 7 for E2_cv) gives P(piece is a matchable wire); pieces with
P >= THRESHOLD are kept. Nothing depends on the quarter: the rule is fixed.

Zig-zags: each straight arm is its own wire (labelling convention), so a piece
meeting another piece at an angle is NOT treated as a duplicate; only parallel
overlap with a longer piece is.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
import os as _os
MODEL_FILE = Path(_os.environ.get("PV_MODEL_FILE", "")) if _os.environ.get("PV_MODEL_FILE") else None
FAMILY_DEG = (0.0, 60.0, 123.5)
OFFS = np.arange(-12, 13, 1.0)          # across-piece profile offsets (px)
C0 = 12                                 # index of offset 0
NEAR_R = 30.0                           # density radius (px)

FEATURES = [
    "length_px", "extent_px", "len_minus_extent", "rms_resid", "max_resid", "angle_off_family",
    "prob_px_mean", "prob_px_min", "prob_ctr_mean", "prob_ctr_min", "prob_ctr_p10", "prob_ctr_p50",
    "frac_ctr_gt05", "band_hi", "band_lo", "frac_band_gt05", "prob_contrast",
    "img_ctr_minus_band", "img_side_hi", "img_side_lo", "img_z", "img_ctr_std",
    "ext_hi", "ext_lo",
    "d_longer", "ang_longer", "len_ratio_longer", "cov_longer", "cov_any", "dup_flag",
    "par_flag", "par_len_ratio", "n_end_contacts", "corner_contact", "collinear_contact",
    "dist_border", "touch_border", "n_near", "len_near", "prob_window",
]


def _angdiff(a, b):
    d = np.abs((np.asarray(a) - np.asarray(b)) % 180.0)
    return np.minimum(d, 180.0 - d)


def _sample(a, rr, cc):
    """Bilinear sample with NaN outside the array."""
    h, w = a.shape
    out = ndi.map_coordinates(a, [rr.ravel(), cc.ravel()], order=1, mode="nearest").reshape(rr.shape)
    bad = (rr < 0) | (rr > h - 1) | (cc < 0) | (cc > w - 1)
    out = out.astype(np.float64)
    out[bad] = np.nan
    return out


def _nanstat(f, x, default=-1.0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(f(x)) if len(x) else default


def _geom(pc, shape):
    P = np.asarray(pc["pixels"], float).reshape(-1, 2)
    m = P.mean(0)
    if len(P) > 1:
        c = P - m
        d = np.linalg.eigh(c.T @ c)[1][:, -1]
    else:
        d = np.array([0.0, 1.0])
    d = d / np.linalg.norm(d)
    n = np.array([-d[1], d[0]])
    t = (P - m) @ d
    r = (P - m) @ n
    q = np.round(P).astype(int)
    ok = (q[:, 0] >= 0) & (q[:, 0] < shape[0]) & (q[:, 1] >= 0) & (q[:, 1] < shape[1])
    q = np.unique(q[ok], axis=0) if ok.any() else np.zeros((0, 2), int)
    orient = float(np.degrees(np.arctan2(-d[0], d[1])) % 180)
    return {"P": P, "m": m, "d": d, "n": n, "t0": float(t.min()), "t1": float(t.max()),
            "rms": float(np.sqrt(np.mean(r ** 2))), "maxr": float(np.abs(r).max()),
            "q": q, "orient": orient, "length": float(pc["length_px"])}


def piece_features(pieces, prob, img) -> np.ndarray:
    """(n_pieces, len(FEATURES)) feature matrix."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _piece_features(pieces, prob, img)


def _piece_features(pieces, prob, img) -> np.ndarray:
    prob = np.asarray(prob, np.float32)
    img = np.asarray(img, np.float32)
    shape = prob.shape
    G = [_geom(pc, shape) for pc in pieces]
    n = len(G)
    if n == 0:
        return np.zeros((0, len(FEATURES)))
    lengths = np.array([g["length"] for g in G])
    orients = np.array([g["orient"] for g in G])
    mids = np.array([g["m"] for g in G])
    # "longer" = earlier in a stable length-descending order (ties broken by index)
    order = np.argsort(-lengths, kind="stable")
    rank = np.empty(n, int)
    rank[order] = np.arange(n)
    trees = [cKDTree(g["q"]) if len(g["q"]) else None for g in G]
    lo = np.array([g["q"].min(0) if len(g["q"]) else g["m"] for g in G])
    hi = np.array([g["q"].max(0) if len(g["q"]) else g["m"] for g in G])
    mid_tree = cKDTree(mids)
    win = 20
    rows = []
    for j, g in enumerate(G):
        f = {}
        f["length_px"] = g["length"]
        ext = g["t1"] - g["t0"] + 1.0
        f["extent_px"] = ext
        f["len_minus_extent"] = g["length"] - ext
        f["rms_resid"] = g["rms"]
        f["max_resid"] = g["maxr"]
        f["angle_off_family"] = float(min(_angdiff(g["orient"], c) for c in FAMILY_DEG))
        q = g["q"]
        pv = prob[q[:, 0], q[:, 1]] if len(q) else np.array([0.0])
        f["prob_px_mean"] = float(pv.mean())
        f["prob_px_min"] = float(pv.min())
        # across-piece profiles
        s = np.arange(g["t0"], g["t1"] + 0.01, 1.0)
        base = g["m"][None, :] + s[:, None] * g["d"][None, :]
        rr = base[:, 0:1] + OFFS[None, :] * g["n"][0]
        cc = base[:, 1:2] + OFFS[None, :] * g["n"][1]
        pp = _sample(prob, rr, cc)
        ii = _sample(img, rr, cc)
        ctr = np.nanmax(pp[:, C0 - 1:C0 + 2], axis=1) if np.isfinite(pp[:, C0 - 1:C0 + 2]).any() else np.array([np.nan])
        f["prob_ctr_mean"] = _nanstat(np.mean, ctr)
        f["prob_ctr_min"] = _nanstat(np.min, ctr)
        f["prob_ctr_p10"] = _nanstat(lambda x: np.percentile(x, 10), ctr)
        f["prob_ctr_p50"] = _nanstat(np.median, ctr)
        f["frac_ctr_gt05"] = _nanstat(lambda x: np.mean(x > 0.5), ctr, 0.0)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                bl = np.nanmean(pp[:, C0 - 6:C0 - 2], axis=1)          # offsets -6..-3
                br = np.nanmean(pp[:, C0 + 3:C0 + 7], axis=1)          # offsets 3..6
                sa, sb = _nanstat(np.mean, bl), _nanstat(np.mean, br)
                f["band_hi"] = max(sa, sb)
                f["band_lo"] = min(sa, sb)
                bmax = np.fmax(bl, br)
                f["frac_band_gt05"] = _nanstat(lambda x: np.mean(x > 0.5), bmax, 0.0)
                f["prob_contrast"] = f["prob_ctr_mean"] - f["band_hi"]
                ic = np.nanmean(ii[:, C0 - 1:C0 + 2], axis=1)
                il = ii[:, C0 - 12:C0 - 5]                              # offsets -12..-6
                ir = ii[:, C0 + 6:C0 + 13]                              # offsets 6..12
                icm = _nanstat(np.mean, ic, 0.5)
                ila, ira = _nanstat(np.mean, il, icm), _nanstat(np.mean, ir, icm)
                band_all = np.concatenate([il[np.isfinite(il)], ir[np.isfinite(ir)]])
                bm = float(band_all.mean()) if len(band_all) else icm
                bs = float(band_all.std()) if len(band_all) > 2 else 0.0
                f["img_ctr_minus_band"] = icm - bm
                f["img_side_hi"] = max(icm - ila, icm - ira)
                f["img_side_lo"] = min(icm - ila, icm - ira)
                f["img_z"] = (icm - bm) / max(bs, 0.01)
                f["img_ctr_std"] = _nanstat(np.std, ic, 0.0)
                # probability beyond each end (3..10 px past the end, |offset| <= 1)
                ends = []
                for sgn, t_end in ((1.0, g["t1"]), (-1.0, g["t0"])):
                    se = t_end + sgn * np.arange(3, 11, 1.0)
                    b2 = g["m"][None, :] + se[:, None] * g["d"][None, :]
                    r2 = b2[:, 0:1] + np.array([-1.0, 0.0, 1.0])[None, :] * g["n"][0]
                    c2 = b2[:, 1:2] + np.array([-1.0, 0.0, 1.0])[None, :] * g["n"][1]
                    v = _sample(prob, r2, c2)
                    vm = np.nanmax(v, axis=1) if np.isfinite(v).any() else np.array([np.nan])
                    ends.append(_nanstat(np.mean, vm))
        f["ext_hi"] = max(ends)
        f["ext_lo"] = min(ends)
        # neighbours
        cand = np.flatnonzero(np.all(lo <= hi[j] + 15, axis=1) & np.all(hi >= lo[j] - 15, axis=1))
        cand = cand[cand != j]
        d_longer, ang_longer, lr_longer = 50.0, 0.0, 0.0
        cov_l = np.zeros(len(q), bool)
        cov_a = np.zeros(len(q), bool)
        dup = par = 0.0
        par_ratio = 0.0
        endpts = np.array([g["m"] + g["t0"] * g["d"], g["m"] + g["t1"] * g["d"]])
        n_end, corner, colin = 0, 0.0, 0.0
        for k in cand:
            if trees[k] is None or not len(q):
                continue
            dist, _ = trees[k].query(q, distance_upper_bound=50.0)
            dist = np.minimum(dist, 50.0)
            near4 = dist <= 4.0
            a = float(_angdiff(g["orient"], orients[k]))
            cov_a |= near4
            longer = rank[k] < rank[j]
            if longer:
                cov_l |= near4
                dm = float(dist.min())
                if dm < d_longer:
                    d_longer, ang_longer, lr_longer = dm, a, lengths[k] / max(g["length"], 1.0)
                if a < 15 and near4.mean() >= 0.5:
                    dup = 1.0
            if a < 15:
                # parallel neighbour 3-8 px away along most of this piece (flank / twin)
                off = np.abs((G[k]["m"] - g["m"]) @ g["n"])
                tk = (G[k]["q"] - g["m"]) @ g["d"]
                ov = (min(tk.max(), g["t1"]) - max(tk.min(), g["t0"])) / max(ext, 1.0)
                if 2.5 <= off <= 8.0 and ov >= 0.5:
                    par = 1.0
                    par_ratio = max(par_ratio, lengths[k] / max(g["length"], 1.0))
            de, _ = trees[k].query(endpts, distance_upper_bound=50.0)
            if (de <= 3.0).any():
                n_end += int((de <= 3.0).sum())
                if a >= 25:
                    corner = 1.0
                elif a < 15:
                    colin = 1.0
        f["d_longer"] = d_longer
        f["ang_longer"] = ang_longer
        f["len_ratio_longer"] = lr_longer
        f["cov_longer"] = float(cov_l.mean()) if len(q) else 0.0
        f["cov_any"] = float(cov_a.mean()) if len(q) else 0.0
        f["dup_flag"] = dup
        f["par_flag"] = par
        f["par_len_ratio"] = par_ratio
        f["n_end_contacts"] = float(n_end)
        f["corner_contact"] = corner
        f["collinear_contact"] = colin
        if len(q):
            db = np.minimum.reduce([q[:, 0], q[:, 1], shape[0] - 1 - q[:, 0], shape[1] - 1 - q[:, 1]])
            f["dist_border"] = float(min(db.min(), 50))
        else:
            f["dist_border"] = 50.0
        f["touch_border"] = float(f["dist_border"] <= 2)
        nb = [k for k in mid_tree.query_ball_point(g["m"], NEAR_R) if k != j]
        f["n_near"] = float(len(nb))
        f["len_near"] = float(lengths[nb].sum()) if nb else 0.0
        r0, c0 = np.round(g["m"]).astype(int)
        y0, y1 = max(r0 - win, 0), min(r0 + win + 1, shape[0])
        x0, x1 = max(c0 - win, 0), min(c0 + win + 1, shape[1])
        f["prob_window"] = float((prob[y0:y1, x0:x1] > 0.5).mean()) if y1 > y0 and x1 > x0 else 0.0
        rows.append([f[k] for k in FEATURES])
    X = np.asarray(rows, float)
    X[~np.isfinite(X)] = -1.0
    return X


_MODEL = None


def load_model():
    global _MODEL
    if _MODEL is None:
        import joblib
        _MODEL = joblib.load(MODEL_FILE)
    return _MODEL


def piece_scores(pieces, prob_crop, img_crop, model=None) -> np.ndarray:
    model = model or load_model()
    X = piece_features(pieces, prob_crop, img_crop)
    if not len(X):
        return np.zeros(0)
    return model["clf"].predict_proba(X)[:, 1]


def verify(pieces, prob_crop, img_crop, model=None, threshold=None):
    """The pieces the verifier keeps (same dicts, input order)."""
    model = model or load_model()
    thr = model["threshold"] if threshold is None else threshold
    sc = piece_scores(pieces, prob_crop, img_crop, model)
    return [pc for pc, s in zip(pieces, sc) if s >= thr]
