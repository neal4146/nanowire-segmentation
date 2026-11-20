"""Shared helpers: cache loading, per-piece GT coverage, labels, fast and official scoring."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path

import os
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "bakeoff"))
sys.path.insert(0, str(HERE))
import common  # noqa: E402,F401
from common import OUT, read_json  # noqa: E402

# which fold models' dev predictions to use (devcv/pred/<PREFIX><fold>/): E_cv, R_cv, ...
PREFIX = os.environ.get("PV_MODEL", "E_cv")
WORK = OUT / "devcv" / f"verifier_{PREFIX}"
WORK.mkdir(parents=True, exist_ok=True)

import pickle  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from skimage import morphology  # noqa: E402

import extract_plus as xp  # noqa: E402
from wire_metrics import _skel_labels  # noqa: E402

TOL = 4.0
COVER = 0.5
_MAN = read_json(OUT / "devcv" / "manifest.json")
FOLDS = {k: [f"{c['case']}_{c['quarter']}" for c in _MAN["cases"] if c["fold"] == k] for k in _MAN["folds"]}


def thin_mask(prob_full):
    """E_thin's recipe (thin_masks.make_mask with holdout/thin_choice.json) on a full-photo prob map."""
    import json
    from thin_masks import CHOICE, make_mask
    th = json.loads(CHOICE.read_text(encoding="utf-8"))
    return make_mask(prob_full, th["kind"], th["t"], th["r"])


def load_cache():
    with open(WORK / "dev_cache.pkl", "rb") as fh:
        return pickle.load(fh)


def unsure_flags(pieces, unsure, shape):
    """Exactly score_pieces' drop rule: True = piece is not scored."""
    out = np.zeros(len(pieces), bool)
    if unsure is None or not unsure.any():
        return out
    near = ndi.distance_transform_edt(unsure == 0) <= TOL
    for j, pc in enumerate(pieces):
        q = np.round(pc["pixels"]).astype(int)
        q = q[(q[:, 0] >= 0) & (q[:, 0] < shape[0]) & (q[:, 1] >= 0) & (q[:, 1] < shape[1])]
        out[j] = not (not len(q) or near[q[:, 0], q[:, 1]].mean() < 0.5)
    return out


def coverage(pieces, gt_inst):
    """fwd, bwd (n_gt, n_pieces) as centreline_scores computes them, each piece rastered alone.
    Also returns the GT ids with a skeleton and per-piece 'frac of pixels within 4 px of GT i'."""
    shape = gt_inst.shape
    sg = _skel_labels(gt_inst)
    gl = np.unique(sg)[1:]
    G_pix = {i: np.argwhere(gt_inst == i) for i in gl}
    G_sk = {i: np.argwhere(sg == i) for i in gl}
    G_tree = {i: cKDTree(G_pix[i]) for i in gl}
    glo = np.array([G_pix[i].min(0) for i in gl]) if len(gl) else np.zeros((0, 2))
    ghi = np.array([G_pix[i].max(0) for i in gl]) if len(gl) else np.zeros((0, 2))
    n = len(pieces)
    fwd = np.zeros((len(gl), n))
    bwd = np.zeros((len(gl), n))
    near70 = np.zeros((len(gl), n))
    for j, pc in enumerate(pieces):
        q = np.round(pc["pixels"]).astype(int)
        q = q[(q[:, 0] >= 0) & (q[:, 0] < shape[0]) & (q[:, 1] >= 0) & (q[:, 1] < shape[1])]
        if not len(q):
            continue
        q = np.unique(q, axis=0)
        y0, x0 = q.min(0)
        y1, x1 = q.max(0) + 1
        sub = np.zeros((y1 - y0, x1 - x0), bool)
        sub[q[:, 0] - y0, q[:, 1] - x0] = True
        sk = np.argwhere(morphology.skeletonize(sub)) + [y0, x0]
        if not len(sk):
            continue
        tq = cKDTree(q)
        cand = np.flatnonzero(np.all(glo <= q.max(0) + TOL + 1, axis=1) & np.all(ghi >= q.min(0) - TOL - 1, axis=1))
        for ii in cand:
            i = gl[ii]
            d1, _ = G_tree[i].query(sk, distance_upper_bound=TOL + 1e-6)
            fwd[ii, j] = np.mean(d1 <= TOL)
            d2, _ = tq.query(G_sk[i], distance_upper_bound=TOL + 1e-6)
            bwd[ii, j] = np.mean(d2 <= TOL)
            d3, _ = G_tree[i].query(q, distance_upper_bound=TOL + 1e-6)
            near70[ii, j] = np.mean(d3 <= TOL)
    return fwd, bwd, gl, near70


def match(C, keep):
    """(tp, matched piece indices) of a one-to-one Hungarian match on C[:, keep] >= COVER."""
    keep = np.flatnonzero(keep)
    if not len(keep) or not C.shape[0]:
        return 0, np.zeros(0, int)
    Ck = C[:, keep]
    r, c = linear_sum_assignment(-Ck)
    ok = Ck[r, c] >= COVER
    return int(ok.sum()), keep[c[ok]]


def prep_quarter(q):
    """Per-quarter labels and matching data (pieces = all LSD pieces)."""
    pcs = q["pieces"]
    shape = q["mask"].shape
    uns = unsure_flags(pcs, q["unsure"], shape)
    fwd, bwd, gl, near70 = coverage(pcs, q["ref"]["_instances"])
    C = np.minimum(fwd, bwd)
    # label 'match': matched when every (scored) piece is kept. Maximum NUMBER of matches
    # first (every pair with C >= 0.5 weighs 1 + C), so no matchable GT line is lost
    _, mi = match(np.where(C >= COVER, 1.0 + C, 0.0), ~uns)
    y_match = np.zeros(len(pcs), int)
    y_match[mi] = 1
    # label 'spec': >= 70% of pixels within 4 px of ONE GT line, not a parallel duplicate of a
    # longer piece on the same line (longer piece also >= 70% on that line)
    lengths = np.array([pc["length_px"] for pc in pcs])
    best_line = near70.argmax(0) if len(gl) else np.zeros(len(pcs), int)
    on_line = near70.max(0) >= 0.7 if len(gl) else np.zeros(len(pcs), bool)
    y_spec = np.zeros(len(pcs), int)
    order = np.argsort(-lengths, kind="stable")
    taken = set()
    for j in order:
        if on_line[j]:
            if best_line[j] in taken:
                continue
            taken.add(best_line[j])
            y_spec[j] = 1
    return {"uns": uns, "C": C, "n_gt": len(gl), "y_match": y_match, "y_spec": y_spec,
            "lengths": lengths}


def fast_pooled(preps, keeps):
    """Pooled wire (F1, R, P, tp, n_pred, n_gt) of kept sets via per-piece coverage."""
    tp = npred = ngt = 0
    for pr, k in zip(preps, keeps):
        k = np.asarray(k, bool) & ~pr["uns"]
        t, _ = match(pr["C"], k)
        tp += t
        npred += int(k.sum())
        ngt += pr["n_gt"]
    R = tp / max(ngt, 1)
    P = tp / max(npred, 1)
    return 2 * P * R / max(P + R, 1e-9), R, P, tp, npred, ngt


def official(quarters, kept_lists, variant="v"):
    """Official pooled scoring: extract_plus.score_pieces (unsure dropped) + pooled_rows."""
    per_q, counts = [], {}
    for q, kept in zip(quarters, kept_lists):
        s, c = xp.score_pieces(kept, q["mask"], q["frame"], q["ref"], drop_inst=q["unsure"])
        per_q.append({"quarter": q["name"], "fold": q["fold"], "contender": "E_thin", "variant": variant, **s})
        counts.setdefault(("E_thin", variant), {})[q["name"]] = c
    row = xp.pooled_rows(per_q, counts)[0]
    return row, per_q, counts
