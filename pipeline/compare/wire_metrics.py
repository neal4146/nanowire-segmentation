"""Evaluation metrics for instance segmentation of wires.

Copied from the review's starter_code/metrics.py so compare/ runs without that
folder. score.py uses centreline_scores() for the wire-level score. The only
change is centreline_scores(return_matches=True), which also returns the matched
wire pairs; bootstrap.py needs them. Scores are unchanged.

mask_instance_scores: IoU-matched precision/recall/F1 (Hungarian matching, IoU >= thr), AJI+ and
                      foreground IoU. Use for wide NbS3 belts.
centreline_scores:    for thin wires whose hand outlines are loose (TiS3). Compares skeletons with a
                      distance tolerance (completeness/correctness, as used for road extraction) and
                      matches individual wires by mutual centreline coverage.
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage import morphology


def _overlap(gt, pr):
    ng, npd = int(gt.max()) + 1, int(pr.max()) + 1
    inter = np.bincount((gt.ravel() * npd + pr.ravel()).astype(np.int64), minlength=ng * npd).reshape(ng, npd)
    ag, ap = inter.sum(1), inter.sum(0)
    union = ag[:, None] + ap[None, :] - inter
    return inter[1:, 1:], union[1:, 1:], ag[1:], ap[1:]


def mask_instance_scores(gt, pr, thr=0.5):
    inter, union, ag, ap = _overlap(gt, pr)
    fg_iou = float(((gt > 0) & (pr > 0)).sum() / max(((gt > 0) | (pr > 0)).sum(), 1))
    if inter.size == 0:
        return dict(precision=0.0, recall=0.0, f1=0.0, aji_plus=0.0, fg_iou=fg_iou, n_gt=int(len(ag)), n_pred=int(len(ap)))
    iou = inter / np.maximum(union, 1)
    r, c = linear_sum_assignment(-iou)
    tp = int((iou[r, c] >= thr).sum())
    P, R = tp / max(len(ap), 1), tp / max(len(ag), 1)
    m = iou[r, c] > 0                      # AJI+: one-to-one (Hungarian) pairs, unmatched areas added to union
    I = inter[r[m], c[m]].sum()
    U = union[r[m], c[m]].sum()
    U += ag[np.setdiff1d(np.arange(len(ag)), r[m])].sum() + ap[np.setdiff1d(np.arange(len(ap)), c[m])].sum()
    return dict(precision=P, recall=R, f1=2 * P * R / max(P + R, 1e-9), aji_plus=float(I / max(U, 1)),
                fg_iou=fg_iou, n_gt=int(len(ag)), n_pred=int(len(ap)))


def _skel_labels(inst):
    """Skeleton of every instance separately (bbox-local, so touching wires keep separate centrelines)."""
    out = np.zeros_like(inst)
    for lab, sl in enumerate(ndi.find_objects(inst), start=1):
        if sl is None:
            continue
        sub = inst[sl] == lab
        out[sl][morphology.skeletonize(sub)] = lab
    return out


def centreline_scores(gt, pr, tol=4, cover=0.5, return_matches=False):
    sg, sp = _skel_labels(gt), _skel_labels(pr)
    dg = ndi.distance_transform_edt(sg == 0)
    dp = ndi.distance_transform_edt(sp == 0)
    completeness = float((dp[sg > 0] <= tol).mean()) if (sg > 0).any() else 0.0
    correctness = float((dg[sp > 0] <= tol).mean()) if (sp > 0).any() else 0.0
    gl, pl = np.unique(sg)[1:], np.unique(sp)[1:]
    gi, pj = {v: i for i, v in enumerate(gl)}, {v: j for j, v in enumerate(pl)}
    len_g = np.bincount(sg.ravel(), minlength=int(gt.max()) + 1)
    len_p = np.bincount(sp.ravel(), minlength=int(pr.max()) + 1)
    fwd = np.zeros((len(gl), len(pl)))    # fraction of predicted centreline lying within tol of GT wire i
    bwd = np.zeros((len(gl), len(pl)))    # fraction of GT centreline lying within tol of predicted wire j
    fp = morphology.disk(tol)

    def near_counts(inst, sl, lab, skel):
        """For one instance, count skeleton pixels of the other set within tol of it (bbox-local)."""
        y0, y1 = max(sl[0].start - tol, 0), min(sl[0].stop + tol, inst.shape[0])
        x0, x1 = max(sl[1].start - tol, 0), min(sl[1].stop + tol, inst.shape[1])
        zone = ndi.binary_dilation(inst[y0:y1, x0:x1] == lab, fp)
        vals = skel[y0:y1, x0:x1][zone]
        return np.unique(vals[vals > 0], return_counts=True)

    for lab, sl in enumerate(ndi.find_objects(gt), start=1):
        if sl is None or lab not in gi:
            continue
        for v, n in zip(*near_counts(gt, sl, lab, sp)):
            fwd[gi[lab], pj[v]] = n / len_p[v]
    for lab, sl in enumerate(ndi.find_objects(pr), start=1):
        if sl is None or lab not in pj:
            continue
        for v, n in zip(*near_counts(pr, sl, lab, sg)):
            bwd[gi[v], pj[lab]] = n / len_g[v]
    C = np.minimum(fwd, bwd)                               # mutual coverage
    tp = 0
    matches = []
    if C.size:
        r, c = linear_sum_assignment(-C)
        ok = C[r, c] >= cover
        tp = int(ok.sum())
        matches = [(int(gl[i]), int(pl[j])) for i, j in zip(r[ok], c[ok])]
    P, R = tp / max(len(pl), 1), tp / max(len(gl), 1)
    out = dict(completeness=completeness, correctness=correctness, wire_precision=P, wire_recall=R,
               wire_f1=2 * P * R / max(P + R, 1e-9), n_gt=int(len(gl)), n_pred=int(len(pl)))
    if return_matches:          # ids with a centreline, and the matched (gt id, pred id) pairs
        out.update(gt_ids=gl.astype(int), pred_ids=pl.astype(int), matches=matches)
    return out
