"""Crossing-aware wire extraction: Nikhil's wire_extract plus fixes, chosen on a dev set.

    venv/Scripts/python bakeoff/extract_plus.py ceiling   # perfect masks of the dev lines (diagnostic)
    venv/Scripts/python bakeoff/extract_plus.py choose    # pick ONE setting on the dev set
    venv/Scripts/python bakeoff/extract_plus.py test      # the declared batch, scored ONCE on test

Why (results/bakeoff/deepdive_2026-11-19, handover section 15): the network
reaches almost every wire, but wire_extract loses many of them. Two wires that
touch end to end (a V or zig-zag; contacts are ~61 deg, between growth
families) skeletonise into one bent path with no branch point, so they are
never split; the labelling rule draws them as separate wires. Even the GT
lines as perfect 3 px masks score only wire F1 0.81 (test) / 0.79 (dev).

Candidate extractors (GRID, frozen 2026-11-19 before any E_cv dev prediction
was looked at; revised once from the first grid after dev-only evidence:
perfect dev masks, A's real dev masks and the old 80k hold-out E masks showed
that min_area 10 adds false wires, a 30 deg RDP split over-splits real
skeletons, and that a family split and LSD do better on thin real masks):
  prod    wire_extract unchanged (min_area 40)
  rdp     split chains where the RDP polyline turns by > turn_deg, skipping
          vertices within turn_junction_excl px of a junction bridge; drop
          split-off pieces < 8 px (end curls); optional collinear rejoin
  arm     split where the directions of the ARM_PX path on either side of a
          vertex differ by > turn_deg (robust to skeleton wobble), one cut per
          ARM_PX, no junction-bridge cuts, no pieces dropped
  fam     split the MASK into the three growth families (0, 60, 123.5 deg,
          wire_extract's family centres) by openings with line elements of
          length L within +-15 deg, plus a residual; production wire_extract
          on each part; optional minimum piece length
  lsd     OpenCV's line segment detector on the blurred mask; each segment
          moved onto the mask's ridge, duplicate sides suppressed, collinear
          ends rejoined within 8 px; minimum piece length
Rule (fixed before choose ran): best pooled dev wire F1 for E_thin's recipe;
ties within 0.002 -> smaller mean |count error| -> simpler mode
(prod < rdp < arm < fam < lsd) -> GRID order. If the winner beats prod by
less than MIN_GAIN, prod is kept and nothing new is scored on test.
`test` applies the ONE chosen setting to A, v3, E and E_thin (4 new test
variants, '<name>+X'); primary comparison: E_thin+X vs E_thin. It refuses to
run twice, and choose refuses to run after test.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.draw import line as draw_line

from common import OUT, PRED, read_json, read_png

DEV = OUT / "devcv"
CHOICE = DEV / "extract_choice.json"
SUMMARY_PLUS = OUT / "summary_plus.csv"
REJOIN_DEG = 15.0
REJOIN_OFFSET_PX = 2.5
END_FIT_PX = 7
ARM_PX = 10
FAMILY_DEG = (0.0, 60.0, 123.5)       # wire_extract --family-centers
FAMILY_HALF_DEG = 15.0
MIN_GAIN = 0.010
MODE_ORDER = {"prod": 0, "rdp": 1, "arm": 2, "fam": 3, "lsd": 4}

PROD = {"mode": "prod"}
GRID = ([PROD]
        + [{"mode": "rdp", "min_area": a, "turn_deg": t, "rejoin_px": g}
           for a in (40, 20) for t in (30, 45) for g in (0, 8)]
        + [{"mode": "arm", "turn_deg": t, "rejoin_px": g} for t in (30, 45) for g in (0, 8)]
        + [{"mode": "fam", "L": L, "min_len_px": m} for L in (11, 15) for m in (0, 10)]
        + [{"mode": "lsd", "min_len_px": m} for m in (10, 15, 20, 25, 30)])


def cfg_name(c: dict) -> str:
    m = c["mode"]
    if m == "prod":
        return "prod"
    if m == "rdp":
        return f"rdp_a{c['min_area']}_t{c['turn_deg']}_g{c['rejoin_px']}"
    if m == "arm":
        return f"arm_t{c['turn_deg']}_g{c['rejoin_px']}"
    if m == "fam":
        return f"fam_L{c['L']}_m{c['min_len_px']}"
    return f"lsd_m{c['min_len_px']}"


def cfg_by_name(name: str) -> dict:
    return next(c for c in GRID if cfg_name(c) == name)


assert len({cfg_name(c) for c in GRID}) == len(GRID)


# --------------------------------------------------------------------------
# chain splitting
# --------------------------------------------------------------------------

def _path_len(P: np.ndarray) -> float:
    return float(np.hypot(*np.diff(P, axis=0).T).sum()) if len(P) > 1 else 0.0


def _jumps(P: np.ndarray) -> np.ndarray:
    """Indices where a chain path jumps (a bridge across a junction cluster)."""
    return np.flatnonzero(np.hypot(*np.diff(P, axis=0).T) > 1.5) if len(P) > 1 else np.zeros(0, int)


def _turn(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if not nu or not nv:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(u @ v / (nu * nv), -1, 1))))


def _cut(P: np.ndarray, cuts: list[int]) -> list[np.ndarray]:
    bounds = [0] + sorted(cuts) + [len(P) - 1]
    return [P[a:b + 1] for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def split_rdp(P, turn_deg, rdp_tol, junction_excl, min_piece_px=8.0) -> list[np.ndarray]:
    """Cut where the RDP polyline turns by > turn_deg (not next to junction bridges)."""
    import wire_extract
    if len(P) < 5:
        return [P]
    idx = wire_extract.rdp_indices(P, rdp_tol)
    jumps = _jumps(P)
    cuts = []
    for k in range(1, len(idx) - 1):
        if junction_excl and len(jumps) and np.abs(jumps - idx[k]).min() <= junction_excl:
            continue
        if _turn(P[idx[k]] - P[idx[k - 1]], P[idx[k + 1]] - P[idx[k]]) > turn_deg:
            cuts.append(int(idx[k]))
    if not cuts:
        return [P]
    pieces = _cut(P, cuts)
    return [q for q in pieces if _path_len(q) >= min_piece_px] or [max(pieces, key=_path_len)]


def _arm_dir(seg: np.ndarray) -> np.ndarray | None:
    if len(seg) < 3:
        return None
    c = seg - seg.mean(0)
    d = np.linalg.eigh(c.T @ c)[1][:, -1]
    return d if d @ (seg[-1] - seg[0]) >= 0 else -d


def split_arms(P, turn_deg, rdp_tol, junction_excl, arm_px=ARM_PX) -> list[np.ndarray]:
    """Cut at RDP vertices where the ARM_PX of path before and after point in
    directions more than turn_deg apart; strongest first, one cut per arm_px."""
    import wire_extract
    if len(P) < 2 * arm_px:
        return [P]
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
    idx = wire_extract.rdp_indices(P, rdp_tol)
    jumps = _jumps(P)
    cand = []
    for i in idx[1:-1]:
        if junction_excl and len(jumps) and np.abs(jumps - i).min() <= junction_excl:
            continue
        if s[i] < 0.7 * arm_px or s[-1] - s[i] < 0.7 * arm_px:
            continue
        before = P[(s >= s[i] - arm_px) & (s <= s[i] - 1)]
        after = P[(s >= s[i] + 1) & (s <= s[i] + arm_px)]
        a, b = _arm_dir(before), _arm_dir(after)
        if a is None or b is None:
            continue
        t = _turn(a, b)
        if t > turn_deg:
            cand.append((t, int(i)))
    cuts = []
    for t, i in sorted(cand, reverse=True):
        if all(abs(s[i] - s[j]) >= arm_px for j in cuts):
            cuts.append(i)
    return _cut(P, cuts) if cuts else [P]


# --------------------------------------------------------------------------
# rejoin, family split, LSD
# --------------------------------------------------------------------------

def _end(P: np.ndarray, which: int):
    """(end point, unit tangent pointing OUT of the piece) at end 0 or 1."""
    seg = P[:END_FIT_PX] if which == 0 else P[-END_FIT_PX:][::-1]
    if len(seg) < 2:
        return seg[0], None
    c = seg - seg.mean(0)
    d = np.linalg.eigh(c.T @ c)[1][:, -1]
    if d @ (seg[0] - seg[-1]) < 0:
        d = -d
    return seg[0], d / np.linalg.norm(d)


def rejoin(pieces: list[dict], gap_px: float) -> list[dict]:
    """Greedily join piece ends that continue each other across a short gap.
    pieces: {'pixels', 'length_px'}; a joined piece's length adds the gaps."""
    if not gap_px or len(pieces) < 2:
        return pieces
    paths = [pc["pixels"] for pc in pieces]
    ends = [(i, w, *_end(P, w)) for i, P in enumerate(paths) for w in (0, 1) if len(P) >= 2]
    if not ends:
        return pieces
    pts = np.array([e[2] for e in ends])
    cands = []
    for a in range(len(ends)):
        near = np.flatnonzero(np.hypot(*(pts - pts[a]).T) <= gap_px)
        for b in near[near > a]:
            (i, wi, pi, di), (j, wj, pj, dj) = ends[a], ends[b]
            if i == j or di is None or dj is None:
                continue
            gap = pj - pi
            dist = float(np.linalg.norm(gap))
            if np.degrees(np.arccos(np.clip(-(di @ dj), -1, 1))) > REJOIN_DEG:
                continue
            if dist > 0:
                if gap @ di < -1.0:
                    continue
                if abs(gap[0] * di[1] - gap[1] * di[0]) > REJOIN_OFFSET_PX:
                    continue
            cands.append((dist, i, wi, j, wj))
    parent = list(range(len(pieces)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    used, joins = set(), []
    for dist, i, wi, j, wj in sorted(cands):
        if (i, wi) in used or (j, wj) in used or find(i) == find(j):
            continue
        used.update({(i, wi), (j, wj)})
        parent[find(j)] = find(i)
        joins.append((i, paths[i][0 if wi == 0 else -1], paths[j][0 if wj == 0 else -1], dist))
    groups: dict = {}
    for k in range(len(pieces)):
        groups.setdefault(find(k), []).append(k)
    bridges: dict = {}
    for i, a, b, dist in joins:
        bridges.setdefault(find(i), []).append((a, b, dist))
    out = []
    for root, ks in groups.items():
        parts = [paths[k] for k in ks]
        length = sum(pieces[k]["length_px"] for k in ks)
        for a, b, dist in bridges.get(root, []):
            rr, cc = draw_line(*np.round(a).astype(int), *np.round(b).astype(int))
            parts.append(np.stack([rr, cc], 1).astype(float))
            length += dist
        out.append({"pixels": np.concatenate(parts) if len(parts) > 1 else parts[0],
                    "length_px": length})
    return out


def family_parts(mask: np.ndarray, L: int) -> list[np.ndarray]:
    """The mask split by growth family (openings with line elements within
    +-FAMILY_HALF_DEG of each centre), plus the residual not in any family."""
    import cv2
    from ridge_nanowire_images2 import line_kernel
    m8 = mask.astype(np.uint8)
    parts = []
    for f in FAMILY_DEG:
        acc = np.zeros_like(m8)
        for a in np.arange(f - FAMILY_HALF_DEG, f + FAMILY_HALF_DEG + 0.1, 3.0):
            acc |= cv2.morphologyEx(m8, cv2.MORPH_OPEN, line_kernel(L, float(a)))
        parts.append(acc.astype(bool) & mask)
    union = np.any(parts, axis=0)
    parts.append(mask & ~ndi.binary_dilation(union, iterations=1))
    return parts


def lsd_pieces(mask: np.ndarray, shift_max: float = 6.0, rejoin_px: float = 8.0) -> list[dict]:
    """Straight segments by OpenCV's LSD on the blurred mask, moved onto the ridge,
    duplicate sides suppressed (longest first), collinear ends rejoined."""
    import cv2
    shape = mask.shape
    img = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.0)
    det = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(
        np.clip(img * 255, 0, 255).astype(np.uint8))[0]
    if det is None:
        return []
    segs = []
    for x1, y1, x2, y2 in det.reshape(-1, 4):
        a, b = np.array([y1, x1], float), np.array([y2, x2], float)
        d = b - a
        n = float(np.hypot(*d))
        if n < 2:
            continue
        nrm = np.array([-d[1], d[0]]) / n
        mid = (a + b) / 2
        best, best_v = 0.0, -1.0
        for sh in np.arange(-shift_max, shift_max + 0.01, 0.5):
            q = np.round(mid + sh * nrm).astype(int)
            if 0 <= q[0] < shape[0] and 0 <= q[1] < shape[1] and img[q[0], q[1]] > best_v + 1e-6:
                best, best_v = sh, img[q[0], q[1]]
        segs.append((a + best * nrm, b + best * nrm, n))
    segs.sort(key=lambda t: -t[2])
    kept, dist_to_kept = [], np.full(shape, np.inf)
    for a, b, n in segs:
        rr, cc = draw_line(*np.round(a).astype(int), *np.round(b).astype(int))
        ok = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
        rr, cc = rr[ok], cc[ok]
        if not len(rr) or (dist_to_kept[rr, cc] <= 2.0).mean() >= 0.6:
            continue
        kept.append({"pixels": np.stack([rr, cc], 1).astype(float), "length_px": n})
        drawn = np.zeros(shape, bool)
        drawn[rr, cc] = True
        y0, y1 = max(rr.min() - 3, 0), min(rr.max() + 4, shape[0])
        x0, x1 = max(cc.min() - 3, 0), min(cc.max() + 4, shape[1])
        local = ndi.distance_transform_edt(~drawn[y0:y1, x0:x1])
        dist_to_kept[y0:y1, x0:x1] = np.minimum(dist_to_kept[y0:y1, x0:x1], local)
    return rejoin(kept, rejoin_px)


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------

def chains(mask: np.ndarray, nm_per_px: float, p, min_area: int = 40) -> list[dict]:
    """wire_extract's chain records with only --min-area changed."""
    import wire_extract
    q = copy.copy(p)
    q.min_area = min_area
    return wire_extract.extract_from_mask(mask, nm_per_px, q)[0]


def _records_to_pieces(records) -> list[dict]:
    out = []
    for rec in records:
        P = np.asarray(rec["_pixels"], float).reshape(-1, 2)
        out.append({"pixels": P, "length_px": _path_len(P)})
    return out


def extract(mask: np.ndarray, nm_per_px: float, p, cfg: dict, cache: dict | None = None) -> list[dict]:
    """Pieces (one per wire): {'pixels' (N, 2) row/col, 'length_px', 'orientation_deg'}.
    cache: a dict kept per mask so a grid runs each expensive step once."""
    cache = {} if cache is None else cache
    mode = cfg["mode"]

    def recs(min_area):
        key = ("chains", min_area)
        if key not in cache:
            cache[key] = chains(mask, nm_per_px, p, min_area)
        return cache[key]

    if mode == "prod":
        out = _records_to_pieces(recs(40))
    elif mode in ("rdp", "arm"):
        pieces = []
        for rec in recs(cfg.get("min_area", 40)):
            P = np.asarray(rec["_pixels"], float).reshape(-1, 2)
            parts = (split_rdp(P, cfg["turn_deg"], p.rdp_tol, p.turn_junction_excl) if mode == "rdp"
                     else split_arms(P, cfg["turn_deg"], p.rdp_tol, p.turn_junction_excl))
            pieces += [{"pixels": q, "length_px": _path_len(q)} for q in parts]
        out = rejoin(pieces, cfg["rejoin_px"])
    elif mode == "fam":
        key = ("fam", cfg["L"])
        if key not in cache:
            cache[key] = [pc for part in family_parts(mask, cfg["L"]) if part.any()
                          for pc in _records_to_pieces(chains(part, nm_per_px, p, 40))]
        out = [pc for pc in cache[key] if pc["length_px"] >= cfg["min_len_px"]]
    elif mode == "lsd":
        if "lsd" not in cache:
            cache["lsd"] = lsd_pieces(mask)
        out = [pc for pc in cache["lsd"] if pc["length_px"] >= cfg["min_len_px"]]
    else:
        raise ValueError(mode)
    res = []
    for pc in out:
        P = pc["pixels"]
        c = P - P.mean(0)
        d = np.linalg.eigh(c.T @ c)[1][:, -1] if len(P) > 1 else np.array([1.0, 0.0])
        res.append({**pc, "orientation_deg": float(np.degrees(np.arctan2(-d[0], d[1])) % 180)})
    return res


def to_instances(pieces: list[dict], shape: tuple) -> np.ndarray:
    inst = np.zeros(shape, np.int32)
    for i, pc in enumerate(pieces, start=1):
        q = np.round(pc["pixels"]).astype(int)
        ok = (q[:, 0] >= 0) & (q[:, 0] < shape[0]) & (q[:, 1] >= 0) & (q[:, 1] < shape[1])
        inst[q[ok, 0], q[ok, 1]] = i
    return inst


# --------------------------------------------------------------------------
# scoring (same matcher as compare/score.py and the tiles of compare/bootstrap.py)
# --------------------------------------------------------------------------

def score_pieces(pieces, mask, frame, ref, drop_inst=None) -> tuple[dict, dict]:
    """(scores, tile counts) for one quarter. drop_inst: label image of the
    'unsure' lines (dev only); pieces lying mostly on them are not scored."""
    from bootstrap import TILE, _tile_ids
    from score import WIRE_COVER, WIRE_TOL_PX
    from wire_metrics import centreline_scores
    shape = mask.shape
    if drop_inst is not None and drop_inst.any():
        near = ndi.distance_transform_edt(drop_inst == 0) <= WIRE_TOL_PX
        keep = []
        for pc in pieces:
            q = np.round(pc["pixels"]).astype(int)
            q = q[(q[:, 0] >= 0) & (q[:, 0] < shape[0]) & (q[:, 1] >= 0) & (q[:, 1] < shape[1])]
            if not len(q) or near[q[:, 0], q[:, 1]].mean() < 0.5:
                keep.append(pc)
        pieces = keep
    pred = to_instances(pieces, shape)
    gt = ref["_instances"]
    w = centreline_scores(gt, pred, tol=WIRE_TOL_PX, cover=WIRE_COVER, return_matches=True)
    nt = (-(-shape[0] // TILE)) * (-(-shape[1] // TILE))
    tid = _tile_ids(np.indices(shape).reshape(2, -1).T, shape).reshape(shape)
    c = {"pix_tp": np.bincount(tid[mask & frame.gt], minlength=nt),
         "pix_pred": np.bincount(tid[mask], minlength=nt),
         "pix_gt": np.bincount(tid[frame.gt], minlength=nt)}
    for side, inst, ids, matched in (("gt", gt, w["gt_ids"], {g for g, _ in w["matches"]}),
                                     ("pred", pred, w["pred_ids"], {q for _, q in w["matches"]})):
        if len(ids):
            cen = np.asarray(ndi.center_of_mass(inst > 0, inst, ids), float).reshape(-1, 2)
            t, hit = _tile_ids(cen, shape), np.array([i in matched for i in ids])
        else:
            t, hit = np.zeros(0, int), np.zeros(0, bool)
        c[f"wire_{side}"] = np.bincount(t, minlength=nt)
        c[f"wire_{side}_tp"] = np.bincount(t[hit], minlength=nt)
    lens = np.array([pc["length_px"] for pc in pieces]) * frame.nm_per_px
    n_gt = ref["count"]
    med = float(np.median(lens)) if len(lens) else None
    s = {"wire_f1": w["wire_f1"], "wire_recall": w["wire_recall"],
         "wire_precision": w["wire_precision"], "n_wires_gt": n_gt, "n_wires_pred": len(pieces),
         "count_err_pct": 100.0 * (len(pieces) - n_gt) / n_gt,
         "length_median_nm_err_pct": (100.0 * (med - ref["length_median_nm"]) / ref["length_median_nm"]
                                      if med is not None else None),
         "length_total_err_pct": 100.0 * (lens.sum() - ref["_length_total_nm"]) / ref["_length_total_nm"]}
    return s, c


def _lines_raster(lines, shape) -> np.ndarray:
    inst = np.zeros(shape, np.int32)
    for i, L in enumerate(lines, start=1):
        for a, b in zip(L[:-1], L[1:]):
            rr, cc = draw_line(int(round(a[0])), int(round(a[1])), int(round(b[0])), int(round(b[1])))
            ok = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
            inst[rr[ok], cc[ok]] = i
    return inst


def _with_total(ref: dict, lines, nm: float) -> dict:
    ref["_length_total_nm"] = float(sum(np.hypot(*np.diff(L, axis=0).T).sum() for L in lines) * nm)
    return ref


def dev_quarters() -> list[dict]:
    """The 5 dev quarters: frame, ref, unsure-line raster, crop, fold."""
    from dev_cv import read_fix_lines, unsure_lines
    from external import lines_to_frame
    from common import find_photo, load_photo
    man = read_json(DEV / "manifest.json")
    out = []
    for c in man["cases"]:
        img01, fm = load_photo(find_photo(c["photo"]))
        r0, c0 = c["crop_origin_row_col"]
        h, w = c["crop_shape_row_col"]
        crop = (slice(r0, r0 + h), slice(c0, c0 + w))
        nm = fm["pixel_size_m"] * 1e9 / fm["rescale_to_50k"]
        lines = read_fix_lines(c["csv"])
        frame, ref = lines_to_frame(lines, (h, w), nm, f"{c['case']}_{c['quarter']}", img01[crop])
        out.append({**c, "img01": img01, "crop": crop, "frame": frame,
                    "ref": _with_total(ref, lines, nm),
                    "unsure": _lines_raster(unsure_lines(c["csv"]), (h, w))})
    return out


def dev_masks(q: dict) -> dict:
    """Masks of one dev quarter: A, and E's two recipes from the fold model that
    did not train on this photo (E: prob > 0.5; E_thin: the chosen ridge)."""
    from pseudo_labels import path_open_clean_fixed
    from thin_masks import CHOICE as THIN_CHOICE, make_mask
    th = json.loads(THIN_CHOICE.read_text(encoding="utf-8"))
    f = DEV / "pred" / f"E_cv{q['fold']}" / f"{q['case']}_prob.png"
    if not f.exists():
        raise SystemExit(f"{f} missing: run bash bakeoff/dev_cv.sh first")
    prob = read_png(f).astype(np.float32) / 255
    if prob.shape != q["img01"].shape:
        raise SystemExit(f"{f}: shape {prob.shape}, photo {q['img01'].shape}")
    return {"A_path_open_clean": path_open_clean_fixed(q["img01"])[q["crop"]],
            "E": (prob > 0.5)[q["crop"]],
            "E_thin": make_mask(prob, th["kind"], th["t"], th["r"])[q["crop"]]}


def pooled_rows(per_q: list[dict], counts: dict) -> list[dict]:
    from bootstrap import pooled
    import pandas as pd
    df = pd.DataFrame(per_q)
    rows = []
    for (contender, variant), per in counts.items():
        sub = df[(df.contender == contender) & (df.variant == variant)]
        _, wf = pooled(per)
        tot = {k: sum(v[k].sum() for v in per.values())
               for k in ("wire_gt", "wire_gt_tp", "wire_pred", "wire_pred_tp")}
        rows.append({"contender": contender, "variant": variant, "wire_f1": wf,
                     "wire_recall": tot["wire_gt_tp"] / tot["wire_gt"],
                     "wire_precision": tot["wire_pred_tp"] / max(tot["wire_pred"], 1),
                     "count_err_pct_pooled": 100 * (sub.n_wires_pred.sum() - sub.n_wires_gt.sum())
                     / sub.n_wires_gt.sum(),
                     "count_abs_err_pct": sub.count_err_pct.abs().mean(),
                     "length_median_abs_err_pct": sub.length_median_nm_err_pct.abs().mean(),
                     "length_total_abs_err_pct": sub.length_total_err_pct.abs().mean()})
    return rows


def _paired(counts_a: dict, counts_b: dict, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Paired tile-bootstrap wire-F1 difference a - b over the same quarters."""
    from bootstrap import pooled
    qs = sorted(counts_a)
    rng = np.random.default_rng(seed)
    nt = {q: len(counts_a[q]["pix_tp"]) for q in qs}
    d = []
    for _ in range(n):
        draw = {q: rng.integers(0, k, k) for q, k in nt.items()}
        d.append(pooled(counts_a, draw)[1] - pooled(counts_b, draw)[1])
    d = np.array(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def ceiling() -> None:
    """Perfect masks (the dev lines widened to 3 and 5 px) through every candidate."""
    import pandas as pd
    from score import extract_params
    p = extract_params()
    per_q, counts = [], {}
    for q in dev_quarters():
        gt_inst = q["ref"]["_instances"]
        for width in (1.0, 2.0):
            mask = ndi.distance_transform_edt(gt_inst == 0) <= width
            name = f"GT_w{int(2 * width + 1)}"
            cache = {}
            for cfg in GRID:
                s, c = score_pieces(extract(mask, q["frame"].nm_per_px, p, cfg, cache), mask,
                                    q["frame"], q["ref"])
                per_q.append({"quarter": q["frame"].name, "contender": name, "variant": cfg_name(cfg), **s})
                counts.setdefault((name, cfg_name(cfg)), {})[q["frame"].name] = c
        print(f"  {q['frame'].name}: done", flush=True)
    s = pd.DataFrame(pooled_rows(per_q, counts)).sort_values(["contender", "wire_f1"], ascending=[True, False])
    s.to_csv(DEV / "extract_ceiling.csv", index=False, float_format="%.4f")
    print(s.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def _rank_key(row) -> tuple:
    c = cfg_by_name(row.variant)
    return (MODE_ORDER[c["mode"]], [cfg_name(g) for g in GRID].index(row.variant))


def choose() -> None:
    import pandas as pd
    from bootstrap import pooled
    from score import extract_params
    if SUMMARY_PLUS.exists():
        raise SystemExit(f"{SUMMARY_PLUS} exists: the test batch was scored; the dev choice is frozen")
    p = extract_params()
    per_q, counts, counts_nd = [], {}, {}
    for q in dev_quarters():
        for contender, m in dev_masks(q).items():
            cache = {}
            for cfg in GRID:
                pcs = extract(m, q["frame"].nm_per_px, p, cfg, cache)
                s, c = score_pieces(pcs, m, q["frame"], q["ref"], drop_inst=q["unsure"])
                _, c2 = score_pieces(pcs, m, q["frame"], q["ref"])
                per_q.append({"quarter": q["frame"].name, "fold": q["fold"], "contender": contender,
                              "variant": cfg_name(cfg), **s})
                counts.setdefault((contender, cfg_name(cfg)), {})[q["frame"].name] = c
                counts_nd.setdefault((contender, cfg_name(cfg)), {})[q["frame"].name] = c2
        print(f"  {q['frame'].name}: done", flush=True)
    pd.DataFrame(per_q).to_csv(DEV / "extract_dev_per_quarter.csv", index=False, float_format="%.4f")
    s = pd.DataFrame(pooled_rows(per_q, counts))
    s["wire_f1_nodrop"] = [pooled(counts_nd[(r.contender, r.variant)])[1] for r in s.itertuples()]
    s.to_csv(DEV / "extract_dev_scores.csv", index=False, float_format="%.4f")
    for contender in ("E_thin", "E", "A_path_open_clean"):
        sub = s[s.contender == contender].sort_values("wire_f1", ascending=False)
        print(f"\n{contender}:")
        print(sub.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    et = s[s.contender == "E_thin"].copy()
    best_f1 = et.wire_f1.max()
    top = et[et.wire_f1 >= best_f1 - 0.002].copy()
    top["_k"] = [_rank_key(r) for r in top.itertuples()]
    top = top.sort_values(["count_abs_err_pct", "_k"], kind="mergesort")
    best = top.iloc[0]
    prod = et[et.variant == "prod"].iloc[0]
    gain = float(best.wire_f1 - prod.wire_f1)
    d = _paired(counts[("E_thin", best.variant)], counts[("E_thin", "prod")])
    adopted = best.variant if gain >= MIN_GAIN else "prod"
    CHOICE.write_text(json.dumps({
        "variant": adopted, "cfg": cfg_by_name(adopted), "best_on_dev": best.variant,
        "rule": "best pooled dev wire F1 for E_thin; ties within 0.002 -> smaller mean |count error| "
                f"-> simpler mode -> GRID order; adopted only if it beats prod by >= {MIN_GAIN}",
        "dev_wire_f1_best": float(best.wire_f1), "dev_wire_f1_prod": float(prod.wire_f1),
        "dev_gain": gain, "dev_gain_tile_bootstrap": d,
        "dev_set": "results/bakeoff/devcv/manifest.json (5 complete-mode hand-fix quarters, fold models)",
        "extract_plus_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "candidates": [cfg_name(c) for c in GRID]}, indent=2), encoding="utf-8")
    print(f"\nbest on dev: {best.variant}: wire F1 {best.wire_f1:.3f} vs prod {prod.wire_f1:.3f} "
          f"(gain {gain:+.3f}, tile bootstrap [{d[1]:+.3f}, {d[2]:+.3f}]); adopted: {adopted}. Wrote {CHOICE}")


def test() -> None:
    """The declared batch: the chosen setting on A, v3, E, E_thin, scored once."""
    import pandas as pd
    from bootstrap import pooled
    from external import lines_to_frame, read_lines
    from pseudo_labels import path_open_clean_fixed
    from score import extract_params
    from score_bakeoff import REF, label_sets
    from wire_classifier import PROD_THR, load_classifier, wire_probability
    from common import TRAIN, TRAIN_E, case_name, find_photo, load_photo, photo_name, run_of, test_photos
    if not CHOICE.exists():
        raise SystemExit(f"{CHOICE} missing: run `extract_plus.py choose` first (dev set only)")
    if SUMMARY_PLUS.exists():
        raise SystemExit(f"{SUMMARY_PLUS} exists: the declared batch was already scored once")
    ch = json.loads(CHOICE.read_text(encoding="utf-8"))
    if ch["variant"] == "prod":
        raise SystemExit("the dev choice kept the production extraction: nothing new to score on test")
    cfg = ch["cfg"]
    batch = ["A_path_open_clean", "v3_production", "E", "E_thin"]
    print(f"declared batch: {batch} x {{prod, {ch['variant']}}}; primary: E_thin+X vs E_thin")
    tests = set(test_photos())
    trained_on = {c["photo"] for t in (TRAIN, TRAIN_E) if (t / "manifest.json").exists()
                  for c in read_json(t / "manifest.json")["cases"]}
    p = extract_params()
    clf = load_classifier()
    per_q, counts, quarters = [], {}, []
    for csv_path, meta in label_sets():
        photo = photo_name(meta["photo"])
        if photo not in tests or photo in trained_on:
            continue
        tag = f"{case_name(photo)}_{meta['quarter']}"
        lines = read_lines(csv_path)
        img01, fm = load_photo(find_photo(photo))
        if abs(fm["rescale_to_50k"] - 1) > 1e-3:
            raise SystemExit(f"{photo} is not a 50,000x photo")
        r0, c0 = meta["crop_origin_row_col"]
        h, w = meta["crop_shape_row_col"]
        crop = (slice(r0, r0 + h), slice(c0, c0 + w))
        nm = fm["pixel_size_m"] * 1e9 / fm["rescale_to_50k"]
        frame, ref = lines_to_frame(lines, (h, w), nm, tag, img01[crop])
        ref = _with_total(ref, lines, nm)
        masks = {"A_path_open_clean": path_open_clean_fixed(img01)[crop],
                 "v3_production": (wire_probability(clf, img01) > PROD_THR)[crop]}
        for name in ("E", "E_thin"):
            full = read_png(PRED / name / f"{case_name(photo)}.png")
            if full.shape != img01.shape:
                raise SystemExit(f"{name}: shape {full.shape}, photo {img01.shape}")
            masks[name] = (full > 127)[crop]
        for name, m in masks.items():
            cache = {}
            for vname, c in (("prod", PROD), (ch["variant"], cfg)):
                s, tc = score_pieces(extract(m, nm, p, c, cache), m, frame, ref)
                key = name if vname == "prod" else f"{name}+X"
                per_q.append({"quarter": tag, "photo": photo, "run": run_of(photo), "contender": key,
                              "variant": vname, **s})
                counts.setdefault(key, {})[tag] = tc
        quarters.append(tag)
        print(f"  {tag}: scored", flush=True)
    quarters = sorted(quarters)                      # draws in score_bakeoff's order
    df = pd.DataFrame(per_q)
    stamp = {"extract_plus_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "choice_sha256": hashlib.sha256(CHOICE.read_bytes()).hexdigest()}
    df.assign(**stamp).to_csv(OUT / "scores_per_quarter_plus.csv", index=False, float_format="%.4f")
    rng = np.random.default_rng(0)
    ntiles = {q: len(counts[REF][q]["pix_tp"]) for q in quarters}
    draws = [{q: rng.integers(0, k, k) for q, k in ntiles.items()} for _ in range(2000)]
    boot = {n: np.array([pooled(per, d)[1] for d in draws]) for n, per in counts.items()}
    rows = []
    for n, per in counts.items():
        sub = df[df.contender == n]
        row = {"contender": n, "variant": sub.variant.iloc[0], "wire_f1": pooled(per)[1],
               "wire_f1_lo": np.percentile(boot[n], 2.5), "wire_f1_hi": np.percentile(boot[n], 97.5),
               "wire_recall": sub.wire_recall.mean(), "wire_precision": sub.wire_precision.mean(),
               "count_abs_err_pct": sub.count_err_pct.abs().mean(),
               "count_err_pct_pooled": 100 * (sub.n_wires_pred.sum() - sub.n_wires_gt.sum())
               / sub.n_wires_gt.sum(),
               "length_median_abs_err_pct": sub.length_median_nm_err_pct.abs().mean(),
               "length_total_abs_err_pct": sub.length_total_err_pct.abs().mean()}
        for ref_name in (REF, f"{REF}+X", n.replace("+X", "")):
            if ref_name == n or ref_name not in boot:
                continue
            d = boot[n] - boot[ref_name]
            row[f"d_vs_{ref_name}"] = float(d.mean())
            row[f"d_vs_{ref_name}_lo"] = float(np.percentile(d, 2.5))
            row[f"d_vs_{ref_name}_hi"] = float(np.percentile(d, 97.5))
            row[f"p_better_vs_{ref_name}"] = float(np.mean(d > 0))
        for run in sorted(df.run.unique()):
            qs = [q for q in quarters if df[df.quarter == q].run.iloc[0] == run]
            row[f"wire_f1_{run.replace(' ', '')}"] = pooled({q: per[q] for q in qs})[1]
        for ph in sorted(df.photo.unique()):
            qs = [q for q in quarters if df[df.quarter == q].photo.iloc[0] == ph]
            row[f"wire_f1_{case_name(ph).rsplit('_', 1)[-1]}"] = pooled({q: per[q] for q in qs})[1]
        rows.append(row)
    s = pd.DataFrame(rows).sort_values("wire_f1", ascending=False).assign(**stamp)
    s.to_csv(SUMMARY_PLUS, index=False, float_format="%.4f")
    print(s.drop(columns=list(stamp)).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nwrote {SUMMARY_PLUS} and scores_per_quarter_plus.csv")


if __name__ == "__main__":
    {"ceiling": ceiling, "choose": choose, "test": test}[sys.argv[1]]()
