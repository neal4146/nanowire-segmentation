"""Measure the TiS3 nanowires in a folder of SEM photos.

    venv\\Scripts\\python measure\\measure_wires.py "D:\\SEM\\TiS3 on sapphire"
    venv\\Scripts\\python measure\\measure_wires.py "<folder>" --out "<results folder>"

or drag the folder onto measure\\measure_wires.bat. See measure/README.md.

Every FEI SEM TIFF in the folder and its subfolders is brought to the
50,000x scale (5.4 nm per pixel, from the scale the microscope stores in the
file). By default (--method unet) its wires are found by the bake-off's best
pipeline (bakeoff/README.md, round 2; test wire F1 0.716 vs 0.506 for
path_open_clean). The steps:
  1. U-Net E2 gives a wire probability per pixel (measure/predict_probs.py;
     GPU if there is one; cached in <out>/probs/).
  2. The thin mask keeps only the ridge of the probability (thin_masks.py).
  3. OpenCV's line segment detector cuts the mask into straight pieces, so a
     zig-zag becomes one wire per arm (bakeoff/extract_plus.py).
  4. A learned check keeps each piece that looks like a real wire
     (bakeoff/verifier/verifier_learned.py).
Widths come from path_open_clean's mask along each wire. --method
path_open_clean (the old default) instead traces path_open_clean's mask with
scripts/ridge/wire_extract.py, the same steps as compare/run_production.py.
By default only photos taken between 50,000x and 80,000x are measured,
because the methods were only checked at those magnifications. --all-scales
measures the rest too.

Writes to --out (default results/measure/<folder name>/):
  images.csv     one row per photo: wire count, wires per square micron,
                 coverage, median length and width, share in each direction
  wires.csv      one row per wire: length, width, angle, direction, position
  folders.csv    one row per subfolder (one growth run each, in the lab's data)
  overlays/      each photo with its measured wires drawn in direction colours
  run_info.json  the settings and counts, so the run can be repeated
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for _sub in ("scripts/ridge", "compare", "bakeoff", "bakeoff/verifier", "measure"):
    sys.path.insert(0, str(ROOT / _sub))

from scipy import ndimage as ndi                                       # noqa: E402
from scipy.spatial import cKDTree                                      # noqa: E402

import wire_extract                                                    # noqa: E402
from ridge_nanowire_images2 import (TARGET_PX, VALIDATED_BAND,         # noqa: E402
                                    load_frame_50k, read_pixel_size)

NM_PER_PX = TARGET_PX * 1e9           # 5.396 nm per pixel at the 50,000x scale
OVERLAY_MAX_SIDE = 4096
# wire_extract's QC flags passed on to wires.csv. Its two bend flags are left
# out: on the hand-labelled photos they fire on ~47% of the hand-drawn wires too,
# so they do not single out mistakes.
FLAG_WORDS = {"short": "short (under 81 nm)",
              "width_outlier": "width outside 16-162 nm",
              "quarter_line": "ends at a quarter line (may be cut there)"}
METHODS = ("unet", "path_open_clean", "v3_forest")
# --method unet: the bake-off round-2 pipeline (bakeoff/README.md, "Round 2")
MODEL = ROOT / "results" / "bakeoff" / "models" / "E2" / "final.pt"
VERIFIER = ROOT / "results" / "bakeoff" / "devcv" / "verifier_E2_cv" / "verifier_learned.joblib"
VENV_DL = ROOT / "venv_dl" / "Scripts" / "python.exe"
# Two wires touch when their centrelines come this close. A wire ending against
# another's side sits half a width (~4 px at the 45 nm median) from that wire's
# centreline, and LSD stops ~1 px short, so 5 px. It is set from that geometry, not
# fitted. The count is sensitive to it (hand photo 1: 43 / 98 / 114 / 150 junctions
# at 2 / 4 / 5 / 6 px), so compare junction counts only between runs of this tool.
CONTACT_PX = 5.0
JUNCTION_MERGE_PX = 6.0   # touching points closer than this are one junction
SEAM_PX = 4.0             # a piece end this close to a quarter line may continue across it
SEAM_GAP_PX = 8.0         # ... to an end at most this far away (extract_plus's LSD rejoin gap)
QUARTER_LINE_PX = 2.0     # an unjoined end this close to a quarter line is flagged (may be cut)


# --------------------------------------------------------------------------
# finding wires
# --------------------------------------------------------------------------

_cache: dict = {}


def wire_params():
    """wire_extract's production settings."""
    if "params" not in _cache:
        _cache["params"] = wire_extract.parse_args([])
    return _cache["params"]


def wire_mask(img01: np.ndarray, method: str) -> np.ndarray:
    if method == "path_open_clean":
        from pseudo_labels import path_open_clean_fixed
        return path_open_clean_fixed(img01)
    if method == "v3_forest":
        from wire_classifier import PROD_THR, load_classifier, wire_probability
        if "rf" not in _cache:
            _cache["rf"] = load_classifier()
        return wire_probability(_cache["rf"], img01) > PROD_THR
    raise ValueError(method)


def mask_from_file(masks: Path, name: str, shape: tuple) -> np.ndarray:
    """A mask made elsewhere (e.g. bakeoff/predict.py): <name>.png, 255 = wire."""
    for stem in (name, name.replace("×", "x")):
        f = masks / f"{stem}.png"
        if f.exists():
            m = np.array(Image.open(f).convert("L")) > 127
            if m.shape != shape:
                raise ValueError(f"{f.name} is {m.shape}, the photo at 50,000x is {shape}")
            return m
    raise FileNotFoundError(f"no {name}.png in {masks}")


# --------------------------------------------------------------------------
# --method unet: network probabilities -> thin mask -> LSD pieces -> verifier
# --------------------------------------------------------------------------

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prob_file(opts: dict, folder: str, path: Path) -> Path:
    """The cached network output of one photo. The name carries the photo's
    content hash (and the folder the network's), so a replaced or same-named
    photo never reuses another photo's output."""
    return Path(opts["probs"]) / folder / f"{path.stem}_{file_sha256(path)[:12]}_prob.png"


def ensure_probs(photos: list, root: Path, opts: dict) -> None:
    """Network probabilities for every photo that will be measured, unless cached.
    Runs in-process if this Python has torch, otherwise through venv_dl. A photo
    the network cannot read gets no file, and measure_photo marks it failed."""
    todo = []
    for path in photos:
        row, ok = photo_check(path, root, opts)
        if ok and not prob_file(opts, row["folder"], path).exists():
            todo.append({"photo": str(path), "out": str(prob_file(opts, row["folder"], path))})
    if not todo:
        return
    try:
        import torch  # noqa: F401
        has_torch = True
    except ImportError:
        has_torch = False
    print(f"network (U-Net E2) on {len(todo)} photo(s); cached in {opts['probs']}", flush=True)
    if has_torch:
        from predict_probs import predict_items
        predict_items(opts["model"], todo)
        return
    if not VENV_DL.exists():
        raise SystemExit("--method unet needs PyTorch: either this Python or the bake-off's "
                         f"venv_dl ({VENV_DL}, bakeoff/setup_dl_env.sh). "
                         "--method path_open_clean needs neither.")
    with tempfile.TemporaryDirectory() as tmp:
        jobs = Path(tmp) / "jobs.json"
        jobs.write_text(json.dumps({"model": opts["model"], "items": todo}), encoding="utf-8")
        res = subprocess.run([str(VENV_DL), str(ROOT / "measure" / "predict_probs.py"), str(jobs)],
                             env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if res.returncode:
        print(f"the network step stopped with exit code {res.returncode}; photos without "
              f"its output are marked failed", flush=True)


def unet_setup(opts: dict, model: Path, verifier: Path, probs_root: Path) -> dict:
    """Check the files, add them to opts and return what run_info.json records.
    Cached probabilities go in a folder named after the network's hash, so a new
    network never reuses old ones."""
    for f, what in ((model, "network"), (verifier, "piece check")):
        if not f.exists():
            raise SystemExit(f"--method unet: the {what} {f} does not exist "
                             f"(bakeoff/README.md, round 2; or use --method path_open_clean)")
    from thin_masks import CHOICE
    m_sha, v_sha = file_sha256(model), file_sha256(verifier)
    v = verifier_model(str(verifier.resolve()))
    opts.update(model=str(model.resolve()), verifier=str(verifier.resolve()),
                probs=str((probs_root / m_sha[:12]).resolve()))
    return {"pipeline": "U-Net probability -> thin ridge mask -> LSD straight pieces "
                        "(all lengths) -> learned piece check; widths from path_open_clean's mask",
            "model": opts["model"], "model_sha256": m_sha,
            "verifier": opts["verifier"], "verifier_sha256": v_sha,
            "verifier_threshold": v["threshold"], "verifier_trained_on": v["trained_on"],
            "thin_mask": json.loads(CHOICE.read_text(encoding="utf-8"))["variant"],
            "probabilities_cached_in": opts["probs"],
            "contact_px": CONTACT_PX, "junction_merge_px": JUNCTION_MERGE_PX}


def verifier_model(path: str):
    if ("verifier", path) not in _cache:
        import joblib
        _cache[("verifier", path)] = joblib.load(path)
    return _cache[("verifier", path)]


def quarters(shape: tuple) -> list[tuple[slice, slice]]:
    """The photo's 4 quarters, as the bake-off cut its dev and test quarters."""
    H, W = shape
    return [(slice(r0, r1), slice(c0, c1)) for r0, r1 in ((0, H // 2), (H // 2, H))
            for c0, c1 in ((0, W // 2), (W // 2, W))]


def join_across_seams(pieces: list, shape: tuple) -> tuple[list, int]:
    """Join pieces from different quarters that continue each other across a
    quarter line, so a wire crossing it is counted once and at full length. The
    rule is extract_plus.rejoin's (ends within SEAM_GAP_PX, directions within
    REJOIN_DEG, sideways offset within REJOIN_OFFSET_PX), applied only to ends
    within SEAM_PX of a quarter line and only between different quarters."""
    import extract_plus as X
    from skimage.draw import line as draw_line
    H, W = shape
    rm, cm = H // 2, W // 2
    near_seam = lambda pt: abs(pt[0] - (rm - 0.5)) <= SEAM_PX or abs(pt[1] - (cm - 0.5)) <= SEAM_PX  # noqa: E731
    pcs = [{**pc, "_tiles": {pc["_tile"]}} for pc in pieces]
    n_joins = 0
    while True:
        ends = [(i, w, *X._end(np.asarray(pc["pixels"], float), w)) for i, pc in enumerate(pcs)
                if len(pc["pixels"]) >= 2 for w in (0, 1)]
        ends = [e for e in ends if e[3] is not None and near_seam(e[2])]
        best = None
        for a in range(len(ends)):
            for b in range(a + 1, len(ends)):
                (i, wi, pi, di), (j, wj, pj, dj) = ends[a], ends[b]
                if i == j or pcs[i]["_tiles"] & pcs[j]["_tiles"]:
                    continue
                gap = pj - pi
                dist = float(np.linalg.norm(gap))
                if dist > SEAM_GAP_PX:
                    continue
                if np.degrees(np.arccos(np.clip(-(di @ dj), -1, 1))) > X.REJOIN_DEG:
                    continue
                if dist > 0 and (gap @ di < -1.0
                                 or abs(gap[0] * di[1] - gap[1] * di[0]) > X.REJOIN_OFFSET_PX):
                    continue
                if best is None or dist < best[0]:
                    best = (dist, i, wi, j, wj)
        if best is None:
            break
        dist, i, wi, j, wj = best
        Pi = np.asarray(pcs[i]["pixels"], float)
        Pj = np.asarray(pcs[j]["pixels"], float)
        Pi = Pi if wi == 1 else Pi[::-1]
        Pj = Pj if wj == 0 else Pj[::-1]
        a, b = np.round(Pi[-1]).astype(int), np.round(Pj[0]).astype(int)
        rr, cc = draw_line(a[0], a[1], b[0], b[1])
        bridge = np.stack([rr, cc], 1)[1:-1].astype(float)
        joined = {"pixels": np.concatenate([Pi, bridge, Pj]),
                  "length_px": pcs[i]["length_px"] + pcs[j]["length_px"] + dist,
                  "_tiles": pcs[i]["_tiles"] | pcs[j]["_tiles"], "_tile": pcs[i]["_tile"]}
        pcs = [pc for k, pc in enumerate(pcs) if k not in (i, j)] + [joined]
        n_joins += 1
    return pcs, n_joins


def unet_pieces(gray: np.ndarray, prob: np.ndarray, opts: dict) -> tuple[list, np.ndarray]:
    """(kept LSD pieces in photo coordinates, thin mask) exactly as
    bakeoff/final_round2.py scored them, quarter by quarter: E_thin's ridge mask
    of the whole photo, cut into the 4 quarters; all LSD pieces of each quarter;
    the learned check at its fitted threshold. Then pieces that continue each
    other across a quarter line are joined. Run on the whole photo at once, the
    same steps score 0.756 vs 0.769 on the dev quarters (cross-validated)."""
    import extract_plus
    import verifier_learned
    from score import extract_params
    from thin_masks import CHOICE, make_mask
    if "thin" not in _cache:
        _cache["thin"] = json.loads(CHOICE.read_text(encoding="utf-8"))
        _cache["extract_params"] = extract_params()
    th = _cache["thin"]
    mask = make_mask(prob, th["kind"], th["t"], th["r"])
    img01 = (gray / 255.0).astype(np.float32)
    kept = []
    for t, (rs, cs) in enumerate(quarters(mask.shape)):
        pieces = extract_plus.extract(mask[rs, cs], NM_PER_PX, _cache["extract_params"],
                                      {"mode": "lsd", "min_len_px": 0})
        for pc in verifier_learned.verify(pieces, prob[rs, cs], img01[rs, cs],
                                          model=verifier_model(opts["verifier"])):
            P = np.asarray(pc["pixels"], float) + [rs.start, cs.start]
            if len(P) > 1:
                # a piece rejoined inside LSD lists its parts and bridges one after
                # another; order the pixels along the (straight) piece so P[0] and
                # P[-1] are its ends (the pixel set, and so the scoring, is unchanged)
                c = P - P.mean(0)
                P = P[np.argsort(c @ np.linalg.eigh(c.T @ c)[1][:, -1], kind="stable")]
            kept.append({**pc, "pixels": P, "_tile": t})
    kept, _ = join_across_seams(kept, mask.shape)
    H, W = mask.shape
    for pc in kept:
        ends = np.asarray(pc["pixels"], float)[[0, -1]]
        pc["_at_quarter_line"] = bool(
            (np.abs(ends[:, 0] - (H // 2 - 0.5)) <= QUARTER_LINE_PX).any()
            or (np.abs(ends[:, 1] - (W // 2 - 0.5)) <= QUARTER_LINE_PX).any())
    return kept, mask


def contacts(paths: list) -> tuple[int, list]:
    """(junction count, junctions touched per wire). Two wires touch where they
    come within CONTACT_PX (crossings, T-contacts and end-to-end corners);
    touching points within JUNCTION_MERGE_PX of each other are one junction."""
    n = len(paths)
    if n < 2:
        return 0, [0] * n
    lo = np.array([P.min(0) for P in paths]) - CONTACT_PX
    hi = np.array([P.max(0) for P in paths]) + CONTACT_PX
    trees = [cKDTree(P) for P in paths]
    points, owners = [], []
    for i in range(n):
        cand = np.nonzero((lo[i + 1:, 0] <= hi[i, 0]) & (hi[i + 1:, 0] >= lo[i, 0])
                          & (lo[i + 1:, 1] <= hi[i, 1]) & (hi[i + 1:, 1] >= lo[i, 1]))[0] + i + 1
        for j in cand:
            d, k = trees[j].query(paths[i], distance_upper_bound=CONTACT_PX)
            if np.isfinite(d).any():
                a = int(np.argmin(d))
                points.append((paths[i][a] + paths[j][k[a]]) / 2)
                owners.append((i, j))
    if not points:
        return 0, [0] * n
    parent = list(range(len(points)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in cKDTree(np.array(points)).query_pairs(JUNCTION_MERGE_PX):
        parent[find(a)] = find(b)
    roots = [find(x) for x in range(len(points))]
    touched = [set() for _ in range(n)]
    for r, (i, j) in zip(roots, owners):
        touched[i].add(r)
        touched[j].add(r)
    return len(set(roots)), [len(t) for t in touched]


def unet_records(gray: np.ndarray, prob: np.ndarray, opts: dict, p) -> tuple[list, dict, np.ndarray]:
    """wire_extract-style records (the fields image_stats, wires.csv and the overlay
    use) for the kept pieces, a summary with the junction count, and a coverage
    mask (each wire drawn at its measured width)."""
    from pseudo_labels import path_open_clean_fixed
    pieces, _ = unet_pieces(gray, prob, opts)
    H, W = gray.shape
    a_mask = path_open_clean_fixed(gray / 255.0)
    edt = ndi.maximum_filter(ndi.distance_transform_edt(a_mask), size=3)
    paths = [np.asarray(pc["pixels"], float).reshape(-1, 2) for pc in pieces]
    n_junctions, touched = contacts(paths)
    records = []
    for i, (pc, P) in enumerate(zip(pieces, paths)):
        q = np.clip(np.round(P).astype(int), 0, [H - 1, W - 1])
        length_px = float(pc["length_px"])
        orient = wire_extract.pca_orientation(P[:, 0], P[:, 1])
        if np.isfinite(orient):
            devs = [wire_extract.dev_180(orient, c) for c in p.family_centers]
            fi = int(np.argmin(np.abs(devs)))
            family, fdev = p.family_centers[fi], float(devs[fi])
        else:
            fi, family, fdev = -1, float("nan"), float("nan")
        we = p.width_exclude_px
        core = q[we:len(q) - we] if len(q) > 2 * we else q[len(q) // 2:len(q) // 2 + 1]
        on = a_mask[core[:, 0], core[:, 1]]
        w_px = float(2 * edt[core[on, 0], core[on, 1]].mean()) if on.mean() >= 0.5 else float("nan")
        dmin = np.minimum(np.minimum(P[:, 0], P[:, 1]), np.minimum(H - 1 - P[:, 0], W - 1 - P[:, 1]))
        flags = []
        if length_px < p.short_px:
            flags.append("short")
        if np.isfinite(w_px) and not (p.width_min_px <= w_px <= p.width_max_px):
            flags.append("width_outlier")
        if pc.get("_at_quarter_line"):
            flags.append("quarter_line")
        records.append({
            "wire_id": i, "length_px": round(length_px, 2),
            "length_nm": round(length_px * NM_PER_PX, 1),
            "width_mean_px": round(w_px, 2), "width_mean_nm": round(w_px * NM_PER_PX, 1),
            "orientation_deg": round(orient, 2) if np.isfinite(orient) else float("nan"),
            "family": family, "family_dev_deg": round(fdev, 2) if np.isfinite(fdev) else float("nan"),
            "off_family": bool(np.isfinite(fdev) and abs(fdev) > p.off_family_deg),
            "n_junctions_traversed": touched[i],
            "border_touching": bool((dmin < p.border_px).any()),
            "qc_flags": ";".join(flags), "_family_idx": fi, "_pixels": q})
    widths = [r["width_mean_px"] for r in records if np.isfinite(r["width_mean_px"])]
    w_fill = float(np.median(widths)) if widths else 2 * p.width_min_px
    cover = np.zeros((H, W), bool)
    for r in records:
        w = r["width_mean_px"] if np.isfinite(r["width_mean_px"]) else w_fill
        q, pad = r["_pixels"], int(np.ceil(w / 2)) + 1
        y0, x0 = np.maximum(q.min(0) - pad, 0)
        y1, x1 = np.minimum(q.max(0) + pad + 1, [H, W])
        line = np.zeros((y1 - y0, x1 - x0), bool)
        line[q[:, 0] - y0, q[:, 1] - x0] = True
        # a 1 px centre line widened to <= (w-1)/2 on each side is w px across
        # (odd w; w-1 for even w, which 2*EDT widths over-read by about 1 px)
        cover[y0:y1, x0:x1] |= ndi.distance_transform_edt(~line) <= (w - 1) / 2
    return records, {"junctions": {"count": n_junctions}}, cover


# --------------------------------------------------------------------------
# one photo
# --------------------------------------------------------------------------

def direction(rec: dict, centres: list) -> str:
    if rec["off_family"] or rec["_family_idx"] < 0:
        return "other"
    return f"{centres[rec['_family_idx']]:g}"


def _median(x) -> float:
    x = [v for v in x if np.isfinite(v)]
    return round(float(np.median(x)), 1) if x else float("nan")


def image_stats(records: list, summary: dict, mask: np.ndarray, p) -> dict:
    """The measurement columns of images.csv, from wire_extract's output on one
    50,000x-scale mask (also used on hand-label masks by check_accuracy.py)."""
    H, W = mask.shape
    area = H * W * (NM_PER_PX / 1000) ** 2             # um^2
    whole = [r for r in records if not r["border_touching"]]
    dirs = [direction(r, p.family_centers) for r in records]
    share = lambda d: round(dirs.count(d) / len(dirs), 3) if dirs else float("nan")  # noqa: E731
    return {
        "width_um": round(W * NM_PER_PX / 1000, 2), "height_um": round(H * NM_PER_PX / 1000, 2),
        "wire_count": len(records),
        "wires_per_um2": round(len(records) / area, 3),
        "wire_length_um_per_um2": round(sum(r["length_nm"] for r in records) / 1000 / area, 3),
        "coverage_pct": round(100 * float(mask.mean()), 2),
        "whole_wire_count": len(whole),
        "median_length_nm": _median([r["length_nm"] for r in whole]),
        "mean_length_nm": round(float(np.mean([r["length_nm"] for r in whole])), 1)
        if whole else float("nan"),
        "median_width_nm": _median([r["width_mean_nm"] for r in records]),
        **{f"share_{c:g}deg": share(f"{c:g}") for c in p.family_centers},
        "share_other": share("other"),
        "junction_count": summary["junctions"]["count"],
    }


def save_overlay(gray: np.ndarray, records: list, path: Path) -> None:
    """The photo with every measured wire painted in its direction colour."""
    base = np.stack([np.clip(gray, 0, 255).astype(np.uint8)] * 3, axis=2)
    colours = [wire_extract.OFF_FAMILY_COLOR if r["off_family"] else
               wire_extract.FAMILY_COLORS.get(r["_family_idx"], wire_extract.OFF_FAMILY_COLOR)
               for r in records]
    rgb = wire_extract._paint(base, records, colours)
    s = min(1.0, OVERLAY_MAX_SIDE / max(rgb.shape[:2]))
    if s < 1.0:
        rgb = cv2.resize(rgb, (int(rgb.shape[1] * s), int(rgb.shape[0] * s)),
                         interpolation=cv2.INTER_AREA)
    dark = rgb[:100, :130]
    rgb[:100, :130] = (dark * 0.35).astype(np.uint8)  # backing box for the key
    y = 22
    for text, col in (("0 deg", wire_extract.FAMILY_COLORS[0]),
                      ("60 deg", wire_extract.FAMILY_COLORS[1]),
                      ("123.5 deg", wire_extract.FAMILY_COLORS[2]),
                      ("other", wire_extract.OFF_FAMILY_COLOR)):
        cv2.putText(rgb, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(rgb, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1, cv2.LINE_AA)
        y += 22
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


def stage_tilt_deg(path: Path) -> float:
    """Stage tilt from the FEI metadata (StageT, radians); 0 if not recorded."""
    m = re.search(r"StageT=([-+0-9.eE]+)", str(Image.open(path).tag_v2.get(34682)))
    return round(float(np.degrees(float(m.group(1)))), 1) if m else 0.0


def photo_check(path: Path, root: Path, opts: dict) -> tuple[dict, bool]:
    """(the start of its images.csv row, whether it gets measured)."""
    folder = path.parent.relative_to(root).as_posix() if path.parent != root else "."
    row = {"image": path.stem, "folder": folder, "status": "measured"}
    try:
        px_m = read_pixel_size(path)[0]
        tilt = stage_tilt_deg(path)
    except Exception:                                  # noqa: BLE001
        row["status"] = "skipped: no microscope scale in the file (not an FEI TIFF?)"
        return row, False
    factor = px_m / TARGET_PX
    in_band = VALIDATED_BAND[0] - 1e-3 <= factor <= VALIDATED_BAND[1] + 1e-3
    row.update(magnification=int(round(50000 / factor)), nm_per_px=round(px_m * 1e9, 3),
               tilt_deg=tilt, in_checked_range="yes" if in_band else "no")
    if not in_band and not opts["all_scales"]:
        row["status"] = (f"skipped: {50000 / factor:,.0f}x is outside 50,000-80,000x "
                         f"(--all-scales measures it anyway)")
        return row, False
    if abs(tilt) > 1 and not opts["include_tilted"]:
        row["status"] = (f"skipped: stage tilted {tilt:g} deg, which shortens wires and "
                         f"bends angles (--include-tilted measures it anyway)")
        return row, False
    return row, True


def measure_photo(path: Path, root: Path, opts: dict) -> tuple[dict, list]:
    """(one images.csv row, its wires.csv rows). Never raises: a problem
    becomes the row's status."""
    row, ok = photo_check(path, root, opts)
    if not ok:
        return row, []
    folder, tilt = row["folder"], row["tilt_deg"]
    in_band = row["in_checked_range"] == "yes"
    try:
        gray, meta = load_frame_50k(path)
        p = wire_params()
        if opts["masks"]:
            mask = mask_from_file(Path(opts["masks"]), path.stem, gray.shape)
        elif opts["method"] == "unet":
            f = prob_file(opts, folder, path)
            if not f.exists():
                raise RuntimeError("the network step could not process this photo (see the log)")
            prob = np.array(Image.open(f)).astype(np.float32) / 255
            if prob.shape != gray.shape:
                raise ValueError(f"{f.name} is {prob.shape}, the photo at 50,000x is {gray.shape}")
            records, summary, mask = unet_records(gray, prob, opts, p)
        else:
            mask = wire_mask(gray / 255.0, opts["method"])
        if opts["masks"] or opts["method"] != "unet":
            records, summary, _ = wire_extract.extract_from_mask(mask, NM_PER_PX, p)
    except Exception as e:                             # noqa: BLE001
        row["status"] = f"failed: {type(e).__name__}: {e}"
        return row, []

    notes = []
    if meta["mag_mismatch"]:
        notes.append("file name magnification disagrees with the scale in the file; "
                     "the file's scale was used")
    if not in_band:
        notes.append("outside the checked magnifications: treat as rough")
    if meta["unreliable_upscale"]:
        notes.append("low magnification: wires are under 2 px wide in the original")
    if abs(tilt) > 1:
        notes.append(f"stage tilted {tilt:g} deg: lengths across the tilt read short by up to "
                     f"{100 * (1 - np.cos(np.radians(tilt))):.0f}% and angles are bent")
    row.update(image_stats(records, summary, mask, p))
    row["notes"] = "; ".join(notes)

    wires = []
    for r in records:
        P = np.asarray(r["_pixels"], float).reshape(-1, 2)
        cy, cx = P.mean(axis=0)
        wires.append({
            "image": path.stem, "folder": folder, "wire_id": r["wire_id"],
            "length_nm": r["length_nm"], "width_nm": r["width_mean_nm"],
            "angle_deg": r["orientation_deg"], "direction": direction(r, p.family_centers),
            "x_um": round(cx * NM_PER_PX / 1000, 3), "y_um": round(cy * NM_PER_PX / 1000, 3),
            "cut_by_edge": "yes" if r["border_touching"] else "no",
            "junctions_crossed": r["n_junctions_traversed"],
            "checks": "; ".join(FLAG_WORDS[f] for f in r["qc_flags"].split(";")
                                if f in FLAG_WORDS),
        })
    if opts["overlays"]:
        save_overlay(gray, records, Path(opts["out"]) / "overlays" / folder / f"{path.stem}.png")
    return row, wires


# --------------------------------------------------------------------------
# a folder
# --------------------------------------------------------------------------

def find_photos(src: Path) -> list[Path]:
    """Every .tif under src, except hand-label overlays ('... Labeled.tif')."""
    if src.is_file():
        return [src]
    return sorted(p for p in src.rglob("*")
                  if p.suffix.lower() in (".tif", ".tiff") and "label" not in p.stem.lower())


def folder_table(images: pd.DataFrame, wires: pd.DataFrame, centres: list) -> pd.DataFrame:
    """One row per subfolder: means over its photos, medians over its wires."""
    rows = []
    for folder, g in images[images.status == "measured"].groupby("folder", sort=True):
        w = wires[wires.folder == folder]
        whole = w[w.cut_by_edge == "no"]
        row = {"folder": folder, "photos_measured": len(g),
               "wires_measured": len(w)}
        for col in ("wires_per_um2", "wire_length_um_per_um2", "coverage_pct"):
            row[f"{col}_mean"] = round(g[col].mean(), 3)
            row[f"{col}_sd"] = round(g[col].std(ddof=1), 3) if len(g) > 1 else float("nan")
        row["median_length_nm"] = round(whole.length_nm.median(), 1) if len(whole) else float("nan")
        row["median_width_nm"] = round(w.width_nm.median(), 1) if len(w) else float("nan")
        for c in [f"{c:g}" for c in centres] + ["other"]:
            key = f"share_{c}deg" if c != "other" else "share_other"
            row[key] = round(float((w.direction == c).mean()), 3) if len(w) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="See measure/README.md for what every column means.")
    ap.add_argument("folder", type=Path, help="folder of SEM .tif photos (subfolders included), "
                                              "or one .tif")
    ap.add_argument("--out", type=Path, default=None,
                    help="where to write the results (default results/measure/<folder name>)")
    ap.add_argument("--method", choices=METHODS, default="unet",
                    help="how wires are found: unet (default, the bake-off's best: network + "
                         "straight pieces + learned check; needs PyTorch, GPU optional), "
                         "path_open_clean (the old default, no PyTorch), or v3_forest")
    ap.add_argument("--model", type=Path, default=MODEL,
                    help="--method unet: the network (default %(default)s)")
    ap.add_argument("--verifier", type=Path, default=VERIFIER,
                    help="--method unet: the learned piece check (default %(default)s)")
    ap.add_argument("--masks", type=Path, default=None,
                    help="use wire masks made elsewhere instead, e.g. a network's from "
                         "bakeoff/predict.py: <photo name>.png at the 50,000x scale, 255 = wire")
    ap.add_argument("--all-scales", action="store_true",
                    help="also measure photos outside 50,000-80,000x (less reliable)")
    ap.add_argument("--include-tilted", action="store_true",
                    help="also measure photos taken with the stage tilted (lengths and "
                         "angles are distorted)")
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1),
                    help="photos measured at the same time (default %(default)s)")
    ap.add_argument("--no-overlays", action="store_true", help="skip the overlay pictures")
    args = ap.parse_args()

    src = args.folder.resolve()
    if not src.exists():
        raise SystemExit(f"{src} does not exist")
    photos = find_photos(src)
    if not photos:
        raise SystemExit(f"no .tif photos in {src}")
    root = src if src.is_dir() else src.parent
    how = f"masks from {args.masks}" if args.masks else args.method
    out = (args.out or ROOT / "results" / "measure" / (src.name if src.is_dir() else src.stem)).resolve()
    if args.out is None and (out / "run_info.json").exists():
        # never overwrite an earlier run made with another method: use '<name> (<method>)'
        before = json.loads((out / "run_info.json").read_text(encoding="utf-8")).get("method")
        if before != how:
            out = out.with_name(f"{out.name} ({'masks' if args.masks else args.method})")
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(photos)} photos in {src}\nmethod: {how}\nresults: {out}\n", flush=True)

    opts = {"method": args.method, "masks": str(args.masks.resolve()) if args.masks else None,
            "all_scales": args.all_scales, "include_tilted": args.include_tilted,
            "overlays": not args.no_overlays, "out": str(out)}
    t0 = time.perf_counter()
    unet_info = {}
    if args.method == "unet" and not args.masks:
        unet_info = unet_setup(opts, args.model, args.verifier, out / "probs")
        ensure_probs(photos, root, opts)
    rows, wires = [], []
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {pool.submit(measure_photo, p, root, opts): p for p in photos}
        for i, fut in enumerate(as_completed(futures), start=1):
            row, w = fut.result()
            rows.append(row)
            wires.extend(w)
            where = f"{row['folder']}/{row['image']}" if row["folder"] != "." else row["image"]
            if row["status"] == "measured":
                msg = (f"{row['wire_count']:4d} wires, {row['wires_per_um2']:.2f} per um2, "
                       f"median length {row['median_length_nm']:.0f} nm")
            else:
                msg = row["status"]
            print(f"[{i}/{len(photos)}] {where}: {msg}", flush=True)

    images = pd.DataFrame(rows).sort_values(["folder", "image"]).reset_index(drop=True)
    wires_df = pd.DataFrame(wires, columns=[
        "image", "folder", "wire_id", "length_nm", "width_nm", "angle_deg", "direction",
        "x_um", "y_um", "cut_by_edge", "junctions_crossed", "checks"])
    if len(wires_df):
        wires_df = wires_df.sort_values(["folder", "image", "wire_id"])
    centres = wire_params().family_centers
    folders = folder_table(images, wires_df, centres)

    # utf-8-sig so Excel shows the x in 50000x photo names correctly
    images.to_csv(out / "images.csv", index=False, encoding="utf-8-sig")
    wires_df.to_csv(out / "wires.csv", index=False, encoding="utf-8-sig")
    folders.to_csv(out / "folders.csv", index=False, encoding="utf-8-sig")
    status = images.status.str.split(":").str[0].value_counts().to_dict()
    (out / "run_info.json").write_text(json.dumps({
        "date": datetime.now().isoformat(timespec="seconds"),
        "input": str(src), "method": how, "all_scales": args.all_scales,
        "include_tilted": args.include_tilted,
        "photos_found": len(photos), "photos": status, "wires": len(wires_df),
        "nm_per_px_measured_at": round(NM_PER_PX, 4),
        "checked_magnifications": "50,000x-80,000x (rescale 0.625-1.0 to the 50,000x scale)",
        "wire_extract_settings": vars(wire_params()),
        **({"unet": unet_info} if unet_info else {}),
        "seconds": round(time.perf_counter() - t0, 1),
    }, indent=2, default=wire_extract._jsonable), encoding="utf-8")

    print(f"\n{status.get('measured', 0)} measured, {status.get('skipped', 0)} skipped, "
          f"{status.get('failed', 0)} failed, {len(wires_df)} wires, "
          f"{time.perf_counter() - t0:.0f} s")
    if len(folders):
        print("\nper folder (mean over its photos):")
        for _, f in folders.iterrows():
            print(f"  {f.folder:40s} {f.photos_measured:3d} photos  "
                  f"{f.wires_per_um2_mean:5.2f} wires/um2  "
                  f"median length {f.median_length_nm:5.0f} nm  "
                  f"coverage {f.coverage_pct_mean:5.1f}%")
    print(f"\nwrote images.csv, wires.csv, folders.csv, run_info.json"
          f"{' and overlays/' if opts['overlays'] else ''} to {out}")


if __name__ == "__main__":
    main()
