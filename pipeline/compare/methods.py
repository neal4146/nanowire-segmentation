"""Methods raced through compare/score.py.

Each method turns a 50k-equivalent frame (img01, [0, 1]) into a wire-likeness
map where higher means more wire-like; score.py thresholds it. To add a filter,
write its response function and append a Method to METHODS.

References (Nikhil's pipelines, same code paths as production):
  sato_dark_v1   sato sigmas 1-3, dark ridges, thr 0.10        README F1 0.500
  sato_dark_v2   v1 response, thr 0.14 + directional closing    README F1 0.534
  rf_v3          17-feature random forest, p > 0.70             README F1 0.571
                 (retrained here on ONE labelled frame per direction)
Candidates from the handover guide (Step 6):
  sato_bright    sato sigmas 1-3, bright ridges (the wires' light rims/cores)
  frangi_dark    frangi sigmas 1-3, dark ridges
  sato_line_fit  Sato + Hough line fitting along the three growth directions
                 (review E1b, line_fit.py). Searches polarity, hysteresis
                 low/high, minimum line length and gap: the review's grid
                 extended with dark polarity, higher thresholds and longer
                 lines, lengths scaled 80k -> 50k.
  local_std      HIGH local standard deviation (sigma 2); drives ~44% of the RF.
                 Polarity checked on both GT frames: median std on wire pixels is
                 ~2x the background's (0.050 vs 0.022), so wires are the
                 high-variance structure at this scale (dark body + bright rims),
                 not the smooth one. Low-std scores pixel F1 ~0.003.
  sato_both      max of dark and bright sato (each scaled by its 99.5th percentile)
  path_open      sato dark, then a grayscale opening along the three growth
                 directions (path openings reduced to straight paths, since TiS3
                 wires are straight): keeps long thin ridges, drops round blobs.
                 Searches the line length and the v2 junction closing.
  steger         Steger line detector (steger.py): centre lines + per-point width;
                 the mask is painted at the detected width. Searches hysteresis
                 low/high and minimum chain length.
Follow-ups to path_open:
  path_open_clean  path_open (closing always on) with options the tuning may
                 switch on: removal of bright compact crystals, 1 px thinning,
                 and a narrower sato (sigmas 1-2) to curb over-marking.
  rf_pathfeat    the v3 forest with the two line-opened sato maps as extra inputs
  rf_pathopen    the v3 forest's probability map put through the path opening
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import scipy.ndimage as ndi
from skimage.filters import sato
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ridge"))

from line_fit import line_fit_mask, line_fit_response           # noqa: E402
from ridge_nanowire_images2 import dir_close, line_kernel       # noqa: E402
from steger import steger_mask, steger_response                 # noqa: E402
from tune_pseudo_mask import ridge_response                     # noqa: E402
from wire_classifier import (FEATURE_NAMES, N_TRAIN, PROD_THR, SEED,  # noqa: E402
                             feature_stack, wire_probability)
from score import Method                                        # noqa: E402

CACHE = ROOT / "results" / "compare" / "cache"


def minmax(r: np.ndarray) -> np.ndarray:
    return (r - r.min()) / (r.max() - r.min() + 1e-12)


def sato_dark(img01: np.ndarray) -> np.ndarray:
    return ridge_response(img01, "sato", (1, 2, 3))      # exactly tune_pseudo_mask


def sato_bright(img01: np.ndarray) -> np.ndarray:
    return minmax(sato(img01, sigmas=(1, 2, 3), black_ridges=False))


def frangi_dark(img01: np.ndarray) -> np.ndarray:
    return ridge_response(img01, "frangi", (1, 2, 3))


def local_std(img01: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Normalised local std (same std feature as the RF's 'std2')."""
    m = ndi.gaussian_filter(img01, sigma)
    m2 = ndi.gaussian_filter(img01 ** 2, sigma)
    return minmax(np.sqrt(np.clip(m2 - m * m, 0, None)))


def sato_both(img01: np.ndarray) -> np.ndarray:
    d = sato(img01, sigmas=(1, 2, 3), black_ridges=True)
    b = sato(img01, sigmas=(1, 2, 3), black_ridges=False)
    return minmax(np.maximum(d / (np.percentile(d, 99.5) + 1e-12),
                             b / (np.percentile(b, 99.5) + 1e-12)))


# growth directions +/- 3 deg; 0/58/122 covers both angle conventions in use
# (0/60/123.5 and its mirror 0/56.5/120 kept the same share of GT pixels)
OPEN_ANGLES = tuple(a + d for a in (0.0, 58.0, 122.0) for d in (-3.0, 0.0, 3.0))


def line_open(resp: np.ndarray, length: int) -> np.ndarray:
    """Max over the growth directions of a grayscale line opening: a pixel stays
    bright only if a straight run of `length` px through it is bright too."""
    r = resp.astype(np.float32)
    return np.max([cv2.morphologyEx(r, cv2.MORPH_OPEN, line_kernel(length, a))
                   for a in OPEN_ANGLES], axis=0)


def path_open_mask(img01: np.ndarray, resp: np.ndarray, thr: float, length: int = 17,
                   close: bool = False) -> np.ndarray:
    m = line_open(resp, length) > thr
    return dir_close(m) if close else m


def bright_particles(img01: np.ndarray, k: float = 6.0, r: int = 3, halo: int = 4) -> np.ndarray:
    """Bright compact crystals: pixels > k robust sigmas above the local mean
    whose bright region is too thick for a wire (survives a radius-r disc
    opening), grown by `halo` px to cover their dark rim."""
    d = img01 - ndi.gaussian_filter(img01, 15)
    mad = np.median(np.abs(d - np.median(d))) * 1.4826
    bright = d > k * mad
    disc = lambda rad: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1,) * 2)  # noqa: E731
    seeds = cv2.morphologyEx(bright.astype(np.uint8), cv2.MORPH_OPEN, disc(r)).astype(bool)
    lab, n = ndi.label(bright, structure=np.ones((3, 3), int))
    keep = np.zeros(n + 1, bool)
    keep[np.unique(lab[seeds])] = True
    keep[0] = False
    return cv2.dilate(keep[lab].astype(np.uint8), disc(halo)).astype(bool)


SATO_SIGMAS = {"1-3": (1, 2, 3), "1-2": (1, 2)}
OPEN_LENGTHS = (17, 25)


def path_open_clean_response(img01: np.ndarray) -> dict:
    """Sato maps (two sigma sets), their line openings, and the particle mask."""
    out = {"particles": bright_particles(img01)}
    for name, sig in SATO_SIGMAS.items():
        r = ridge_response(img01, "sato", sig)
        out[name] = r
        for L in OPEN_LENGTHS:
            out[(name, L)] = line_open(r, L)
    return out


def path_open_clean_mask(img01: np.ndarray, resp: dict, thr: float, sig: str = "1-3",
                         length: int = 17, particles: bool = False,
                         erode: int = 0) -> np.ndarray:
    m = dir_close(resp[(sig, length)] > thr)
    if particles:
        m &= ~resp["particles"]
    if erode:
        m = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8),
                      iterations=erode).astype(bool)
    return m


def _rf_features(img01: np.ndarray, path_features: bool) -> np.ndarray:
    """wire_classifier's 17 features, plus the sato line openings (17, 25 px)."""
    F = feature_stack(img01)
    if path_features:
        s = sato_dark(img01)
        F = np.concatenate([F, np.stack([line_open(s, L) for L in OPEN_LENGTHS], -1)], -1)
    return F


def _train_rf(frame, tag: str, path_features: bool) -> callable:
    """wire_classifier's RF (same sampling and hyper-parameters), trained on ONE
    labelled frame so the other frame stays unseen."""
    path = CACHE / f"{tag}_trained_on_{frame.name}.joblib"
    if path.exists():
        clf = joblib.load(path)
    else:
        print(f"  training {tag} on {frame.name} only (~1-2 min) ...", flush=True)
        rng = np.random.default_rng(SEED)
        F = _rf_features(frame.img01, path_features)
        X = F.reshape(-1, F.shape[-1])
        y = frame.gt.ravel()
        pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
        k = N_TRAIN // 2
        idx = np.concatenate([rng.choice(pos, min(k, len(pos)), replace=False),
                              rng.choice(neg, min(k, len(neg)), replace=False)])
        clf = RandomForestClassifier(n_estimators=150, min_samples_leaf=25,
                                     random_state=0, n_jobs=-1)
        clf.fit(X[idx], y[idx])
        CACHE.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, path, compress=3)
        if path_features:
            names = FEATURE_NAMES + [f"open{L}" for L in OPEN_LENGTHS]
            top = sorted(zip(clf.feature_importances_, names), reverse=True)[:5]
            print("    top features: " + ", ".join(f"{n} {v:.2f}" for v, n in top), flush=True)
    if not path_features:
        return lambda img01: wire_probability(clf, img01)       # production code path

    def predict(img01: np.ndarray) -> np.ndarray:
        F = _rf_features(img01, True)
        return clf.predict_proba(F.reshape(-1, F.shape[-1]))[:, 1].reshape(img01.shape)
    return predict


def train_rf_v3(frame) -> callable:
    return _train_rf(frame, "rf_v3", path_features=False)


def train_rf_pathfeat(frame) -> callable:
    return _train_rf(frame, "rf_pathfeat", path_features=True)


SATO_V1 = Method("sato_dark_v1", response=sato_dark, author_thr=0.10,
                 note="Nikhil v1 reference")
SATO_V2 = Method("sato_dark_v2", response=sato_dark, post=dir_close, author_thr=0.14,
                 note="Nikhil v2 reference: + directional closing")
RF_V3 = Method("rf_v3", train=train_rf_v3, author_thr=PROD_THR,
               note="Nikhil v3 reference, trained on the other frame")
LINE_FIT = Method(
    "sato_line_fit", response=line_fit_response, make_mask=line_fit_mask,
    # hysteresis low. The review's grid stopped at 0.25 and so did every pick in
    # a first run here, so the search goes higher (180 combinations in total)
    thresholds=(0.12, 0.18, 0.25, 0.35, 0.5),
    params={"pol": ["bright", "both", "dark"], "high": [0.5, 0.7],
            "line_len": [10, 16, 24], "gap": [3, 5]},
    # the review's pick (tuned on frame 1, bright 0.25/0.5, len 25, gap 8 at 80k)
    author_thr=0.25, author_params={"pol": "bright", "high": 0.5, "line_len": 16, "gap": 5},
    measures_width=False, note="review E1b: Sato + 3-direction line fitting")

PATH_OPEN = Method("path_open", response=sato_dark, make_mask=path_open_mask,
                   params={"length": [11, 17, 25], "close": [False, True]},
                   note="guide candidate: straight path openings on sato dark")
STEGER = Method("steger", response=steger_response, make_mask=steger_mask,
                thresholds=(0.15, 0.2, 0.25, 0.3, 0.4),
                params={"high": [0.4, 0.5, 0.65], "min_len": [8, 15]},
                note="guide candidate: Steger centre lines + widths")
PATH_OPEN_CLEAN = Method(
    "path_open_clean", response=path_open_clean_response, make_mask=path_open_clean_mask,
    grid_map=lambda r: r["1-3"],
    params={"sig": ["1-3", "1-2"], "length": list(OPEN_LENGTHS),
            "particles": [False, True], "erode": [0, 1]},
    # one fixed setting for production (run_production.py): what both pixel-F1 tuning
    # directions agreed on, at the mean of their two thresholds (0.0557, 0.0575);
    # particle removal off (picked in one direction only, and it removes labelled wire)
    author_thr=0.0566, author_params={"sig": "1-2", "length": 25, "particles": False, "erode": 0},
    note="path openings + optional particle removal / thinning / narrower sato")
RF_PATHFEAT = Method("rf_pathfeat", train=train_rf_pathfeat,
                     note="RF v3 features + sato line openings as two extra inputs")
RF_PATHOPEN = Method("rf_pathopen", train=train_rf_v3, make_mask=path_open_mask,
                     params={"length": list(OPEN_LENGTHS), "close": [False, True]},
                     note="RF v3 probability map, then the path opening (+ closing)")

METHODS = [
    SATO_V1,
    SATO_V2,
    Method("sato_bright", response=sato_bright, note="guide candidate"),
    Method("sato_both", response=sato_both, note="guide candidate"),
    Method("frangi_dark", response=frangi_dark, note="guide candidate"),
    Method("local_std", response=local_std, note="guide candidate"),
    LINE_FIT,
    PATH_OPEN,
    PATH_OPEN_CLEAN,
    STEGER,
    RF_V3,
    RF_PATHFEAT,
    RF_PATHOPEN,
]
