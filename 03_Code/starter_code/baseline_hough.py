"""Experiment E1b (TiS3): improved classical baseline = ridge filter + direction-constrained Hough line
fitting + collinear merging. Tuned on TiS3 image 1, tested on TiS3 image 2 (and vice versa).
Run from the Materials folder:  python 03_Code/starter_code/baseline_hough.py
"""
import os, sys, json, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LABEL_SOURCES, load_gray, normalize
from metrics import centreline_scores
from baseline_ridge import ridge_response, ridge_hough_instances, gt_for

HERE = os.path.dirname(os.path.abspath(__file__))
keys = [k for k in LABEL_SOURCES if k.startswith("TiS3")]
imgs = {k: normalize(load_gray(os.path.join(HERE, "..", "..", LABEL_SOURCES[k]))) for k in keys}
gts = {k: gt_for(k, imgs[k].shape) for k in keys}
grid = dict(pol=["bright", "both"], low=[0.12, 0.18, 0.25], high=[0.35, 0.5], line_len=[15, 25], gap=[4, 8])
results = {}
for tune, test in [(keys[0], keys[1]), (keys[1], keys[0])]:
    best = None
    Rcache = {p: ridge_response(imgs[tune], (1, 2, 3), p) for p in grid["pol"]}
    for pol, lo, hi, ll, gp in itertools.product(*grid.values()):
        s = centreline_scores(gts[tune], ridge_hough_instances(imgs[tune], Rcache[pol], lo, hi, ll, gp))
        if best is None or s["wire_f1"] > best[0]:
            best = (s["wire_f1"], pol, lo, hi, ll, gp)
    f, pol, lo, hi, ll, gp = best
    s = centreline_scores(gts[test], ridge_hough_instances(imgs[test], ridge_response(imgs[test], (1, 2, 3), pol), lo, hi, ll, gp))
    s = {a: (round(b, 3) if isinstance(b, float) else b) for a, b in s.items()}
    results["tune=%s -> test=%s" % (tune, test)] = dict(params=dict(polarity=pol, low=lo, high=hi, line_len=ll, gap=gp),
                                                       tune_wire_f1=round(f, 3), test=s)
    print("tuned on", tune, "F1=%.3f" % f, "params", (pol, lo, hi, ll, gp), "-> test", test, s, flush=True)
json.dump(results, open(os.path.join(HERE, "outputs", "E1b_ridge_hough_TiS3.json"), "w"), indent=1)
