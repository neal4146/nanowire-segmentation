"""Race every method in methods.METHODS through the shared scorer.

    venv/Scripts/python compare/run_all.py                  # all methods
    venv/Scripts/python compare/run_all.py sato_dark_v1 rf_v3   # selected

Runs the v1 sanity check first and stops if the scorer disagrees with the
README. A run of selected methods replaces only their rows; the other methods'
results from earlier runs are kept. Outputs go to results/compare/:
  results.csv     one row per method and protocol, test-frame means
  results.json    the same plus every cross-frame direction, GT references
                  and the extraction parameters (for reproducibility)
  overlays/       <method>_<protocol>_<test frame>.png
                  green = correct wire, orange = false alarm, blue = missed

The 'gt_mask' row runs the wire step on the hand-labelled mask itself. Pixel
and wire scores are 1 by definition; its wire_f1_hand (~0.56) is the most any
method can reach against the raw hand-drawn instances (see score.py).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from score import (OUT, _mean, extract_params, evaluate, gt_reference,  # noqa: E402
                   load_gt, sanity_check, score_mask)
import wire_extract                                   # noqa: E402  (path set by score)
from methods import METHODS                           # noqa: E402

COLUMNS = [
    "method", "protocol", "thr_1to2", "thr_2to1",
    "pixel_recall", "pixel_precision", "pixel_f1", "cov_pred_over_gt",
    "wire_recall", "wire_precision", "wire_f1", "wire_f1_hand",
    "count_abs_err_pct", "length_median_nm_abs_err_pct",
    "width_median_nm_abs_err_pct", "junctions_abs_err_pct", "junction_recall",
    "runtime_s", "params_1to2", "params_2to1",
]


def _params(kw: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in kw.items())


def _row(name: str, protocol: str, res: dict) -> dict:
    d = res["directions"]
    return {"method": name, "protocol": protocol,
            "thr_1to2": d[0]["thr"], "thr_2to1": d[1]["thr"],
            "params_1to2": _params(d[0]["params"]), "params_2to1": _params(d[1]["params"]),
            **res["mean"]}


def main() -> None:
    wanted = set(sys.argv[1:])
    frames = load_gt()
    sanity_check(frames)

    p = extract_params()
    refs = {f.name: gt_reference(f, p) for f in frames}
    for f in frames:
        r = refs[f.name]
        print(f"GT reference {f.name}: {r['count']} wires, median length "
              f"{r['length_median_nm']:.0f} nm, median width {r['width_median_nm']:.1f} nm, "
              f"{r['junctions']} junctions")

    detail = {}
    if wanted and (OUT / "results.json").exists():
        detail = json.load(open(OUT / "results.json"))["results"]   # keep other methods

    # ceiling: the hand-labelled mask through the same wire step
    ceil = [{"train": "-", "test": f.name, "thr": float("nan"), "params": {},
             **score_mask(f.gt, f, p, refs[f.name])} for f in frames]
    detail["gt_mask"] = {"ceiling": {"directions": ceil, "mean": _mean(ceil)}}

    for m in METHODS:
        if wanted and m.name not in wanted:
            continue
        print(f"\n== {m.name} ({m.note})", flush=True)
        res = evaluate(m, frames, p, refs)
        detail[m.name] = res
        for protocol, r in res.items():
            mm = r["mean"]
            print(f"  {protocol:7s} pixel F1 {mm['pixel_f1']:.3f} "
                  f"(R {mm['pixel_recall']:.3f} P {mm['pixel_precision']:.3f})  "
                  f"wire F1 {mm['wire_f1']:.3f}  count err {mm['count_abs_err_pct']:.0f}%  "
                  f"junction recall {mm['junction_recall']:.3f}", flush=True)
            for d in r["directions"]:
                print(f"           {d['train']} -> {d['test']}: thr {d['thr']:.4g} "
                      f"{_params(d['params'])}", flush=True)

    order = ["gt_mask"] + [m.name for m in METHODS]
    rows = [_row(name, protocol, r)
            for name in sorted(detail, key=lambda n: order.index(n) if n in order else len(order))
            for protocol, r in detail[name].items()]
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).reindex(columns=COLUMNS)
    df.to_csv(OUT / "results.csv", index=False, float_format="%.4f")

    refs_json = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                 for k, v in refs.items()}
    with open(OUT / "results.json", "w") as fh:
        json.dump({"gt_reference": refs_json, "results": detail,
                   "extract_params": vars(p)}, fh, indent=2,
                  default=wire_extract._jsonable)

    show = df[["method", "protocol", "pixel_f1", "cov_pred_over_gt", "wire_f1",
               "wire_f1_hand", "count_abs_err_pct",
               "length_median_nm_abs_err_pct", "junction_recall", "runtime_s"]]
    print("\n" + show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {OUT / 'results.csv'} and results.json")


if __name__ == "__main__":
    main()
