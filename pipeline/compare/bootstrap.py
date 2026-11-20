"""Bootstrap confidence intervals for the race in results/compare/results.json.

Two labelled frames are too few to tell a real win from luck by eye. This cuts
each test frame into 160 x 160 px tiles (24 per frame), resamples the tiles with
replacement (within each frame, the same draw for every method, so comparisons
are paired) and recomputes pooled pixel F1 and wire F1 2000 times.

Each (method, protocol) mask is rebuilt from the settings saved in
results.json, so run compare/run_all.py first:

    venv/Scripts/python compare/bootstrap.py

Writes results/compare/bootstrap.csv:
  pixel_f1, wire_f1                pooled over both test frames (micro average,
                                   so slightly different from results.csv's mean)
  *_lo, *_hi                       95% interval
  d_*_vs_<ref>, p_better_vs_<ref>  paired difference to each reference row
                                   (sato_dark_v1/author, rf_v3/author) and the
                                   share of resamples in which this row is ahead

Wires are assigned to the tile holding their centreline's centre of mass, so a
long wire counts in one tile. Tiles are not fully independent (wires cross tile
edges), so treat the intervals as approximate, not exact.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))

from score import (OUT, WIRE_COVER, WIRE_TOL_PX, chains_to_instances,  # noqa: E402
                   extract_params, gt_reference, load_gt)
import wire_extract                                                     # noqa: E402
from methods import METHODS                                             # noqa: E402
from wire_metrics import centreline_scores                              # noqa: E402

TILE = 160
N_BOOT = 2000
SEED = 0
REFS = [("sato_dark_v1", "author"), ("rf_v3", "author")]


def _tile_ids(rc: np.ndarray, shape: tuple) -> np.ndarray:
    nx = -(-shape[1] // TILE)
    r = np.clip(rc[:, 0], 0, shape[0] - 1).astype(int) // TILE
    c = np.clip(rc[:, 1], 0, shape[1] - 1).astype(int) // TILE
    return r * nx + c


def tile_counts(mask: np.ndarray, frame, p, ref: dict) -> dict:
    """Per-tile pixel and whole-wire counts of one test mask."""
    shape = mask.shape
    nt = (-(-shape[0] // TILE)) * (-(-shape[1] // TILE))
    rc = np.indices(shape).reshape(2, -1).T
    tid = _tile_ids(rc, shape).reshape(shape)
    c = {"pix_tp": np.bincount(tid[mask & frame.gt], minlength=nt),
         "pix_pred": np.bincount(tid[mask], minlength=nt),
         "pix_gt": np.bincount(tid[frame.gt], minlength=nt)}

    records, _, _ = wire_extract.extract_from_mask(mask, frame.nm_per_px, p)
    pred = chains_to_instances(records, shape)
    gt = ref["_instances"]
    w = centreline_scores(gt, pred, tol=WIRE_TOL_PX, cover=WIRE_COVER, return_matches=True)
    sides = (("gt", gt, w["gt_ids"], {g for g, _ in w["matches"]}),
             ("pred", pred, w["pred_ids"], {q for _, q in w["matches"]}))
    for side, inst, ids, matched in sides:
        if len(ids):
            cen = np.asarray(ndi.center_of_mass(inst > 0, inst, ids), float).reshape(-1, 2)
            t = _tile_ids(cen, shape)
            hit = np.array([i in matched for i in ids])
        else:
            t, hit = np.zeros(0, int), np.zeros(0, bool)
        c[f"wire_{side}"] = np.bincount(t, minlength=nt)
        c[f"wire_{side}_tp"] = np.bincount(t[hit], minlength=nt)
    return c


def pooled(per: dict, draw: dict | None = None) -> tuple[float, float]:
    """(pixel F1, wire F1) from tile counts summed over both test frames."""
    tot = {}
    for name, cnt in per.items():
        idx = draw[name] if draw is not None else slice(None)
        for k, v in cnt.items():
            tot[k] = tot.get(k, 0) + v[idx].sum()
    pix = 2 * tot["pix_tp"] / max(tot["pix_pred"] + tot["pix_gt"], 1)
    R = tot["wire_gt_tp"] / max(tot["wire_gt"], 1)
    P = tot["wire_pred_tp"] / max(tot["wire_pred"], 1)
    return pix, 2 * P * R / max(P + R, 1e-9)


def collect(results: dict, frames: list, p, refs: dict) -> dict:
    """Rebuild every saved test mask and count it per tile."""
    fmap = {f.name: f for f in frames}
    methods = {m.name: m for m in METHODS}
    counts = {}
    for mname, protos in results.items():
        if mname not in methods:
            continue                                  # gt_mask ceiling row
        m = methods[mname]
        fitted, resp = {}, {}
        for proto, r in protos.items():
            per = {}
            for d in r["directions"]:
                train, test = fmap[d["train"]], fmap[d["test"]]
                if train.name not in fitted:
                    fitted[train.name] = m.fit(train)
                if (train.name, test.name) not in resp:
                    resp[(train.name, test.name)] = fitted[train.name](test.img01)
                mask = m.mask(test.img01, resp[(train.name, test.name)], d["thr"], d["params"])
                per[test.name] = tile_counts(mask, test, p, refs[test.name])
            counts[(mname, proto)] = per
            print(f"  counted {mname}/{proto}", flush=True)
    return counts


def main() -> None:
    saved = json.load(open(OUT / "results.json"))["results"]
    frames = load_gt()
    p = extract_params()
    refs = {f.name: gt_reference(f, p) for f in frames}
    counts = collect(saved, frames, p, refs)

    rng = np.random.default_rng(SEED)
    ntiles = {f.name: len(next(iter(counts.values()))[f.name]["pix_tp"]) for f in frames}
    draws = [{n: rng.integers(0, k, k) for n, k in ntiles.items()} for _ in range(N_BOOT)]
    boot = {key: np.array([pooled(per, d) for d in draws]) for key, per in counts.items()}

    rows = []
    for key, per in counts.items():
        pix, wf = pooled(per)
        b = boot[key]
        row = {"method": key[0], "protocol": key[1],
               "pixel_f1": pix, "pixel_f1_lo": np.percentile(b[:, 0], 2.5),
               "pixel_f1_hi": np.percentile(b[:, 0], 97.5),
               "wire_f1": wf, "wire_f1_lo": np.percentile(b[:, 1], 2.5),
               "wire_f1_hi": np.percentile(b[:, 1], 97.5)}
        for ref in REFS:
            if ref not in boot or ref == key:
                continue
            tag = ref[0].replace("sato_dark_", "")
            for j, metric in enumerate(("pixel", "wire")):
                d = b[:, j] - boot[ref][:, j]
                row[f"d_{metric}_vs_{tag}"] = float(np.mean(d))
                row[f"d_{metric}_vs_{tag}_lo"] = float(np.percentile(d, 2.5))
                row[f"d_{metric}_vs_{tag}_hi"] = float(np.percentile(d, 97.5))
                row[f"p_better_{metric}_vs_{tag}"] = float(np.mean(d > 0))
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("wire_f1", ascending=False)
    df.to_csv(OUT / "bootstrap.csv", index=False, float_format="%.4f")

    def ci(r, k):
        return f"{r[k]:.3f} [{r[k + '_lo']:.3f}, {r[k + '_hi']:.3f}]"

    print(f"\n{'method/protocol':26s} {'pixel F1 [95%]':>24s} {'wire F1 [95%]':>24s} "
          f"{'P(wire > v1)':>13s} {'P(wire > rf)':>13s}")
    for _, r in df.iterrows():
        pv1 = r.get("p_better_wire_vs_v1")
        prf = r.get("p_better_wire_vs_rf_v3")
        print(f"{r['method'] + '/' + r['protocol']:26s} {ci(r, 'pixel_f1'):>24s} "
              f"{ci(r, 'wire_f1'):>24s} "
              f"{'ref' if pd.isna(pv1) else f'{pv1:.2f}':>13s} "
              f"{'ref' if pd.isna(prf) else f'{prf:.2f}':>13s}")
    print(f"\nwrote {OUT / 'bootstrap.csv'}")


if __name__ == "__main__":
    main()
