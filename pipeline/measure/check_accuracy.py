"""How close measure_wires.py's numbers come to hand labels.

    venv/Scripts/python measure/check_accuracy.py                          # --method unet (default)
    venv/Scripts/python measure/check_accuracy.py --method path_open_clean

Runs the tool end to end on the two hand-labelled 80,000x photos
(data/real/OneDrive_1_7-17-2026/TiS3_80000×_{1,2}.tif). It also traces the
hand-drawn wires with the same tracing step (wire_extract), then compares the
per-photo numbers.

This is a best case for both methods: path_open_clean's one setting was tuned on
these same two photos, and the U-Net (--method unet) trained on them. The fair
check is the line-labelled test quarters in bakeoff/ (bakeoff/README.md: wire F1,
count and length errors on 16 held-out quarters). The hand reference here also
follows wire_extract's convention (zig-zag arms joined into one bent wire, wires
broken where they cross), which --method unet does not, so count, length and
share_other do not compare like for like for it.
Writes results/measure/accuracy.csv (path_open_clean) or accuracy_unet.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure_wires as mw                         # noqa: E402  (also sets up the paths)
import wire_extract                                # noqa: E402
from gt_labels import gt_frames                    # noqa: E402
from score import load_gt                          # noqa: E402

MEASURES = ["wire_count", "wires_per_um2", "wire_length_um_per_um2", "coverage_pct",
            "median_length_nm", "median_width_nm", "share_0deg", "share_60deg",
            "share_123.5deg", "share_other", "junction_count"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--method", choices=("unet", "path_open_clean"), default="unet")
    args = ap.parse_args()
    p = mw.wire_params()
    raw = {fr["name"]: fr["raw"] for fr in gt_frames()}
    opts = {"method": args.method, "masks": None, "all_scales": False,
            "include_tilted": False, "overlays": False, "out": ""}
    if args.method == "unet":
        mw.unet_setup(opts, mw.MODEL, mw.VERIFIER, mw.ROOT / "results" / "measure" / "probs_accuracy")
        for parent in sorted({path.parent for path in raw.values()}):
            mw.ensure_probs([q for q in raw.values() if q.parent == parent], parent, opts)
    rows = []
    for frame in load_gt():
        tool, _ = mw.measure_photo(raw[frame.name], raw[frame.name].parent, opts)
        if tool["status"] != "measured":
            raise SystemExit(f"{frame.name}: {tool['status']}")
        records, summary, _ = wire_extract.extract_from_mask(frame.gt, frame.nm_per_px, p)
        hand = mw.image_stats(records, summary, frame.gt, p)
        for m in MEASURES:
            h, t = hand[m], tool[m]
            share = m.startswith("share_")
            rows.append({"photo": frame.name, "measure": m, "hand_labels": h, "tool": t,
                         "error": round(100 * (t - h), 1) if share else
                         round(100 * (t - h) / h, 1) if h else float("nan"),
                         "error_unit": "percentage points" if share else "%"})
    df = pd.DataFrame(rows)
    out = mw.ROOT / "results" / "measure" / ("accuracy.csv" if args.method == "path_open_clean"
                                             else f"accuracy_{args.method}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    wide = df.pivot(index="measure", columns="photo", values=["hand_labels", "tool", "error"])
    print(f"{'measure':24s}" + "".join(f"{n:>30s}" for n in sorted(df.photo.unique()))
          + f"{'mean |error|':>14s}")
    for m in MEASURES:
        cells = "".join(f"{wide.loc[m, ('hand_labels', n)]:>10.3g} -> {wide.loc[m, ('tool', n)]:<8.3g}"
                        f"({wide.loc[m, ('error', n)]:+.0f})".rjust(30)
                        for n in sorted(df.photo.unique()))
        unit = "pp" if m.startswith("share_") else "%"
        print(f"{m:24s}{cells}{df[df.measure == m].error.abs().mean():>11.0f} {unit}")
    print(f"\n(hand -> tool, error in % or percentage points)\nwrote {out}")


if __name__ == "__main__":
    main()
