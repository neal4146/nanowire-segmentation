"""Contender E_RB's data: E's labels plus BACKGROUND on the flanks of wires.

    venv/Scripts/python bakeoff/build_rb_data.py      # ~1 min; validates, then writes train_ERB/, copypaste_ERB/
    bash bakeoff/dev_cv_variant.sh RB --data ERB --rot-deg 25

Why (pipeline audit, handover section 15/16): in the pseudo-labelled photos
the 1-6 px around a WIRE label are mostly IGNORE (the agreement band, where
A and the v3 forest disagree), so the network is never told that a wire's
flank or shadow is not wire. Its outputs are 11-13 px wide against 4.4-4.7 px
hand outlines, and on test the flank and shadow streaks are a main source of
false pieces.

Rule, per pixel of a pseudo-labelled region:
  IGNORE -> BACKGROUND if all hold
    - it is IGNORE in E's label AND in the agreement label (train/): only
      the A/v3 disagreement band, never E's faint-wire ignore or a hand fix
    - it is 3-6 px from the nearest WIRE pixel (the flank zone)
    - no line detector flags it: outside pseudo_labels.faint_candidates
      (cached in train_E/faint/), so a real neighbouring wire that only one
      method saw keeps its IGNORE
  Never changed: the hand-outlined photos, every hand-fixed quarter, WIRE.
The same rule runs on copypaste_E (with the source photo's agreement label
and faint mask), so pasted pairs and crossings keep their labels.

Validation (before anything is written): the rule is simulated inside the 5
complete-mode quarters (where every wire is hand-drawn) on their automatic
labels, and the new background pixels are compared with the hand lines.
The data is written, and train_ERB/VALIDATED created, only if at most
MAX_ON_WIRE of the new background lies within ON_WIRE_PX of a hand line.
"""
from __future__ import annotations

import csv
import json

import numpy as np
from scipy import ndimage as ndi
from skimage.draw import line as draw_line

from common import (BG, COPYPASTE_E, IGNORE, OUT, TRAIN, TRAIN_E, TRAIN_FIX_DIR, WIRE, read_json,
                    read_png, save_u8, test_photos, write_json)

TRAIN_ERB = OUT / "train_ERB"
COPYPASTE_ERB = OUT / "copypaste_ERB"
D_MIN, D_MAX = 3.0, 6.0
ON_WIRE_PX = 2.5
MAX_ON_WIRE = 0.05


def flank_to_bg(lab_e, base, faint, skip):
    d = ndi.distance_transform_edt(lab_e != WIRE)
    cand = ((lab_e == IGNORE) & (base == IGNORE) & ~faint & (d >= D_MIN) & (d <= D_MAX) & ~skip)
    out = lab_e.copy()
    out[cand] = BG
    return out, cand


def fixes_of(case: str) -> list[dict]:
    out = []
    for f in TRAIN_FIX_DIR.glob(f"{case}_*.json"):
        m = json.loads(f.read_text(encoding="utf-8"))
        if m["case"] == case:
            out.append({**m, "csv": f.with_suffix(".csv")})
    return out


def skip_mask(shape, fixes) -> np.ndarray:
    m = np.zeros(shape, bool)
    for f in fixes:
        r0, c0 = f["crop_origin_row_col"]
        h, w = f["crop_shape_row_col"]
        m[r0:r0 + h, c0:c0 + w] = True
    return m


def lines_raster(csv_path, shape, origin) -> np.ndarray:
    r0, c0 = origin
    wires: dict[int, list] = {}
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            if row["layer"] == "add wires":
                wires.setdefault(int(row["wire_id"]), []).append((float(row["row"]) + r0, float(row["col"]) + c0))
    m = np.zeros(shape, bool)
    for pts in wires.values():
        for a, b in zip(pts[:-1], pts[1:]):
            rr, cc = draw_line(int(round(a[0])), int(round(a[1])), int(round(b[0])), int(round(b[1])))
            ok = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
            m[rr[ok], cc[ok]] = True
    return m


def validate() -> dict:
    """Simulate the rule inside the complete-mode quarters, against the hand lines."""
    man = read_json(TRAIN_E / "manifest.json")
    n_new = n_on = n_near = 0
    per = []
    for c in man["cases"]:
        for f in fixes_of(c["case"]):
            if not f.get("complete"):
                continue
            base = read_png(TRAIN / "labels" / f"{c['case']}.png")
            faint = read_png(TRAIN_E / "faint" / f"{c['case']}.png") > 0
            sim = base.copy()
            sim[faint & (sim == BG)] = IGNORE               # E's automatic label before the hand fix
            q = skip_mask(base.shape, [f])
            _, cand = flank_to_bg(sim, base, faint, ~q)    # only inside this quarter
            dl = ndi.distance_transform_edt(~lines_raster(f["csv"], base.shape, f["crop_origin_row_col"]))
            new, on, near = int(cand.sum()), int((cand & (dl <= ON_WIRE_PX)).sum()), int((cand & (dl <= 4)).sum())
            n_new, n_on, n_near = n_new + new, n_on + on, n_near + near
            per.append({"quarter": f"{c['case']}_{f['quarter']}", "new_bg_px": new,
                        "on_wire": on / max(new, 1), "within_4px": near / max(new, 1)})
    res = {"new_bg_px": n_new, "on_wire_share": n_on / max(n_new, 1),
           "within_4px_share": n_near / max(n_new, 1), "per_quarter": per,
           "rule": f"IGNORE in E and agreement, {D_MIN}-{D_MAX} px from WIRE, not faint, not hand-fixed",
           "pass": n_on / max(n_new, 1) <= MAX_ON_WIRE}
    for r in per:
        print(f"  {r['quarter']:24s} new BG {r['new_bg_px']:6d} px; within {ON_WIRE_PX} px of a hand line "
              f"{r['on_wire']:.3f}, within 4 px {r['within_4px']:.3f}")
    print(f"validation: {n_new} new BG px, {res['on_wire_share']:.3f} on a hand-drawn wire "
          f"(limit {MAX_ON_WIRE}), {res['within_4px_share']:.3f} within 4 px -> {'PASS' if res['pass'] else 'FAIL'}")
    return res


def ring_stats(lab, wire) -> list[float]:
    d = ndi.distance_transform_edt(~wire)
    return [float((lab[(d > k - 1) & (d <= k)] == BG).mean()) for k in range(1, 7)]


def build(res) -> None:
    tests = set(test_photos())
    for src, dst, is_cp in ((TRAIN_E, TRAIN_ERB, False), (COPYPASTE_E, COPYPASTE_ERB, True)):
        man = read_json(src / "manifest.json")
        for sub in ("images", "labels"):
            (dst / sub).mkdir(parents=True, exist_ok=True)
        before, after, conv = np.zeros(6), np.zeros(6), 0
        n = 0
        for c in man["cases"]:
            if c.get("photo") in tests:
                raise SystemExit(f"{c['photo']} is a test photo")
            img_path = src / "images" / f"{c['case']}.png"
            (dst / "images" / img_path.name).write_bytes(img_path.read_bytes())
            lab = read_png(src / "labels" / f"{c['case']}.png")
            src_case = c.get("from", c["case"])
            if c["source"] == "hand" or not (TRAIN_E / "faint" / f"{src_case}.png").exists():
                save_u8(dst / "labels" / f"{c['case']}.png", lab)
                continue
            base = read_png(TRAIN / "labels" / f"{src_case}.png")
            faint = read_png(TRAIN_E / "faint" / f"{src_case}.png") > 0
            new, cand = flank_to_bg(lab, base, faint, skip_mask(lab.shape, fixes_of(src_case)))
            save_u8(dst / "labels" / f"{c['case']}.png", new)
            w = lab == WIRE
            before += ring_stats(lab, w)
            after += ring_stats(new, w)
            conv += int(cand.sum())
            n += 1
        write_json(dst / "manifest.json", {**man, "settings": {**man.get("settings", {}),
                   "built_from": src.name, "flank_background": res["rule"]}})
        print(f"{src.name} -> {dst.name}: {len(man['cases'])} cases, {conv} px IGNORE -> BG; "
              f"BG share at 1..6 px from WIRE: before {np.round(before / max(n, 1), 2).tolist()}, "
              f"after {np.round(after / max(n, 1), 2).tolist()}")


if __name__ == "__main__":
    res = validate()
    if not res["pass"]:
        raise SystemExit("validation failed: nothing written")
    build(res)
    write_json(TRAIN_ERB / "VALIDATED", res)
    print(f"wrote {TRAIN_ERB}/VALIDATED")
