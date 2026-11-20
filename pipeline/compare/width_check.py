"""How wide are the TiS3 wires really? Hand-picked wires on 200,000x photos vs.
the widths the pipelines report.

At 200k (1.35 nm/px) a wire is resolved: a lighter body between two dark edge
lines. For each hand-picked straight stretch (SEGMENTS, full-resolution pixel
coordinates picked on a grid), this script:
  1. samples cross-sections every 2 px along the middle 70% of the stretch,
  2. aligns them (cross-correlation, plus a linear drift fit, so a rough pick is fine),
  3. averages them, and on the mean profile measures
       edge width   distance between the two dark edge minima
       outer width  distance between the points where the profile climbs back
                    halfway from each edge minimum to the background level,
  4. repeats 2-3 on blocks of the stretch for a spread.
It writes width_check/profiles_<photo>.png (every measured stretch, its mean
profile and the marked edges) to check by eye, and width_check.csv.

Then it measures, on the 50k photos of the same growth run (SF251213M, 60 min),
the median width that Steger (compare/steger.py) reports per centre point, and
reads the v3 pipeline's wire_extract widths, for comparison.

    venv/Scripts/python compare/width_check.py
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from PIL import Image, ImageDraw
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ridge"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ridge_nanowire_images2 import load_frame_50k, read_pixel_size   # noqa: E402
from steger import steger_response                                   # noqa: E402

HIGHMAG = Path(r"C:/Users/Acer/Downloads/Materials_ThangPham/Materials_ThangPham/SF251213M TiS3 60 min")
OUT = ROOT / "results" / "compare" / "width_check"

# hand-picked straight, isolated stretches: photo -> [((x0, y0), (x1, y1)), ...]
SEGMENTS = {
    "SF251213M_200000×_40": [
        ((200, 284), (540, 284)), ((110, 447), (610, 447)), ((170, 578), (990, 578)),
        ((715, 135), (885, 425)), ((930, 460), (1160, 452)), ((1162, 494), (1011, 762)),
        ((95, 150), (160, 262)), ((410, 850), (455, 925)), ((655, 70), (668, 380)),
    ],
    "SF251213M_200000×_26": [
        ((100, 118), (650, 117)), ((440, 340), (530, 180)), ((780, 150), (930, 410)),
        ((30, 805), (700, 801)), ((740, 820), (920, 505)), ((1020, 830), (1140, 610)),
        ((1260, 650), (1420, 370)), ((1050, 535), (1290, 521)), ((1010, 868), (1500, 869)),
        ((500, 565), (570, 690)), ((600, 704), (740, 698)),
    ],
}
MAX_BLOCK_SD_NM = 5.0   # a stretch whose blocks disagree more than this is dropped
HALF = 40          # px sampled on each side of the picked line
STEP_S = 0.5       # px across
STEP_T = 2.0       # px along


def load_highmag(name: str) -> tuple[np.ndarray, float]:
    path = HIGHMAG / f"{name}.tif"
    px, im = read_pixel_size(path)
    a = np.array(im).astype(np.float64)[:1024]
    return ndi.gaussian_filter(a, 1.0), px * 1e9


def _sample(img, p0, p1, offsets, ss):
    """Cross-sections along p0->p1; offsets[i] shifts section i along the normal."""
    d = p1 - p0
    L = np.hypot(*d)
    u = d / L
    n = np.array([-u[1], u[0]])
    ts = np.arange(0.15 * L, 0.85 * L, STEP_T)
    if offsets is None:
        offsets = np.zeros(len(ts))
    centres = p0[None] + ts[:, None] * u[None] + offsets[:, None] * n[None]
    xs = centres[:, 0:1] + ss[None] * n[0]
    ys = centres[:, 1:2] + ss[None] * n[1]
    prof = ndi.map_coordinates(img, [ys.ravel(), xs.ravel()], order=1, mode="nearest")
    return ts, prof.reshape(xs.shape)


def aligned_profiles(img, p0, p1):
    ss = np.arange(-HALF, HALF + 1e-9, STEP_S)
    ts, P = _sample(img, p0, p1, None, ss)
    offsets = np.zeros(len(ts))
    for _ in range(3):
        tmpl = P.mean(0) - P.mean()
        shifts = []
        for row in P:
            c = np.correlate(row - row.mean(), tmpl, mode="same")
            mid = len(c) // 2
            win = slice(mid - int(15 / STEP_S), mid + int(15 / STEP_S) + 1)
            shifts.append((np.argmax(c[win]) - int(15 / STEP_S)) * STEP_S)
        shifts = np.asarray(shifts)
        # robust linear drift (a rough pick is slightly rotated)
        A = np.stack([np.ones_like(ts), ts], 1)
        coef = np.linalg.lstsq(A, shifts, rcond=None)[0]
        resid = shifts - A @ coef
        ok = np.abs(resid) <= max(2.0, 2.5 * np.median(np.abs(resid)))
        coef = np.linalg.lstsq(A[ok], shifts[ok], rcond=None)[0]
        offsets = offsets + A @ coef
        ts, P = _sample(img, p0, p1, offsets, ss)
    return ss, P


def measure_profile(ss, prof):
    """Edge width (between the two dark minima around the brighter body) and
    outer width (half-way back to background outside each minimum), in px."""
    bg = np.median(np.r_[prof[: int(8 / STEP_S)], prof[-int(8 / STEP_S):]])
    span = prof.max() - prof.min() + 1e-9
    mins, props = find_peaks(-prof, prominence=0.04 * span, distance=int(3 / STEP_S))
    best = None
    for i in range(len(mins)):
        for j in range(i + 1, len(mins)):
            a, b = mins[i], mins[j]
            w = (b - a) * STEP_S
            if not 6 <= w <= 50:
                continue
            body = prof[a:b + 1].max()
            if body <= max(prof[a], prof[b]):
                continue
            score = props["prominences"][i] + props["prominences"][j]
            if best is None or score > best[0]:
                best = (score, a, b)
    if best is None:
        return None
    _, a, b = best

    def outward(k, step):
        half = prof[k] + 0.5 * (bg - prof[k])
        while 0 < k < len(prof) - 1 and prof[k] < half:
            k += step
        return k

    lo, hi = outward(a, -1), outward(b, +1)
    return {"edge_px": (b - a) * STEP_S, "outer_px": (hi - lo) * STEP_S,
            "a": a, "b": b, "lo": lo, "hi": hi}


def measure_highmag() -> pd.DataFrame:
    rows = []
    OUT.mkdir(parents=True, exist_ok=True)
    for name, segs in SEGMENTS.items():
        img, nm = load_highmag(name)
        panels = []
        for k, (q0, q1) in enumerate(segs):
            p0, p1 = np.array(q0, float), np.array(q1, float)
            ss, P = aligned_profiles(img, p0, p1)
            m = measure_profile(ss, P.mean(0))
            blocks = [measure_profile(ss, B.mean(0)) for B in np.array_split(P, max(1, len(P) // 15))]
            bw = [b["edge_px"] * nm for b in blocks if b]
            row = {"photo": name, "wire": k, "nm_per_px": nm, "n_sections": len(P),
                   "edge_nm": m["edge_px"] * nm if m else np.nan,
                   "outer_nm": m["outer_px"] * nm if m else np.nan,
                   "edge_nm_block_sd": float(np.std(bw)) if len(bw) > 1 else np.nan,
                   "body": "bright" if m and P.mean(0)[(m["a"] + m["b"]) // 2] > np.median(P) + 0.5 * P.std() else "dark/grey"}
            rows.append(row)
            panels.append(_panel(P, m, f"{k}: edge {row['edge_nm']:.1f} nm, outer {row['outer_nm']:.1f} nm"))
        _save_panels(panels, OUT / f"profiles_{name.replace('×', 'x')}.png")
    return pd.DataFrame(rows)


def _panel(P, m, title):
    """Straightened strip (rows = along the wire) above its mean profile."""
    strip = P[:: max(1, len(P) // 60)]
    s = (strip - strip.min()) / (np.ptp(strip) + 1e-9)
    strip_img = Image.fromarray((s * 255).astype(np.uint8)).resize((322, 120))
    W, H = 322, 260
    canvas = Image.new("RGB", (W, H), "white")
    canvas.paste(strip_img.convert("RGB"), (0, 18))
    d = ImageDraw.Draw(canvas)
    d.text((2, 2), title, fill=(0, 0, 0))
    prof = P.mean(0)
    y = 250 - (prof - prof.min()) / (np.ptp(prof) + 1e-9) * 110
    x = np.linspace(0, W - 1, len(prof))
    d.line(list(zip(x, y)), fill=(0, 0, 0), width=2)
    if m:
        for key, col in (("a", (220, 0, 0)), ("b", (220, 0, 0)), ("lo", (0, 120, 220)), ("hi", (0, 120, 220))):
            xx = x[m[key]]
            d.line([(xx, 140), (xx, 255)], fill=col, width=1)
            d.line([(xx, 18), (xx, 138)], fill=col, width=1)
    return canvas


def _save_panels(panels, path):
    cols = 4
    rows = -(-len(panels) // cols)
    W, H = panels[0].size
    sheet = Image.new("RGB", (cols * W, rows * H), "white")
    for i, p in enumerate(panels):
        sheet.paste(p, ((i % cols) * W, (i // cols) * H))
    sheet.save(path)


def steger_widths_50k() -> pd.DataFrame:
    """Median Steger width per 50k photo of the same run (strong centre points)."""
    rows = []
    src = ROOT / "data" / "real" / "NanowireImages2"
    for path in sorted(src.glob("*.tif")):
        gray, meta = load_frame_50k(path)
        if not meta["scale_in_validated_band"]:
            continue
        r = steger_response(gray / 255.0)
        strong = r["strength"] > 0.3
        nm = 8.288e-6 / 1536 * 1e9
        rows.append({"photo": path.stem, "steger_width_nm": float(np.median(r["width"][strong])) * nm,
                     "steger_points": int(strong.sum())})
    return pd.DataFrame(rows)


def pipeline_widths() -> pd.DataFrame:
    """wire_extract's median per-wire width (v3 masks) for the same 50k photos."""
    rows = []
    for summ in sorted((ROOT / "results" / "wire_extract").glob("SF251213M_*/summary.json")):
        s = json.loads(summ.read_text())
        rows.append({"photo": summ.parent.name,
                     "v3_width_nm": s["wires_all"]["width_mean_nm"].get("median")})
    return pd.DataFrame(rows)


def main() -> None:
    hm = measure_highmag()
    hm.to_csv(OUT / "width_check.csv", index=False, float_format="%.2f")
    print(hm.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    good = hm.dropna(subset=["edge_nm"])
    good = good[~(good.edge_nm_block_sd > MAX_BLOCK_SD_NM)]   # e.g. a particle on the wire
    print(f"\n200k hand-picked wires: n = {len(good)} of {len(hm)} "
          f"(dropped: block spread > {MAX_BLOCK_SD_NM} nm)")
    print(f"  edge width  median {good.edge_nm.median():.1f} nm  (IQR {good.edge_nm.quantile(.25):.1f}"
          f"-{good.edge_nm.quantile(.75):.1f})")
    print(f"  outer width median {good.outer_nm.median():.1f} nm  (IQR {good.outer_nm.quantile(.25):.1f}"
          f"-{good.outer_nm.quantile(.75):.1f})")

    print(f"  by body: " + ", ".join(f"{b} {g.edge_nm.median():.1f} / {g.outer_nm.median():.1f} nm "
                                    f"(n={len(g)})" for b, g in good.groupby("body")))
    st = steger_widths_50k()
    pw = pipeline_widths()
    comp = st.merge(pw, on="photo", how="left")
    comp.to_csv(OUT / "width_50k_methods.csv", index=False, float_format="%.2f")
    print("\n50k photos, same run:")
    print(comp.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print(f"  Steger median {comp.steger_width_nm.median():.1f} nm | "
          f"v3 wire_extract median {comp.v3_width_nm.median():.1f} nm")


if __name__ == "__main__":
    main()
