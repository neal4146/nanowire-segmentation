"""Hard cases by copy-paste: real wires pasted as close parallel pairs and as
crossings onto the pseudo-labelled training photos (contender C).

    venv/Scripts/python bakeoff/copy_paste.py               # one pasted copy per photo (~1 min)
    venv/Scripts/python bakeoff/copy_paste.py --copies 2 --events 40

The agreement pseudo-labels ignore exactly the places where path_open_clean
and the forest disagree, so a network trained on them never sees a target for
the two cases they get wrong. This makes those cases with exact labels:
  pair      two wires of the same direction side by side, --gap px apart edge
            to edge (default 2-8 px = 11-43 nm); the gap is labelled background
  crossing  a wire of another direction laid across the first; both stay whole
Each pasted wire is one hand-drawn instance from the two hand-labelled photos:
its filled outline plus a MARGIN px rim that carries the dark edge contrast.
So the pasted pixels are real SEM pixels and the label is the hand label.
Wires are never rotated (they keep their growth direction and its shading);
a parallel partner is another wire of the same direction, or the same wire
turned 180 degrees. Brightness and contrast are matched to the destination's
background.

Run bakeoff/pseudo_labels.py first. Writes results/bakeoff/copypaste/
images/CP_<case>_<k>.png, labels/ (0 background, 1 wire, 2 ignore, like
train/), previews/ and manifest.json.

Contender E (bakeoff/build_e_data.py) also pastes FAINT wires (--faint N):
a sprite's own contrast scaled down to 2-3.5 noise units and added onto the
destination, 40% of them cut to 6-20 px short rods (+1 noise unit, since a
short faint rod needs more contrast to be seen at all). They are labelled
wire, which no pseudo-label can do: the agreement labels call faint wires
background because neither filter sees them.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import scipy.ndimage as ndi

from common import (BG, COPYPASTE, IGNORE, TRAIN, WIRE, read_json, read_png, save_u8,
                    write_json)

FAMILY_ANGLES = (0.0, 58.0, 122.0)   # growth directions, degrees, y axis up (as methods.py)
MARGIN = 2           # px of rim pasted around each hand outline
JOIN_CUT = 3         # px of a wire cut off where it meets another wire
CONTEXT = 10         # px around an instance used to measure its background
MAX_SIDE = 220       # px: longest sprite that is used
MAX_WIDTH = 12       # px (65 nm): wider hand outlines are bright rods, not the thin wires
CLEAR = 0.85         # share of a pasted footprint that must land on agreed background
P_PAIR = 0.6         # share of events that are parallel pairs (the rest are crossings)


@dataclass
class Sprite:
    img: np.ndarray      # [0, 1] pixels of the footprint's bounding box
    body: np.ndarray     # filled hand outline
    foot: np.ndarray     # body grown by MARGIN: the pixels that get pasted
    family: int          # index into FAMILY_ANGLES
    width: float         # px, body area / length
    bg_med: float        # background level and spread around the wire in its photo
    bg_mad: float

    def flipped(self) -> "Sprite":
        f = lambda a: a[::-1, ::-1].copy()                      # noqa: E731
        return Sprite(f(self.img), f(self.body), f(self.foot), self.family,
                      self.width, self.bg_med, self.bg_mad)


def _mad(v: np.ndarray) -> float:
    return float(np.median(np.abs(v - np.median(v))) * 1.4826) if v.size else 0.0


def _axis(body: np.ndarray) -> tuple[float, float, float]:
    """(direction in degrees, y up, in [0, 180); length along it; extent across it), px."""
    r, c = np.nonzero(body)
    pts = np.stack([c - c.mean(), -(r - r.mean())], 1)        # x right, y up
    w, v = np.linalg.eigh(np.cov(pts.T))
    d, n = v[:, 1], v[:, 0]
    ang = np.degrees(np.arctan2(d[1], d[0])) % 180
    proj, side = pts @ d, pts @ n
    lo, hi = np.percentile(side, [2, 98])
    return float(ang), float(proj.max() - proj.min() + 1), float(hi - lo + 1)


def make_sprites(frames) -> list[Sprite]:
    """Every clean, straight hand-drawn wire of the hand-labelled photos."""
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * MARGIN + 1,) * 2)
    join = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * JOIN_CUT + 1,) * 2)
    out = []
    for fr in frames:
        H, W = fr.gt.shape
        for i, sl in enumerate(ndi.find_objects(fr.instances), start=1):
            if sl is None:
                continue
            r0, r1 = sl[0].start - CONTEXT, sl[0].stop + CONTEXT
            c0, c1 = sl[1].start - CONTEXT, sl[1].stop + CONTEXT
            if r0 < 0 or c0 < 0 or r1 > H or c1 > W:
                continue                                       # touches the photo edge
            win = (slice(r0, r1), slice(c0, c1))
            gt = fr.gt[win]
            # almost every wire meets another at a corner or crossing: cut the
            # last JOIN_CUT px before any other wire off, so the sprite ends cleanly
            body = fr.instances[win] == i
            body &= ~cv2.dilate((gt & ~body).astype(np.uint8), join).astype(bool)
            lab, n = ndi.label(body)
            if n == 0:
                continue
            body = lab == 1 + int(np.argmax(np.bincount(lab.ravel())[1:]))
            if body.sum() < 40:
                continue
            ang, length, across = _axis(body)
            diff = [min(abs(ang - a), 180 - abs(ang - a)) for a in FAMILY_ANGLES]
            fam = int(np.argmin(diff))
            width = body.sum() / length
            if diff[fam] > 10 or length < 20 or length / width < 4:
                continue                                       # not a straight wire piece
            if across > MAX_WIDTH or across > 1.6 * width:
                continue                # a thick bright rod, or an outline that was never filled
            # the rim never includes other wires or the cut-off piece
            foot = cv2.dilate(body.astype(np.uint8), grow).astype(bool) \
                & ~cv2.dilate((gt & ~body).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            bg = fr.img01[win][~cv2.dilate(gt.astype(np.uint8), grow).astype(bool)]
            rr, cc = np.nonzero(foot)
            box = (slice(rr.min(), rr.max() + 1), slice(cc.min(), cc.max() + 1))
            if max(np.ptp(rr), np.ptp(cc)) + 1 > MAX_SIDE or bg.size < 50:
                continue
            out.append(Sprite(fr.img01[win][box].astype(np.float32), body[box], foot[box],
                              fam, float(width), float(np.median(bg)), _mad(bg)))
    return out


def paste(img, lab, protect, s: Sprite, r0: int, c0: int) -> None:
    """Paste sprite s with its top-left at (r0, c0); updates img, lab, protect."""
    h, w = s.body.shape
    reg = (slice(r0, r0 + h), slice(c0, c0 + w))
    dst, L, prot = img[reg], lab[reg], protect[reg]
    bg = dst[(L == BG) & ~s.foot]
    if bg.size < 30:
        bg = dst[~s.foot] if (~s.foot).sum() >= 30 else dst.ravel()
    scale = np.clip(_mad(bg) / max(s.bg_mad, 1e-4), 0.6, 1.6)
    new = (s.img - s.bg_med) * scale + np.median(bg)

    alpha = np.clip(ndi.gaussian_filter(s.foot.astype(np.float32), 0.7) * 1.3, 0, 1)
    alpha[s.body] = 1.0
    alpha[~s.body & prot] = 0.0          # never paste a rim over a wire pasted before
    img[reg] = np.clip(alpha * new + (1 - alpha) * dst, 0, 1)

    rim = ~s.body & (alpha > 0.3)
    L[rim & (L == WIRE) & ~prot] = IGNORE            # a destination wire partly covered
    L[rim & (alpha >= 0.9) & (L != WIRE)] = BG       # now the pasted wire's own background
    L[s.body] = WIRE
    prot |= s.body


def _place(s: Sprite, centre_rc, shape):
    """Top-left that puts the sprite's body centroid at centre_rc, or None if outside."""
    r, c = np.nonzero(s.body)
    r0 = int(round(centre_rc[0] - r.mean()))
    c0 = int(round(centre_rc[1] - c.mean()))
    h, w = s.body.shape
    if r0 < 0 or c0 < 0 or r0 + h > shape[0] or c0 + w > shape[1]:
        return None
    return r0, c0


def _free(s: Sprite, at, protect) -> bool:
    """True if the body lands at least 1 px clear of every wire pasted so far."""
    h, w = s.body.shape
    reg = protect[at[0]:at[0] + h, at[1]:at[1] + w]
    near = cv2.dilate(reg.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return not (near & s.body).any()


def _clear(s: Sprite, at, lab, skip=None) -> bool:
    """True if the footprint lands (mostly) on agreed background, not on crystals
    or ignored clutter. skip: pixels of the photo not to count (a wire being crossed)."""
    h, w = s.body.shape
    reg = (slice(at[0], at[0] + h), slice(at[1], at[1] + w))
    f = s.foot & ~skip[reg] if skip is not None else s.foot
    return f.any() and (lab[reg][f] == BG).mean() >= CLEAR


def add_event(img, lab, protect, sprites, by_fam, rng, gap_range) -> str | None:
    a = sprites[rng.integers(len(sprites))]
    H, W = img.shape
    for _ in range(30):
        at_a = _place(a, (rng.uniform(0, H), rng.uniform(0, W)), img.shape)
        if at_a and _free(a, at_a, protect) and _clear(a, at_a, lab):
            break
    else:
        return None
    rr, cc = np.nonzero(a.body)
    ca = (at_a[0] + rr.mean(), at_a[1] + cc.mean())
    th = np.radians(FAMILY_ANGLES[a.family])
    along = np.array([-np.sin(th), np.cos(th)])                  # (row, col)
    perp = np.array([np.cos(th), np.sin(th)])

    if rng.random() < P_PAIR:
        kind = "pair"
        same = [s for s in by_fam[a.family] if s is not a]
        b = same[rng.integers(len(same))] if same and rng.random() < 0.7 else a.flipped()
        if rng.random() < 0.5:
            b = b.flipped()
        gap = rng.uniform(*gap_range)
        d = (a.width + b.width) / 2 + gap
        length = a.body.sum() / a.width
        centre = np.array(ca) + perp * d * rng.choice([-1, 1]) \
            + along * rng.uniform(-0.3, 0.3) * length
    else:
        kind = "crossing"
        other = [s for f, ss in by_fam.items() if f != a.family for s in ss]
        if not other:
            return None
        b = other[rng.integers(len(other))]
        k = rng.integers(len(rr))                                   # a point on wire a
        centre = np.array([at_a[0] + rr[k], at_a[1] + cc[k]])
    at_b = _place(b, centre, img.shape)
    if at_b is None or (kind == "pair" and not _free(b, at_b, protect)):
        return None
    under_a = np.zeros(lab.shape, bool)
    under_a[at_a[0]:at_a[0] + a.foot.shape[0], at_a[1]:at_a[1] + a.foot.shape[1]] = a.foot
    if not _clear(b, at_b, lab, skip=under_a):
        return None
    if kind == "pair":                       # the partner must not touch wire a either
        trial = protect.copy()
        trial[at_a[0]:at_a[0] + a.body.shape[0], at_a[1]:at_a[1] + a.body.shape[1]] |= a.body
        if not _free(b, at_b, trial):
            return None
    paste(img, lab, protect, a, *at_a)
    paste(img, lab, protect, b, *at_b)
    return kind


def shortened(s: Sprite, length: float, rng) -> Sprite | None:
    """A `length` px piece of sprite s (short rods, contender E)."""
    r, c = np.nonzero(s.body)
    pts = np.stack([c - c.mean(), -(r - r.mean())], 1)
    _, v = np.linalg.eigh(np.cov(pts.T))
    proj = pts @ v[:, 1]
    lo, hi = proj.min(), proj.max()
    if hi - lo < length + 2:
        return None
    a = rng.uniform(lo, hi - length)
    body = np.zeros_like(s.body)
    sel = (proj >= a) & (proj <= a + length)
    body[r[sel], c[sel]] = True
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * MARGIN + 1,) * 2)
    foot = cv2.dilate(body.astype(np.uint8), grow).astype(bool) & s.foot
    rr, cc = np.nonzero(foot)
    box = (slice(rr.min(), rr.max() + 1), slice(cc.min(), cc.max() + 1))
    return Sprite(s.img[box], body[box], foot[box], s.family, s.width, s.bg_med, s.bg_mad)


def contrast(s: Sprite) -> float:
    """Typical |wire - background| of a sprite, in its own noise units."""
    return float(np.median(np.abs(s.img[s.body] - s.bg_med)) / max(s.bg_mad, 1e-4))


def paste_faint(img, lab, protect, s: Sprite, r0: int, c0: int, snr: float) -> None:
    """Paste sprite s at reduced contrast: its deviation from its own background,
    scaled so the wire sits `snr` noise units from the destination background,
    is ADDED to the destination (which keeps the destination's own noise)."""
    h, w = s.body.shape
    reg = (slice(r0, r0 + h), slice(c0, c0 + w))
    dst, L, prot = img[reg], lab[reg], protect[reg]
    bg = dst[(L == BG) & ~s.foot]
    if bg.size < 30:
        bg = dst[~s.foot] if (~s.foot).sum() >= 30 else dst.ravel()
    # share of the original contrast kept; a hand-drawn wire that is already
    # fainter than asked is pasted as it is (a person saw it in its own photo)
    f = min(1.0, snr / max(contrast(s), 1e-3))
    dev = (s.img - s.bg_med) * (_mad(bg) / max(s.bg_mad, 1e-4)) * f
    alpha = np.clip(ndi.gaussian_filter(s.foot.astype(np.float32), 0.7) * 1.3, 0, 1)
    alpha[s.body] = 1.0
    alpha[~s.body & prot] = 0.0
    img[reg] = np.clip(dst + alpha * dev, 0, 1)
    L[~s.body & (alpha > 0.3) & (L == WIRE) & ~prot] = IGNORE
    # at low contrast the outline's faded edge is hard to see: label the outline
    # thinned by 1 px as wire and that edge as ignore
    core = ndi.binary_erosion(s.body) if s.width >= 5 else s.body
    if core.sum() < 0.3 * s.body.sum():
        core = s.body
    L[s.body & ~core] = IGNORE
    L[core] = WIRE
    prot |= s.body


def add_faint(img, lab, protect, sprites, rng, p_short: float, snr_range) -> str | None:
    """One faint wire (contender E): a whole sprite, or with p_short a 6-20 px
    piece of one (short rods need a little more contrast to be seen at all)."""
    s = sprites[rng.integers(len(sprites))]
    short = rng.random() < p_short
    if short:
        s = shortened(s, rng.uniform(6, 20), rng)
        if s is None:
            return None
    lo, hi = snr_range
    snr = rng.uniform(lo + (1.0 if short else 0.0), hi + (0.5 if short else 0.0))
    H, W = img.shape
    for _ in range(30):
        at = _place(s, (rng.uniform(0, H), rng.uniform(0, W)), img.shape)
        if at and _free(s, at, protect) and _clear(s, at, lab):
            paste_faint(img, lab, protect, s, *at, snr)
            return "faint_short" if short else "faint"
    return None


def preview(img01: np.ndarray, lab: np.ndarray, protect: np.ndarray) -> np.ndarray:
    """Gray; green = wire label, cyan = pasted wire, magenta tint = ignored."""
    g = (img01 * 255).astype(np.float32)
    rgb = np.stack([g] * 3, axis=2)
    ign = lab == IGNORE
    rgb[ign] = 0.55 * rgb[ign] + 0.45 * np.array([255, 0, 255])
    rgb[(lab == WIRE) & ~protect] = (40, 210, 40)
    rgb[protect] = (0, 220, 255)
    return rgb.astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--copies", type=int, default=1, help="pasted copies per training photo")
    ap.add_argument("--events", type=int, default=30, help="pair/crossing attempts per copy")
    ap.add_argument("--gap", type=float, nargs=2, default=(2.0, 8.0), help="pair gap range, px")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--faint", type=int, default=0,
                    help="faint-wire attempts per copy (contender E; 0 keeps C's data unchanged)")
    ap.add_argument("--p-short", type=float, default=0.4, help="share of faint wires cut to 6-20 px")
    ap.add_argument("--snr", type=float, nargs=2, default=(2.0, 3.5),
                    help="faint wire contrast range, in background noise units (short ones +1)")
    ap.add_argument("--src", type=Path, default=TRAIN, help="training set to paste onto")
    ap.add_argument("--dst", type=Path, default=COPYPASTE, help="where the pasted photos go")
    ap.add_argument("--sprite-photos", nargs="*", default=None,
                    help="cut sprites only from these hand-labelled photos (hold-out runs: not from the held-out one)")
    args = ap.parse_args()

    from PIL import Image
    from score import load_gt

    src, dst = args.src, args.dst
    manifest = read_json(src / "manifest.json")
    frames = load_gt()
    if args.sprite_photos is not None:
        frames = [f for f in frames if f.name in set(args.sprite_photos)]
        if not frames:
            raise SystemExit(f"no hand-labelled photo called {args.sprite_photos}")
    sprites = make_sprites(frames)
    by_fam = {f: [s for s in sprites if s.family == f] for f in range(len(FAMILY_ANGLES))}
    print(f"{len(sprites)} hand-drawn wires usable as sprites "
          f"({', '.join(f'{FAMILY_ANGLES[f]:.0f} deg: {len(v)}' for f, v in by_fam.items())})")
    if len(sprites) < 10:
        raise SystemExit("too few clean hand-drawn wires to paste")

    rng = np.random.default_rng(args.seed)
    (dst / "previews").mkdir(parents=True, exist_ok=True)
    cases = []
    for c in manifest["cases"]:
        if c["source"] not in ("pseudo", "corrected"):
            continue
        for k in range(args.copies):
            img = read_png(src / "images" / f"{c['case']}.png").astype(np.float32) / 255.0
            lab = read_png(src / "labels" / f"{c['case']}.png").copy()
            protect = np.zeros(lab.shape, bool)
            kinds = [add_event(img, lab, protect, sprites, by_fam, rng, args.gap)
                     for _ in range(args.events)]
            kinds += [add_faint(img, lab, protect, sprites, rng, args.p_short, args.snr)
                      for _ in range(args.faint)]
            name = f"CP_{c['case']}_{k}"
            save_u8(dst / "images" / f"{name}.png", np.round(img * 255))
            save_u8(dst / "labels" / f"{name}.png", lab)
            if len(cases) < 20:
                Image.fromarray(preview(img, lab, protect)).save(dst / "previews" / f"{name}.png")
            cases.append({"case": name, "from": c["case"], "photo": c["photo"], "run": c["run"],
                          "source": "copypaste", "shape": list(lab.shape),
                          "pairs": kinds.count("pair"), "crossings": kinds.count("crossing"),
                          "faint": kinds.count("faint"), "faint_short": kinds.count("faint_short"),
                          "wire": float((lab == WIRE).mean()),
                          "ignore": float((lab == IGNORE).mean())})
            extra = (f", {cases[-1]['faint']:2d} faint + {cases[-1]['faint_short']:2d} short faint"
                     if args.faint else "")
            print(f"  {name:30s} {cases[-1]['pairs']:2d} pairs, {cases[-1]['crossings']:2d} crossings"
                  + extra, flush=True)
    write_json(dst / "manifest.json", {
        "settings": {"copies": args.copies, "events": args.events, "gap_px": list(args.gap),
                     "margin_px": MARGIN, "p_pair": P_PAIR, "seed": args.seed,
                     "n_sprites": len(sprites), "source_set": str(src.name),
                     "sprite_photos": [f.name for f in frames],
                     "faint": args.faint, "p_short": args.p_short, "snr": list(args.snr)},
        "cases": cases})
    print(f"wrote {len(cases)} pasted photos to {dst}")


if __name__ == "__main__":
    main()
