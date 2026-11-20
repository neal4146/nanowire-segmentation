"""Steger line detector: sub-pixel centre lines plus a width for every centre point.

Steger (1998, IEEE TPAMI 20(2)): smooth with a Gaussian of scale sigma, take
the Hessian at each pixel, and use the eigenvector with the largest |eigenvalue|
as the line normal n. Along n, the second-order Taylor expansion has its
extremum at offset t = -(g_n) / (g_nn). A pixel is a centre-line point when
that extremum falls inside the pixel (|t n_x|, |t n_y| <= 0.5) and the
curvature across the line is strong. Dark lines (TiS3 wires) have g_nn > 0.

Width: from each centre point, the nearest gradient-magnitude peak on each
side along +/- n (a local maximum at least half the profile's maximum) marks
the wire edge; width = the distance between the two edges, as in Steger's
edge-based width step (without his bias correction). Taking the strongest
peak instead lands on the outer edge of the bright rim and doubles the width.

Mask for the shared scorer: every pixel within half the local width of a
centre point. So wire_extract's width measurement reports Steger's width,
not the threshold's.
"""
from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi

SIGMAS = (1.5, 2.0, 2.5)    # px at 50k: wire bodies ~5 px wide -> sigma >= w / (2 sqrt 3)
EDGE_SEARCH = np.arange(0.5, 8.01, 0.5)      # px searched along the normal, each side
WIDTH_CLIP = (2.0, 14.0)                     # px


def _hessian_frame(img: np.ndarray, sigma: float):
    d = lambda o: ndi.gaussian_filter(img, sigma, order=o)   # noqa: E731  (row, col) orders
    return d((0, 1)), d((1, 0)), d((0, 2)), d((2, 0)), d((1, 1))


def steger_response(img01: np.ndarray, polarity: str = "dark") -> dict:
    """Centre-line points, their scale-normalised strength, normal and width."""
    img = img01.astype(np.float64)
    best = None
    for s in SIGMAS:
        rx, ry, rxx, ryy, rxy = _hessian_frame(img, s)
        tmp = np.sqrt((rxx - ryy) ** 2 + 4 * rxy ** 2)
        lam_hi, lam_lo = (rxx + ryy + tmp) / 2, (rxx + ryy - tmp) / 2
        lam = lam_hi if polarity == "dark" else lam_lo          # curvature across the line
        # eigenvector (nx, ny) of [[rxx, rxy], [rxy, ryy]] for eigenvalue lam
        nx, ny = rxy, lam - rxx
        alt = np.abs(nx) + np.abs(ny) < 1e-12
        nx, ny = np.where(alt, lam - ryy, nx), np.where(alt, rxy, ny)
        norm = np.hypot(nx, ny) + 1e-20
        nx, ny = nx / norm, ny / norm
        strength = (lam if polarity == "dark" else -lam) * s ** 2
        if best is None:
            best = dict(strength=strength, nx=nx, ny=ny, rx=rx, ry=ry, rxx=rxx, ryy=ryy, rxy=rxy,
                        sigma=np.full(img.shape, s))
        else:
            better = strength > best["strength"]
            for k, v in (("strength", strength), ("nx", nx), ("ny", ny), ("rx", rx), ("ry", ry),
                         ("rxx", rxx), ("ryy", ryy), ("rxy", rxy)):
                best[k] = np.where(better, v, best[k])
            best["sigma"] = np.where(better, s, best["sigma"])

    nx, ny = best["nx"], best["ny"]
    g_n = best["rx"] * nx + best["ry"] * ny
    g_nn = best["rxx"] * nx * nx + 2 * best["rxy"] * nx * ny + best["ryy"] * ny * ny
    t = -g_n / np.where(np.abs(g_nn) < 1e-12, 1e-12, g_nn)
    centre = (np.abs(t * nx) <= 0.5) & (np.abs(t * ny) <= 0.5) & (best["strength"] > 0)
    strength = np.where(centre, best["strength"], 0.0)
    strength = strength / (np.percentile(strength[centre], 99.5) + 1e-12) if centre.any() else strength

    # width: nearest gradient-magnitude maximum on each side along the normal
    rr, cc = np.nonzero(centre)
    width = np.zeros(img.shape)
    if rr.size:
        gm = ndi.gaussian_gradient_magnitude(img, 1.0)
        pr = rr + t[rr, cc] * ny[rr, cc]                  # sub-pixel centre (row, col)
        pc = cc + t[rr, cc] * nx[rr, cc]
        sides = []
        for sign in (1, -1):
            r = pr[:, None] + sign * EDGE_SEARCH[None] * ny[rr, cc][:, None]
            c = pc[:, None] + sign * EDGE_SEARCH[None] * nx[rr, cc][:, None]
            prof = ndi.map_coordinates(gm, [r.ravel(), c.ravel()], order=1,
                                       mode="nearest").reshape(r.shape)
            padded = np.pad(prof, ((0, 0), (1, 1)), mode="edge")
            peak = ((prof >= padded[:, :-2]) & (prof >= padded[:, 2:])
                    & (prof >= 0.5 * prof.max(axis=1, keepdims=True)))
            first = np.where(peak.any(axis=1), np.argmax(peak, axis=1), np.argmax(prof, axis=1))
            sides.append(EDGE_SEARCH[first])
        width[rr, cc] = np.clip(sides[0] + sides[1], *WIDTH_CLIP)
    return {"strength": strength.astype(np.float32), "width": width.astype(np.float32)}


def steger_mask(img01: np.ndarray, resp: dict, thr: float, high: float = 0.3,
                min_len: int = 8) -> np.ndarray:
    """8-connected hysteresis on the centre points, short chains dropped, then
    each kept centre point painted as a disc of its own width."""
    s = resp["strength"]
    low = s > thr
    lab, n = ndi.label(low, structure=np.ones((3, 3), int))
    if n == 0:
        return np.zeros(s.shape, bool)
    has_high = np.zeros(n + 1, bool)
    has_high[np.unique(lab[s > max(high, thr)])] = True
    size = np.bincount(lab.ravel(), minlength=n + 1)
    keep = has_high & (size >= min_len)
    keep[0] = False
    centres = keep[lab]
    if not centres.any():
        return centres
    dist, (ir, ic) = ndi.distance_transform_edt(~centres, return_indices=True)
    return dist <= resp["width"][ir, ic] / 2.0
