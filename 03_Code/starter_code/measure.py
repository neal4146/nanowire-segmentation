"""Per-instance morphology from an instance mask (rectangle model) and orientation utilities.
Angle convention everywhere: degrees in [0,180), measured from the +x (column) axis in image
coordinates (y pointing down)."""
import numpy as np
from scipy import ndimage as ndi
from skimage import measure

def instance_table(inst, pixel_nm=1.0):
    rows = []
    for r in measure.regionprops(inst):
        mu = r.moments_central                      # mu[p,q]: p = row power, q = col power
        mxx, myy, mxy = mu[0, 2] / mu[0, 0], mu[2, 0] / mu[0, 0], mu[1, 1] / mu[0, 0]
        ang = 0.5 * np.degrees(np.arctan2(2 * mxy, mxx - myy)) % 180
        lam = np.linalg.eigvalsh([[mxx, mxy], [mxy, myy]])
        L, W = np.sqrt(12 * lam[1]), np.sqrt(12 * max(lam[0], 1e-9))   # exact for a solid rectangle
        rows.append({"label": int(r.label), "area_px": int(r.area), "length_nm": L * pixel_nm, "width_nm": W * pixel_nm,
                     "angle_deg": float(ang), "touches_border": bool(min(r.bbox[:2]) == 0 or r.bbox[2] == inst.shape[0] or r.bbox[3] == inst.shape[1])})
    return rows

def dominant_directions(angles_deg, weights=None, spacing=60.0):
    """Fit three directions theta, theta+60, theta+120 (mod 180). Returns theta in [0,60) and
    R in [0,1] (1 = all orientations exactly on the three directions, ~0 = random)."""
    a = np.radians(np.asarray(angles_deg, float) * 360.0 / spacing)
    w = np.ones_like(a) if weights is None else np.asarray(weights, float)
    c, s = (w * np.cos(a)).sum(), (w * np.sin(a)).sum()
    return (np.degrees(np.arctan2(s, c)) * spacing / 360.0) % spacing, float(np.hypot(c, s) / w.sum())

def structure_tensor_orientations(img, sigma=1.5, rho=5.0):
    """Label-free local line orientation (same convention) and coherence in [0,1]."""
    g = ndi.gaussian_filter(img.astype(float), sigma)
    gy, gx = np.gradient(g)
    Jxx, Jyy, Jxy = (ndi.gaussian_filter(v, rho) for v in (gx * gx, gy * gy, gx * gy))
    grad_ang = 0.5 * np.degrees(np.arctan2(2 * Jxy, Jxx - Jyy))
    coh = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2) / (Jxx + Jyy + 1e-12)
    return (grad_ang + 90) % 180, coh
