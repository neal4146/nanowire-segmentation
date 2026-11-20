"""Shared helpers: loading FEI/Helios SEM TIFFs, cropping the info bar, pixel size, stage tilt."""
import os, re
import numpy as np
from PIL import Image

# Source file of each manually labelled raw image (found by image matching; see report Section 2.2).
LABEL_SOURCES = {
    "NbS3 1": "02_Data/SEM images/NbS3 on sapphire/251006N (T(S)=225C, T(Nb)=775C, 30mins)/251006N_47.tif",
    "NbS3 2": "02_Data/SEM images/NbS3 on sapphire/251015M (T(S)=225C, T(Nb)=775C, 15mins)/251015M_12.tif",
    "NbS3 3": "02_Data/SEM images/NbS3 on sapphire/251006N (T(S)=225C, T(Nb)=775C, 30mins)/251006N_45.tif",
    "NbS3 4": "02_Data/SEM images/NbS3 on sapphire/251006N (T(S)=225C, T(Nb)=775C, 30mins)/251006N_24.tif",
    "NbS3 5": "02_Data/SEM images/NbS3 on sapphire/251006N (T(S)=225C, T(Nb)=775C, 30mins)/251006N_20.tif",
    "NbS3 6": "02_Data/SEM images/NbS3 on sapphire/251006N (T(S)=225C, T(Nb)=775C, 30mins)/251006N_17.tif",
    "NbS3 7": "02_Data/SEM images/NbS3 on sapphire/251006N (T(S)=225C, T(Nb)=775C, 30mins)/251006N_9.tif",
    "TiS3_80000\u00d7_1": "02_Data/SEM images/TiS3 on sapphire/SF251211M TiS3 30 min/SF251211M_80000\u00d7_66.tif",
    "TiS3_80000\u00d7_2": "02_Data/SEM images/TiS3 on sapphire/SF251211M TiS3 30 min/SF251211M_80000\u00d7_62.tif",
}

def fei_metadata(path):
    """Return dict with pixel_nm, hv_kV, tilt_deg, res_y (image rows without info bar), or None."""
    try:
        t = Image.open(path).tag_v2.get(34682)
    except Exception:
        return None
    if not t:
        return None
    g = lambda k: re.search(k + r"=([-\d.eE+]+)", t)
    hfw, rx, ry = float(g("HFW").group(1)), int(g("ResolutionX").group(1)), int(g("ResolutionY").group(1))
    st = g("StageT")
    return {"pixel_nm": hfw / rx * 1e9, "hv_kV": float(g(r"\nHV").group(1)) / 1000 if g(r"\nHV") else None,
            "tilt_deg": float(np.degrees(float(st.group(1)))) if st else 0.0, "res_x": rx, "res_y": ry}

def load_gray(path, crop_bar=True):
    """Load as float32 in [0,1]; handles 8/16-bit and RGB(A). Crops the FEI info bar (keeps top 1024/1094 of rows)."""
    im = Image.open(path)
    a = np.array(im).astype(np.float32)
    if a.ndim == 3:
        a = a[..., :3].mean(-1)
    a /= 65535.0 if a.max() > 255 else 255.0
    if crop_bar:
        a = a[: int(round(a.shape[0] * 1024 / 1094))]
    return a

def normalize(a, lo=1, hi=99):
    p1, p2 = np.percentile(a, [lo, hi])
    return np.clip((a - p1) / (p2 - p1 + 1e-9), 0, 1).astype(np.float32)
