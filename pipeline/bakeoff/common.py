"""Paths and helpers shared by the bake-off scripts (see bakeoff/README.md).

Everything works at the 50k-equivalent scale (5.396 nm/px) that production
and compare/ use: photos go through ridge_nanowire_images2.load_frame_50k,
the hand-labelled 80k photos through compare/score.load_gt.

Label values in every training label image:
  0 background, 1 wire, 2 ignore (not trained on)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ridge"))
sys.path.insert(0, str(ROOT / "compare"))

MATERIALS = ROOT.parent if (ROOT.parent / "03_Code").is_dir() else ROOT.parent / "Materials_ThangPham"


def _first(env: str, *paths: Path) -> Path:
    """$env if set, else the first path that exists (else the first path)."""
    if env in os.environ:
        return Path(os.environ[env])
    return next((p for p in paths if p.exists()), paths[0])


# the laptop layout first, then the copies the README puts in data/ on ARC
PHOTO_ROOT = _first("BAKEOFF_PHOTOS", MATERIALS / "02_Data" / "SEM images" / "TiS3 on sapphire",
                    ROOT / "data" / "real" / "TiS3 on sapphire")
LABEL_DIR = _first("BAKEOFF_LABELS", MATERIALS / "03_Code" / "labelling",
                   ROOT / "data" / "labels")

OUT = ROOT / "results" / "bakeoff"           # results/ is untracked by git
TRAIN = OUT / "train"                        # images/, labels/, manifest.json
COPYPASTE = OUT / "copypaste"                # images/, labels/, manifest.json
TRAIN_E = OUT / "train_E"                    # contender E: train/ + faint ignore + hand fixes
COPYPASTE_E = OUT / "copypaste_E"            # contender E: pairs, crossings + faint wires
# hand corrections of TRAINING photos (never test photos): correct_training.py
TRAIN_FIX_DIR = LABEL_DIR / "train_fixes"
TEST_IMAGES = OUT / "test_images"            # <case>_0000.png, the photos to predict
PRED = OUT / "pred"                          # pred/<contender>/<case>.png (0/255 masks)
MODELS = OUT / "models"
TEST_LIST = Path(__file__).resolve().parent / "test_photos.txt"

BG, WIRE, IGNORE = 0, 1, 2
# rescale factors to 50k where the filters were tuned and checked (50k-80k photos);
# outside it the pseudo-labels are not trustworthy (ridge_nanowire_images2 docstring)
BAND = (0.625, 1.0)
# <run>_<mag>x_<n>; skips the tilted frame (SF251211M_87_30.0deg) and unnamed files
NAME_RE = re.compile(r"^SF\d+[A-Z]_\d+×_\d+$")


def case_name(photo: str) -> str:
    """ASCII case name for a photo (nnU-Net and Windows tools dislike the ×)."""
    return photo.replace("×", "x")


def photo_name(text: str) -> str:
    """Photo name with the × restored (accepts a plain x, like label_wires.py)."""
    return re.sub(r"(\d)x_", r"\1×_", text.strip())


def run_of(photo: str) -> str:
    """Growth run of a photo, from the folder names (SF251211N -> 15 min, ...)."""
    prefix = photo.split("_")[0]
    for d in PHOTO_ROOT.iterdir():
        if d.is_dir() and d.name.startswith(prefix):
            return " ".join(d.name.split()[-2:])        # "SF251211N TiS3 15 min" -> "15 min"
    return prefix


def test_photos() -> list[str]:
    """Photos reserved for the test set (bakeoff/test_photos.txt)."""
    names = []
    for line in TEST_LIST.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line:
            names.append(photo_name(line))
    return names


def find_photo(photo: str) -> Path:
    hits = sorted(PHOTO_ROOT.glob(f"*/{photo_name(photo)}.tif"))
    if not hits:
        raise SystemExit(f"photo {photo}.tif not found under {PHOTO_ROOT}")
    return hits[0]


def in_band_photos() -> list[Path]:
    """Every TiS3 photo whose scale is inside BAND, sorted by name."""
    from ridge_nanowire_images2 import TARGET_PX, read_pixel_size
    out = []
    for p in sorted(PHOTO_ROOT.glob("*/*.tif")):
        if not NAME_RE.match(p.stem):
            continue
        f = read_pixel_size(p)[0] / TARGET_PX
        if BAND[0] - 1e-3 <= f <= BAND[1] + 1e-3:
            out.append(p)
    return out


def load_photo(path: Path) -> tuple[np.ndarray, dict]:
    """(img01 at 50k-equivalent, meta) exactly as production loads it."""
    from ridge_nanowire_images2 import load_frame_50k
    gray, meta = load_frame_50k(path)
    return gray / 255.0, meta


def native_hash(path: Path) -> str:
    """Hash of the raw pixel data, to spot the same photo under two names."""
    return hashlib.md5(np.ascontiguousarray(np.array(Image.open(path))).tobytes()).hexdigest()


def save_u8(path: Path, a: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(a, np.uint8), mode="L").save(path)


def img_to_u8(img01: np.ndarray) -> np.ndarray:
    return np.clip(np.round(img01 * 255.0), 0, 255).astype(np.uint8)


def read_png(path: Path) -> np.ndarray:
    return np.array(Image.open(path))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
