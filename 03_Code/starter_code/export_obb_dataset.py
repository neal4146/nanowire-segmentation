"""Export the converted instance masks as an oriented-bounding-box (OBB) dataset in Ultralytics YOLO format.

Each wire becomes a rotated rectangle (centre, length, width, angle) fitted from its mask moments.
Images are tiled (default 1024 x 1024 with 256 px overlap) so thin TiS3 wires keep native resolution.
Also reports how well a rectangle describes each labelled wire (mask-vs-rectangle IoU).

Run from the Materials folder:   python 03_Code/starter_code/export_obb_dataset.py
Then (in an environment with a CUDA build of PyTorch and `pip install ultralytics`):
    yolo obb train data=starter_code/outputs/obb_dataset/data.yaml model=yolo11s-obb.pt imgsz=1024 epochs=150 batch=4
(YOLO26 OBB models can be used the same way if your Ultralytics version provides them.)
"""
import os, sys, json
import numpy as np
from PIL import Image
from skimage import measure, draw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LABEL_SOURCES, load_gray, normalize

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(HERE, "outputs", "labels")
OUT = os.path.join(HERE, "outputs", "obb_dataset")
CLASSES = {"NbS3": 0, "TiS3": 1}
# Split by image; NbS3 2 comes from a different growth run (251015M) and is held out as a test image.
SPLIT = {"NbS3 1": "train", "NbS3 3": "train", "NbS3 5": "train", "NbS3 6": "train", "NbS3 7": "train",
         "NbS3 4": "val", "NbS3 2": "test", "TiS3_80000×_1": "train", "TiS3_80000×_2": "val"}
TILE, STEP = 1024, 768


def rect_from_mask(m):
    """Rotated rectangle from second moments: returns centre (x, y), length, width, angle (rad) and corners."""
    ys, xs = np.nonzero(m)
    cx, cy = xs.mean(), ys.mean()
    cov = np.cov(np.vstack([xs - cx, ys - cy]))
    evals, evecs = np.linalg.eigh(cov)
    L, W = np.sqrt(12 * evals[1]), np.sqrt(12 * max(evals[0], 1e-9))
    u, v = evecs[:, 1], evecs[:, 0]                      # long axis, short axis (x, y)
    corners = np.array([[cx, cy] + a * L / 2 * u + b * W / 2 * v for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))])
    return (cx, cy), L, W, np.arctan2(u[1], u[0]), corners


def rect_iou(m, corners):
    rr, cc = draw.polygon(corners[:, 1], corners[:, 0], m.shape)
    r = np.zeros_like(m); r[rr, cc] = True
    return (m & r).sum() / max((m | r).sum(), 1)


if __name__ == "__main__":
    stats = {}
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(OUT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUT, "labels", split), exist_ok=True)
    for key, src in LABEL_SOURCES.items():
        inst = np.array(Image.open(os.path.join(LAB, key + "_instances.png"))).astype(np.int64)
        img = normalize(load_gray(os.path.join(HERE, "..", "..", src)))
        if img.shape != inst.shape:
            img = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize(inst.shape[::-1], Image.BILINEAR)) / 255.0
        cls = CLASSES[key[:4]]
        rects, ious = [], []
        for r in measure.regionprops(inst):
            m = inst == r.label
            _, L, W, _, corners = rect_from_mask(m)
            rects.append(corners); ious.append(rect_iou(m, corners))
        stats[key] = dict(n=len(ious), rect_iou_median=round(float(np.median(ious)), 3),
                          frac_rect_iou_ge_0_8=round(float(np.mean(np.array(ious) >= 0.8)), 3))
        H, Wd = inst.shape
        ys = list(range(0, max(H - TILE, 0) + 1, STEP)) or [0]
        xs = list(range(0, max(Wd - TILE, 0) + 1, STEP)) or [0]
        if ys[-1] + TILE < H: ys.append(H - TILE)
        if xs[-1] + TILE < Wd: xs.append(Wd - TILE)
        for y0 in ys:
            for x0 in xs:
                name = "%s_%d_%d" % (key.replace(" ", "_").replace("×", "x"), y0, x0)
                crop = img[y0:y0 + TILE, x0:x0 + TILE]
                Image.fromarray((crop * 255).astype(np.uint8)).save(os.path.join(OUT, "images", SPLIT[key], name + ".png"))
                lines = []
                for c in rects:
                    cc = c - [x0, y0]
                    inside = ((cc[:, 0] >= 0) & (cc[:, 0] < crop.shape[1]) & (cc[:, 1] >= 0) & (cc[:, 1] < crop.shape[0])).mean()
                    if inside < 0.5:            # keep wires at least half inside the tile
                        continue
                    cc[:, 0] = np.clip(cc[:, 0], 0, crop.shape[1] - 1) / crop.shape[1]
                    cc[:, 1] = np.clip(cc[:, 1], 0, crop.shape[0] - 1) / crop.shape[0]
                    lines.append("%d " % cls + " ".join("%.6f" % v for v in cc.ravel()))
                open(os.path.join(OUT, "labels", SPLIT[key], name + ".txt"), "w").write("\n".join(lines))
        print(key, stats[key])
    with open(os.path.join(OUT, "data.yaml"), "w") as f:
        f.write("path: %s\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: NbS3\n  1: TiS3\n" % OUT.replace("\\", "/"))
    json.dump(stats, open(os.path.join(OUT, "rectangle_fit_stats.json"), "w"), indent=1)
