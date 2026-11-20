"""Convert hand-drawn coloured outlines (burned into RGBA copies) into 16-bit instance masks.

Steps: stroke detection (colour or large difference from the raw image) -> link open stroke ends
(annotators often left the short ends of wires open) -> fill enclosed regions -> give stroke pixels back
to the adjacent instance (the true boundary lies under the stroke) -> flag suspicious regions for QC.
Usage: python convert_labels.py  (run from the Materials folder; writes starter_code/outputs/labels)
"""
import os, glob, json
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import morphology, measure, segmentation, draw
from common import load_gray

MAX_GAP = 22      # px; longest open gap to bridge between stroke end points
MIN_AREA = 150    # px; smaller regions are dropped
OUT = os.path.join(os.path.dirname(__file__), "outputs", "labels")

def endpoints(skel):
    nb = ndi.convolve(skel.astype(int), np.ones((3, 3), int), mode="constant") - skel
    return np.argwhere(skel & (nb == 1))

def link_gaps(stroke, max_gap=MAX_GAP):
    skel = morphology.skeletonize(stroke)
    ep = endpoints(skel)
    if len(ep) < 2:
        return stroke
    out = stroke.copy()
    d = np.sqrt(((ep[:, None, :] - ep[None, :, :]) ** 2).sum(-1)); np.fill_diagonal(d, np.inf)
    used = set()
    for i in np.argsort(d.min(1)):
        if i in used: continue
        j = int(np.argmin(d[i]))
        if d[i, j] <= max_gap and j not in used:
            rr, cc = draw.line(*ep[i], *ep[j]); out[rr, cc] = True; used |= {i, int(j)}
    return morphology.binary_dilation(out, morphology.disk(1)) | stroke

def convert(lab_path, raw_path):
    lab = np.array(Image.open(lab_path))[..., :3].astype(np.float32)
    lab = lab[: int(round(lab.shape[0] * 1024 / 1094))]
    raw = load_gray(raw_path) * 255
    if raw.shape != lab.shape[:2]:
        raw = np.array(Image.fromarray(raw.astype(np.uint8)).resize((lab.shape[1], lab.shape[0]), Image.BILINEAR), np.float32)
    chroma = lab.max(-1) - lab.min(-1)
    diff = np.abs(lab.mean(-1) - raw)
    stroke = (chroma > 40) | ((diff > 60) & (lab.max(-1) < 40))       # coloured or black strokes
    stroke = morphology.remove_small_objects(stroke, 20)
    stroke = morphology.binary_closing(stroke, morphology.disk(1))
    stroke = link_gaps(stroke)
    filled = ndi.binary_fill_holes(stroke)
    interior = filled & ~stroke
    lab_i = measure.label(interior, connectivity=1)
    lab_i = morphology.remove_small_objects(lab_i, MIN_AREA)
    inst = segmentation.expand_labels(lab_i, distance=4) * filled      # hand stroke pixels back
    inst, _, _ = segmentation.relabel_sequential(inst)
    # QC flag: region bounded by strokes of several different colours -> probably a background pocket
    hue = (np.degrees(np.arctan2(np.sqrt(3) * (lab[..., 1] - lab[..., 2]), 2 * lab[..., 0] - lab[..., 1] - lab[..., 2])) % 360) // 45
    flags = {}
    for r in measure.regionprops(inst):
        ring = morphology.binary_dilation(inst == r.label, morphology.disk(3)) & stroke & (chroma > 40)
        h = hue[ring]
        if h.size > 30:
            frac = np.bincount(h.astype(int), minlength=8) / h.size
            flags[int(r.label)] = bool((frac > 0.25).sum() >= 2)
    return inst.astype(np.uint16), flags, lab.astype(np.uint8)

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    summary = {}
    for lp in sorted(glob.glob("02_Data/Manual labelling/*abeled*")):
        base = os.path.basename(lp)
        key = base.replace(" Labeled", "").rsplit(".", 1)[0]
        rp = [p for p in glob.glob("02_Data/Manual labelling/*") if "abeled" not in p and os.path.basename(p).rsplit(".", 1)[0] == key][0]
        inst, flags, _ = convert(lp, rp)
        Image.fromarray(inst).save(os.path.join(OUT, key + "_instances.png"))
        json.dump(flags, open(os.path.join(OUT, key + "_qc_flags.json"), "w"))
        summary[key] = {"instances": int(inst.max()), "flagged_for_QC": int(sum(flags.values())), "shape": list(inst.shape)}
        print(key, summary[key])
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
