# Starter code for the NbS3/TiS3 SEM segmentation project

Prototype scripts written while reviewing the project (see the report, Appendix B). They run on CPU with
Python + numpy/scipy/scikit-image/Pillow (`pip install numpy scipy scikit-image pillow tifffile`).
Run everything **from the Materials folder** (the top folder that contains `02_Data/` and `03_Code/`).

| File | What it does | Output |
|---|---|---|
| `common.py` | Loads FEI/Helios TIFFs, crops the 70-px information bar, reads pixel size, kV and **stage tilt** from TIFF tag 34682; maps each labelled image to its original file. | – |
| `convert_labels.py` | Turns the hand-drawn coloured outlines into 16-bit instance masks: stroke detection, **linking of open outline ends** (≤22 px), filling, returning stroke pixels to the wire, QC flags for probable background "pockets". | `outputs/labels/*_instances.png`, `*_qc_flags.json`, `summary.json` |
| `measure.py` | Per-wire length/width/angle (rectangle model from mask moments); fits the three epitaxial directions; label-free orientation from the structure tensor. | – |
| `metrics.py` | IoU-matched precision/recall/F1 + AJI+ (for NbS3 belts) and **centreline-based** wire matching (for thin TiS3 wires whose outlines are loose). | – |
| `baseline_ridge.py` | Experiment E1: two-polarity Sato ridge filter → hysteresis → skeleton → one instance per branch; grid-tuned on one image, tested on the others. | `outputs/E1_ridge_baseline.json` |
| `baseline_hough.py` | Experiment E1b (TiS3): ridge map + Hough line fitting restricted to the three growth directions + collinear merging. | `outputs/E1b_ridge_hough_TiS3.json` |
| `export_obb_dataset.py` | Exports rotated-rectangle labels in Ultralytics YOLO-OBB format (1024-px tiles, split by image) and reports how well a rectangle fits each wire. | `outputs/obb_dataset/` |

Typical order:
```
python 03_Code/starter_code/convert_labels.py
python 03_Code/starter_code/baseline_ridge.py
python 03_Code/starter_code/baseline_hough.py
python 03_Code/starter_code/export_obb_dataset.py
```

Oriented-box pilot (needs a CUDA build of PyTorch; see report Section 7.1):
```
pip install ultralytics
yolo obb train data=03_Code/starter_code/outputs/obb_dataset/data.yaml model=yolo11s-obb.pt imgsz=1024 epochs=150 batch=4
```

**Caveats.** The converted masks are drafts: correct them in napari before using them as ground truth
(QC flags mark likely background pockets; some wires in the labelled images were never outlined).
TiS3 outlines are about twice as wide as the wires, so do not use them for width measurement.
