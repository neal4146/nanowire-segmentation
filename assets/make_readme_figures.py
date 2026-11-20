"""Build README figures. Uses only a photo the network never trained on (SF251211M_80000x_62,
not a test photo) plus the scored test numbers from summary_round2.csv."""
import csv, sys
from pathlib import Path
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"C:\Users\Acer\Downloads\Materials_ThangPham")
RES = ROOT / "Ridge-Filtering-Nanowires-main" / "results"
OUT = ROOT / "Materials_ThangPham" / "assets"
OUT.mkdir(exist_ok=True)
STEM = "SF251211M_80000×_62"
RUN = "SF251211M TiS3 30 min"

def imread(p, flags=cv2.IMREAD_COLOR):
    a = np.fromfile(str(p), dtype=np.uint8)
    return cv2.imdecode(a, flags)

new = imread(RES / "measure" / "TiS3 on sapphire (unet)" / "overlays" / RUN / f"{STEM}.png")
old = imread(RES / "measure" / "TiS3 on sapphire" / "overlays" / RUN / f"{STEM}.png")
raw = imread(ROOT / "Materials_ThangPham" / "02_Data" / "SEM images" / "TiS3 on sapphire" / RUN / f"{STEM}.tif",
             cv2.IMREAD_GRAYSCALE)
prob = imread(next((RES / "measure" / "TiS3 on sapphire (unet)" / "probs").rglob(f"{STEM}_*_prob.png")),
              cv2.IMREAD_GRAYSCALE)
H, W = new.shape[:2]
print("overlay", new.shape, "raw", raw.shape, "prob", prob.shape)

# raw tif may include a databar below the image; prob covers the measured area
ph, pw = prob.shape
raw_c = raw[:round(ph * raw.shape[1] / pw), :]  # drop the SEM databar; prob is a rescaled copy
raw_r = cv2.resize(raw_c, (W, H), interpolation=cv2.INTER_AREA)
prob_r = cv2.resize(prob, (W, H), interpolation=cv2.INTER_AREA)

TEXT = "#0b0b0b"; MUTED = "#52514e"; GRID = "#e4e3df"; BLUE = "#2a78d6"; GRAY = "#a3a29c"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": TEXT,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": GRID, "figure.facecolor": "white", "axes.facecolor": "white"})

def rgb(a):
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB) if a.ndim == 3 else a

# crop away the legend box (top-left), keep a dense region
y0, y1, x0, x1 = 110, 470, 150, 690

def panel_row(imgs, titles, fname, cmaps):
    fig, axes = plt.subplots(1, len(imgs), figsize=(5.2 * len(imgs), 5.2 * (y1 - y0) / (x1 - x0) + 0.6))
    for ax, im, t, cm in zip(axes, imgs, titles, cmaps):
        ax.imshow(rgb(im)[y0:y1, x0:x1], cmap=cm, vmin=0, vmax=255)
        ax.set_title(t, fontsize=13, color=TEXT, pad=8, loc="left")
        ax.axis("off")
    fig.tight_layout(pad=0.6)
    fig.savefig(OUT / fname, dpi=130)
    plt.close(fig)

panel_row([raw_r, old, new],
          ["SEM image (TiS$_3$ on sapphire, 30 min)", "Before: classical ridge filter", "After: U-Net + LSD + verifier"],
          "before_after.png", ["gray", None, None])
panel_row([raw_r, prob_r, new],
          ["1. Raw SEM", "2. U-Net wire probability", "3. Straight pieces kept by the verifier"],
          "pipeline.png", ["gray", "magma", None])

# ---- test-set wire F1 progression ----
rows = {r["contender"]: r for r in csv.DictReader(open(RES / "bakeoff" / "summary_round2.csv", encoding="utf-8-sig"))}
steps = [("Classical ridge filter (old lab tool)", 0.506, 0.481, 0.531),
         ("U-Net, round 1", 0.529, 0.503, 0.554),
         ("+ thin ridge mask", 0.570, 0.541, 0.598),
         ("+ LSD straight-line extraction", 0.639, 0.611, 0.665)]
b = rows["E2_thin+V"]
steps.append(("+ learned piece verifier (current)", float(b["wire_f1"]), float(b["wire_f1_lo"]), float(b["wire_f1_hi"])))
assert abs(float(rows["A_path_open_clean"]["wire_f1"]) - 0.506) < 0.001
assert abs(float(rows["E_thin+X"]["wire_f1"]) - 0.639) < 0.001

fig, ax = plt.subplots(figsize=(9, 3.6))
ys = np.arange(len(steps))[::-1]
for y, (name, v, lo, hi) in zip(ys, steps):
    c = BLUE if "current" in name else GRAY
    ax.barh(y, v, height=0.56, color=c, zorder=2)
    ax.plot([lo, hi], [y, y], color=TEXT, lw=1.4, zorder=3)
    ax.text(hi + 0.01, y, f"{v:.3f}", va="center", fontsize=11, color=TEXT)
ax.set_yticks(ys, [s[0] for s in steps], fontsize=11)
ax.set_xlim(0, 0.85)
ax.set_xlabel("Wire F1 on 1,401 hand-labelled held-out wires (whiskers: 95% bootstrap interval)")
ax.grid(axis="x", color=GRID, zorder=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
fig.tight_layout()
fig.savefig(OUT / "f1_progression.png", dpi=150)
plt.close(fig)

# ---- per-photo: old vs new ----
photos = [("60", "30 min"), ("37", "60 min"), ("15", "60 min"), ("22", "15 min")]
a = rows["A_path_open_clean"]
fig, ax = plt.subplots(figsize=(7.5, 3.0))
for i, (p, t) in enumerate(photos):
    va, vb = float(a[f"wire_f1_{p}"]), float(b[f"wire_f1_{p}"])
    ax.plot([va, vb], [i, i], color=GRID, lw=3, zorder=1)
    ax.scatter([va], [i], s=90, color=GRAY, zorder=2, edgecolor="white", linewidth=2)
    ax.scatter([vb], [i], s=90, color=BLUE, zorder=3, edgecolor="white", linewidth=2)
    ax.text(vb + 0.02, i, f"+{vb - va:.2f}", va="center", fontsize=10, color=MUTED)
ax.set_yticks(range(len(photos)), [f"Test photo {p} ({t})" for p, t in photos])
ax.invert_yaxis()
ax.set_xlim(0.2, 0.95)
ax.set_xlabel("Wire F1")
ax.scatter([], [], s=70, color=GRAY, label="Classical ridge filter")
ax.scatter([], [], s=70, color=BLUE, label="Current pipeline")
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=10)
ax.grid(axis="x", color=GRID, zorder=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
fig.tight_layout()
fig.savefig(OUT / "per_photo.png", dpi=150)
plt.close(fig)

# ---- orientation histogram across the TiS3 run (length-weighted) ----
ang, ln = [], []
for r in csv.DictReader(open(RES / "measure" / "TiS3 on sapphire (unet)" / "wires.csv", encoding="utf-8-sig")):
    try:
        ang.append(float(r["angle_deg"]) % 180); ln.append(float(r["length_nm"]))
    except ValueError:
        pass
ang, ln = np.array(ang), np.array(ln)
print("wires", len(ang))
fig, ax = plt.subplots(figsize=(9, 3.2))
ang = np.where(ang >= 150, ang - 180, ang)  # 0 and 180 are the same direction
bins = np.arange(-30, 151, 2)
h, _ = np.histogram(ang, bins=bins, weights=ln / 1000)
ctr = bins[:-1] + 1
ax.bar(ctr, h, width=1.7, color=BLUE, zorder=2)
for lo, hi in ((-30, 30), (30, 90), (90, 150)):
    m = (ang >= lo) & (ang < hi)
    pk = np.median(ang[m])
    i = np.argmax(np.where((ctr >= lo) & (ctr < hi), h, -1))
    ax.text(ctr[i] + 3, h[i] * 0.95, f"{100 * ln[m].sum() / ln.sum():.0f}% of length\nmedian {pk:.0f}°",
            color=MUTED, fontsize=10, va="top")
ax.set_xlim(-31, 151)
ax.set_xlabel("Wire direction on screen (degrees)")
ax.set_ylabel("Total wire length (µm)")
ax.grid(axis="y", color=GRID, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "orientations.png", dpi=150)
plt.close(fig)
print("done", sorted(p.name for p in OUT.iterdir()))
