# Measuring TiS₃ nanowires in SEM photos

Point the tool at a folder of SEM photos. For every photo it finds the
nanowires and measures each one: length, width, direction and position. It
also gives totals per photo (wire count, wires per µm², coverage) and per
growth run. You get spreadsheets you can open in Excel, plus a picture of
each photo with its measured wires drawn on.

## Run it

**Easiest:** drag a folder onto `measure\measure_wires.bat`.

**From a terminal**, in the repository folder:

```
venv\Scripts\python measure\measure_wires.py "D:\SEM\TiS3 on sapphire"
```

- Subfolders are included. Put each growth run in its own subfolder and you
  get one row per run in `folders.csv`.
- Results go to `results\measure\<folder name>\`. Use `--out <folder>` to
  write them somewhere else. If that folder already holds a run made with
  another method, the new run goes to `results\measure\<folder name> (<method>)\`
  instead, so earlier results are never overwritten.
- **First-time setup:** the repository's `venv` (main `README.md`:
  `pip install -r requirements.txt`), plus PyTorch for the network.
  - The tool uses PyTorch from its own Python if that has it. Otherwise it
    runs the network through the bake-off's `venv_dl` (`bakeoff/setup_dl_env.sh`).
  - It also needs the trained network `results\bakeoff\models\E2\final.pt` and
    the piece check `results\bakeoff\devcv\verifier_E2_cv\verifier_learned.joblib`.
    `--model` and `--verifier` point elsewhere.
  - Without PyTorch, `--method path_open_clean` still works (the old default).
- **Speed:** the 48 usable TiS₃ photos (of 179) take about 41 s on the lab
  laptop's RTX 4070. The network's output is cached in `<results>\probs\` and
  reused, so a rerun takes about 23 s.
  - Without a GPU, the network takes about 7–10 s per photo (measured on a
    32-thread CPU; slower on smaller PCs).
  - CPU output differs very slightly from the GPU's (under 1% of pixels by
    1–2/255, about 1 wire in 400). The scored test results used the GPU.

### How it finds the wires

The default, `--method unet`, is the best pipeline from the bake-off
(`bakeoff/README.md`, "Round 2"):

1. **Network.** A U-Net gives every pixel a probability of being wire. It was
   trained mostly on automatic labels where two classical filters agree, plus
   hand corrections on 10 training photos and 2 fully hand-labelled photos.
2. **Thin mask.** Only the ridge (centre line) of each wire's probability is
   kept, so neighbouring wires do not merge.
3. **Straight pieces.** A line segment detector cuts the mask into straight
   pieces. A zig-zag or V becomes one wire per arm (the labelling rule: a
   zig-zag is several nanowires), and pieces that continue each other across
   a small gap are rejoined.
4. **Learned check.** A small model looks at each piece (length, how sure the
   network is along it, contrast against its surroundings, whether it is a
   shadow or flank next to a longer wire, and more). It keeps the pieces that
   look like real wires, including short ones.

The thin mask is made for the whole photo and then cut into its 4 quarters.
Steps 3–4 run on each quarter, exactly as they were tested; on whole photos
they scored lower on the dev set (0.756 vs 0.769).

Then a wire crossing a quarter line is joined into one when both halves pass
the check and line up (within 15° and 2.5 px sideways, with a gap of at most
8 px). Otherwise the piece stops at the line. Such pieces are flagged in
`checks` (about 4.5% of wires); they may be the cut half of a longer wire.

Each wire's width is read from path_open_clean's mask along the wire. If that
mask misses the wire, the width is left empty.

### Which photos it measures

- **FEI SEM `.tif` files only.** The scale (nm per pixel) is read from the
  file, never from its name. `.jpg` copies have no scale, so they are not
  read. Files with "label" in the name (hand-label overlays) are skipped.
- **Only 50,000×–80,000× by default.** The method was checked only at those
  magnifications. Other photos are listed in `images.csv` as skipped, with
  their magnification. `--all-scales` measures them anyway; treat those
  numbers as rough. The test photos were all 50,000×; at 65,000–80,000× each
  quarter is smaller than the ones the piece check was fitted on.
- **Only photos taken looking straight down.** Photos taken with the stage
  tilted (recorded in the file; 18 of the 179 TiS₃ photos: 14 at 30°, 4 at
  45°) are skipped. Tilt shortens wires by up to 13% at 30° and 29% at 45°,
  and bends their angles. `--include-tilted` measures them anyway, with a note.
- **A photo that cannot be read** is listed as failed, and the run carries on.

## What you get

| File | What is in it |
|---|---|
| `images.csv` | one row per photo (columns below) |
| `wires.csv` | one row per wire (columns below) |
| `folders.csv` | one row per subfolder: the mean and SD of wires per µm², wire length per µm² and coverage over its photos; median length and width over its wires; share of wires in each direction |
| `overlays\` | each photo with its measured wires drawn in colour: **blue** 0°, **green** 60°, **red** 123.5°, **white** none of the three |
| `probs\` | the network's output per photo, reused by later runs. Each file name carries the photo's content hash, and the folder carries the network's, so a changed photo or network is never matched to old output. |
| `run_info.json` | date, input folder, method, the network's and piece check's file hashes, and every setting, so a run can be repeated |

### `images.csv`

| Column | Meaning |
|---|---|
| `image`, `folder` | the photo's file name, and its subfolder (`.` = the top folder) |
| `status` | `measured`, or why it was skipped or failed |
| `magnification` | from the scale stored in the file (a few file names are 2× off; the file wins) |
| `nm_per_px` | the photo's own pixel size. Every photo is measured at 5.4 nm per pixel (50,000×). |
| `tilt_deg` | stage tilt recorded in the file (0 = looking straight down) |
| `in_checked_range` | `yes` if 50,000×–80,000× |
| `width_um`, `height_um` | size of the area measured (information bar removed) |
| `wire_count` | wires found, including those cut off by the photo's edge |
| `wires_per_um2` | `wire_count` per µm² |
| `wire_length_um_per_um2` | total wire length (µm) per µm² of substrate |
| `coverage_pct` | % of the photo covered by wires, each drawn at its measured width (wires without a width at the photo's median width) |
| `whole_wire_count` | wires not cut by the photo's edge |
| `median_length_nm`, `mean_length_nm` | over the whole wires only. Wires cut by the edge would read short. |
| `median_width_nm` | median over the wires of each wire's mean width (wires without a width left out) |
| `share_0deg`, `share_60deg`, `share_123.5deg`, `share_other` | fraction of wires in each direction. 0° is horizontal; angles go clockwise on screen, so 60° wires run down to the right and 123.5° wires up to the right. |
| `junction_count` | places where wires touch or cross, including the corners of zig-zags (centre lines within 5 px). Compare it only between runs of this tool. |
| `notes` | e.g. the file name's magnification disagrees with the file's scale |

### `wires.csv`

| Column | Meaning |
|---|---|
| `image`, `folder`, `wire_id` | which photo, and the wire's number in it (from 0) |
| `length_nm` | along the wire |
| `width_nm` | mean width along the wire, from path_open_clean's mask. Empty when that mask misses the wire: about 12% of wires, mostly short ones (28% of wires under 81 nm), and more often on 15-min photos (18%). |
| `angle_deg` | 0–180°, same convention as above |
| `direction` | `0`, `60`, `123.5`, or `other` (more than 20° from all three) |
| `x_um`, `y_um` | the wire's midpoint, µm from the photo's top-left corner |
| `cut_by_edge` | `yes` if the wire runs off the photo, so its length is a lower bound |
| `junctions_crossed` | junctions the wire touches |
| `checks` | `short (under 81 nm)`, `width outside 16-162 nm`, or `ends at a quarter line (may be cut there)`. The wire is still counted. Filter these out if they shouldn't count for your question. |

## How accurate is it

**The fairest check is the bake-off's 16 held-out test quarters** (4 photos, 1401
hand-drawn wires, one straight line per wire, every wire drawn; see
`bakeoff/README.md`). No test photo was used for training, and every setting
was fitted on other photos. Caveats:

- The test quarters have been reused: 11 pipelines were scored on them, and
  the idea for the piece check came from studying the errors there.
- The piece check's random seed happened to be the best of 5 on the dev set.

So expect slightly worse numbers on new photos. The table scores the pieces of
each quarter (steps 1–4, before quarter-line joins) on the 50,000× test photos:

| Measure, on the 16 test quarters | `--method unet` | `--method path_open_clean` |
|---|---|---|
| wire F1 (wires found and matched to hand lines), pooled | **0.716** | 0.506 |
| wire count, pooled | **+2%** | +25% |
| wire count, mean of \|error\| per quarter | **18%** | 68% |
| median length, mean signed error | **−15%** (reads short) | −40% |
| total length, mean of \|error\| | **10%** | 16% |
| worst photo: 15-min photo 22, wire count | +37% | +211% |

The errors vary from photo to photo. Per test photo, `unet`'s wire count is
off by −16% to +37% and its median length reads 1% to 30% short (worst on the
15-min photo); per quarter the spread is wider. Differences of a few tens of
percent between single photos can be method error. Compare runs by their
means over many photos.

The TiS₃ folder's own photos are mostly the network's training photos (41 of
the 48 measured), so the test numbers above, not those photos, say how it
does on new ones.

`measure\check_accuracy.py` also runs the tool on the two hand-outlined 80,000×
photos (`results\measure\accuracy_unet.csv`; `--method path_open_clean` →
`accuracy.csv`). Read it with care:

- **It is a best case for both methods:** the network trained on these two
  photos, and path_open_clean's setting was tuned on them.
- **Its reference follows path_open_clean's own convention.** The hand
  outlines are traced with the same tracer (wire_extract) that path_open_clean
  uses. That tracer joins zig-zag arms into one bent wire (which then counts
  as `other`) and breaks wires where they cross, so its wires are much shorter
  than the hand-drawn lines. That is why path_open_clean looks better there
  on count (9% vs 22%), median length (3% vs 25%) and `share_other`
  (hand 14–18%, `unet` 0–3%). On the line-labelled test quarters above, the
  order is reversed.

| Measure on the 2 hand-outlined photos | `unet` | `path_open_clean` |
|---|---|---|
| median width | 46.0, 46.4 nm vs hand 44.8, 37.5 (**13%**) | 50.0, 53.7 nm (27%) |
| coverage | 21.8%, 14.8% vs hand 12.8%, 9.7% (+61%) | 20.7%, 15.2% (+59%) |
| junctions | 113, 85 vs hand 120, 120 (18%) | 139, 93 (19%) |
| shares of 0° / 60° / 123.5° | within 1–10 percentage points | within 0–11 |

The hand "junctions" are the tracer's graph junctions, a different definition
from the tool's contacts. The tool's count depends strongly on its 5 px
contact distance: hand photo 1 gives 43 / 98 / 114 / 150 junctions at 2 / 4 /
5 / 6 px.

What this means in practice:

- **Good for comparing and measuring samples:** wire density, wire length per
  µm², median length (reads ~15% short on average), direction shares.
- **Use only for comparisons, not as absolute values:** width (from
  path_open_clean's mask, reads wide), coverage (reads ~1.5–1.7× the hand
  area) and junction count (depends on the 5 px contact distance).
- **Compare photos taken at the same magnification.** Wires per µm² rises
  with magnification, because more short wires are resolved. Wire length per
  µm² changes much less, so prefer it when magnifications differ.
- **Weakest on sparse 15-min photos:** the network marks some clutter (etch
  pits, particle edges) as wire there. Test photo 22 still counts 37% too
  many wires.
- **Known misses:** dark rods, and some very faint wires.

## Other ways to find the wires

- `--method path_open_clean`: the previous default, a classical filter traced
  with wire_extract. It needs no PyTorch. It is less accurate (table above),
  but its numbers match earlier runs (`results\measure\TiS3 on sapphire\`).
- `--method v3_forest`: Nikhil's random-forest classifier (pipeline v3), kept
  for comparison (see `compare/RESULTS.md`).
- `--masks <folder>`: measure wire masks made by something else. One
  `<photo name>.png` per photo at the 50,000× scale, white = wire, traced with
  wire_extract.
- Rerun `check_accuracy.py` after any change.
