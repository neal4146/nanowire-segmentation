# bakeoff/: which model should segment the TiS₃ wires?

A head-to-head test of three trained networks against the best filter so far
(`path_open_clean`, from `compare/`). Every contender is scored the same way on
photos that no method trained or tuned on: quarters of four 50,000× photos,
labelled with one line per wire.

| | Contender | Trained on |
|---|---|---|
| **A** | `path_open_clean`, fixed production setting | nothing new: the bar to beat |
| **B** | nnU-Net v2 (2D, its own defaults) | agreement pseudo-labels + the 2 hand-labelled photos |
| **C** | nnU-Net v2 | B + copy-paste hard cases |
| **D** | U-Net with an ImageNet-pretrained ResNet-34 encoder | the same data as B (`D_cp`: as C) |
| **E** | D's U-Net + skeleton recall loss | C's data with the three fixes for faint wires (see "Contender E" below) |

Nikhil's production forest (`v3_production`) is scored too, as a reference.

**Why these four.** nnU-Net is the strongest default for thin structures when
there is plenty of data (it beat every other method by 10+ points on the
OmniCrack30k crack benchmark). With very few labels, a pretrained encoder has
done better (ESRVS, 2026: one labelled image, Dice 0.80 vs 0.71 for a plain
U-Net). Here there are only 2 hand-labelled photos but ~40 pseudo-labelled
ones, so neither result settles it; this test does.

## The data

- **Photos.** Only 49 of the 225 TiS₃ photos are at 50k–80k, the scale
  band where the filters were tuned and checked. 4 of them are the test photos
  (`test_photos.txt`), 2 are raw copies of the hand-labelled photos, and the
  other **41 get pseudo-labels**. Everything is at the 50k-equivalent scale
  (5.4 nm/px).
- **Agreement pseudo-labels** (`pseudo_labels.py`). A pixel is labelled *wire*
  only where `path_open_clean` and the v3 forest both say wire, and
  *background* only where both say background (1 px clear of
  `path_open_clean`'s band). Everything else, about 12% of each photo, is
  **ignored**: no loss there, so the networks never copy either method's
  systematic mistakes (fused parallel wires, short fragments, round
  particles). Both methods are used at their production cut-offs.
- **Hand labels.** The two 80k photos with their filled hand outlines, as
  in `compare/`. nnU-Net samples photos uniformly, so each one is used 8 times
  (`--hand-copies`).
- **Copy-paste hard cases** (`copy_paste.py`, contender C). There are 92 thin,
  straight wires cut from the hand-labelled photos. Each is a real hand outline
  plus a 2 px rim, trimmed 3 px short of any wire it touched. They are pasted
  onto clean background in each pseudo-labelled photo as **close parallel
  pairs** (2–8 px gap, labelled background) and **crossings** (both wires whole),
  about 10 pairs and 8 crossings per photo. These are exactly the cases the
  ignore regions leave unlabelled. Wires are never rotated, and brightness and
  contrast are matched to the destination.

## Rules that keep the test honest

1. **Test photos never train anything.** `pseudo_labels.py` leaves them out,
   `export_nnunet.py` refuses to run if one is in the training data, and
   `score_bakeoff.py` skips any labelled photo not listed in `test_photos.txt`
   or found in the training manifest. Edit `test_photos.txt` only before step 1.
2. **Label from scratch.** Draw every line yourself. Lines pre-drawn by a
   method would favour that method.
3. **No threshold is tuned on the test photos.** nnU-Net's mask is its argmax
   and the U-Net's is p > 0.5; A and v3 use their production settings.
4. **A win needs its interval above zero.** `summary.csv` gives each
   contender's wire-F1 difference to A with a paired 95% interval.

## Running it

Run everything from the repo root. The CPU steps use the main `venv`; the GPU
steps use `venv_dl` (PyTorch, nnU-Net, segmentation_models_pytorch).

### 0. Label the test quarters (laptop)

```
cd ..\Materials_ThangPham\03_Code\labelling
..\..\..\labelling_env\Scripts\python label_wires.py --photo SF251213M_50000x_37 --quarter tl
```

Use any photo in `test_photos.txt`, quarters `tl tr bl br`. Aim for about 8
quarters, at least two per growth run. Each takes 30–45 minutes.

### 1. Training data (laptop or ARC, ~10 min)

```
venv/Scripts/python bakeoff/pseudo_labels.py --previews   # 41 photos, ~10 s each
venv/Scripts/python bakeoff/copy_paste.py                 # ~1 min
venv/Scripts/python bakeoff/export_nnunet.py              # nnU-Net datasets 601 (B), 602 (C) + test images
```

Check a few `results/bakeoff/train/previews/` and `copypaste/previews/`:
green is wire, magenta is ignored, cyan is pasted. On ARC,
`sbatch slurm/bakeoff_prep.sbatch` runs all three. It reads the photos from
`data/real/TiS3 on sapphire/`, a copy of `02_Data/SEM images/TiS3 on sapphire/`
(set `$BAKEOFF_PHOTOS` to use another folder). Alternatively, build the data on
the laptop and copy `results/bakeoff/` to the same place in the repo on ARC.

### 2. Train and predict (ARC)

Once, on a login node (it has internet; it also caches the encoder weights):

```
bash bakeoff/setup_dl_env.sh
```

Then:

```
sbatch slurm/bakeoff_nnunet.sbatch B
sbatch slurm/bakeoff_nnunet.sbatch C
sbatch slurm/bakeoff_unet.sbatch D
sbatch slurm/bakeoff_unet.sbatch D_cp --copy-paste
```

Each job trains and then writes its masks for the test photos to
`results/bakeoff/pred/<name>/`. nnU-Net's default 1000 epochs take about a
day or more on a T4. Every job resumes when you resubmit it, so a time limit
only means submitting the same command again. For a quick first look, use
`TRAINER=nnUNetTrainer_250epochs sbatch slurm/bakeoff_nnunet.sbatch B`; its
predictions go to `pred/B_250epochs/`.

On the laptop (RTX 4070, `venv_dl/Scripts/...`), D trains fine: a test run
went at 17 steps/s with smaller batches, so the default 20,000 steps should
take roughly an hour. nnU-Net does not: one epoch took
13 minutes, because on Windows its data loading leaves the GPU 90% idle. Run B
and C on ARC.

### 3. Score (laptop, where the labels are)

Copy `results/bakeoff/pred/` back from ARC, then:

```
venv/Scripts/python bakeoff/score_bakeoff.py
```

It scores every folder in `pred/` plus A and v3 on every labelled test
quarter. It writes:

| File | Contents |
|---|---|
| `results/bakeoff/summary.csv` | pooled pixel F1 and whole-wire F1 with 95% intervals, count / length error, junction recall, the difference to A (`d_wire_f1_vs_A`, its interval, `p_better_wire_vs_A`), wire F1 per growth run |
| `results/bakeoff/scores_per_quarter.csv` | every contender on every quarter |
| `results/bakeoff/overlays/` | green = correct, orange = false alarm, blue = missed |

## Files

| File | What it does |
|---|---|
| `common.py` | paths, the test list, photo discovery and loading (production's `load_frame_50k`) |
| `test_photos.txt` | the four photos kept out of training, one or two per growth run |
| `pseudo_labels.py` | agreement pseudo-labels and the hand-labelled photos → `results/bakeoff/train/` |
| `copy_paste.py` | parallel pairs and crossings pasted from hand-labelled wires → `results/bakeoff/copypaste/` |
| `export_nnunet.py` | nnU-Net datasets 601 (B) and 602 (C), ignore label 2; test images → `results/bakeoff/test_images/` |
| `train_unet.py` | contender D: U-Net with a pretrained encoder. It mirrors nnU-Net's data mix, z-scoring and Dice + CE loss; `--skeleton-recall W` adds the Skeleton Recall loss; `--data E` trains on E's data |
| `build_e_data.py` | contender E's data: faint ignore + hand fixes → `results/bakeoff/train_E/`, then faint paste → `copypaste_E/` |
| `holdout_E.sh` | E's recipe trained twice, each without one hand-labelled photo (sprites only from the other), predicted on it: only to choose E's mask setting |
| `thin_masks.py` | `choose`: threshold / probability-ridge settings scored on the two held-out photos, best by wire F1; `apply`: that one setting on E's test probabilities → `pred/E_thin/` |
| `../../Materials_ThangPham/03_Code/labelling/correct_training.py` | napari tool to add missed wires to TRAINING photos (refuses test photos) → `labelling/train_fixes/` |
| `predict.py` | masks for the test photos from a trained nnU-Net (`nnunet`) or U-Net (`unet`) |
| `score_bakeoff.py` | the scores, intervals and overlays above |
| `holdout_check.py`, `holdout_check.sh` | the early check below |
| `setup_dl_env.sh`, `requirements-dl.txt` | builds `venv_dl` |

## Contender E: fixing what the pseudo-labels teach wrongly (2026-11-18)

The early check below found D mostly learns A. The cause is in the labels: A
needs a 25 px straight DARK run, so faint wires, short rods (the test labels
have 5-8 px ones) and the bodies of bright rods all become agreed
*background*, and B, C and D are trained to call them background. E keeps
D's network and fixes the data three ways:

1. **Faint ignore** (`pseudo_labels.faint_candidates`). Dark and bright ridge
   maps opened with a 7 px line (not 25), cut at each photo's own noise level
   (99.5th percentile of its background); line-shaped components (3:1 or
   longer, 6+ px with a pixel above the 99.9th percentile, or 12+ px) turn
   their background into *ignore*. Mean ignore goes from 11.8% to 14.3%.
2. **Faint paste** (`copy_paste.py --faint`). 15 hand-drawn wires per photo
   pasted at 2-3.5 noise units of contrast (40% cut to 6-20 px short rods,
   +1 unit), labelled *wire*, on top of C's pairs and crossings.
3. **Hand fixes** (`correct_training.py`). On 8 suggested training quarters,
   draw lines only for wires the green automatic label misses, and 'not wire'
   lines over false ones. Added lines become a 3 px wire core in a 13 px
   ignore band. Those photos are drawn 3x as often (`--corrected-copies`).
   With `--complete` every wire in the quarter is drawn, and the lines
   replace the automatic label there (background except the lines, so the
   pseudo-labels' false wires are fixed too; automatic wire within 12 px of
   a line is ignored; red = unsure, ignored). Quarters 27 tl, 31 br and 5 tr
   are add-only; the rest complete. Check image per complete quarter:
   `train_E/previews/<photo>_<quarter>_complete.png` (red = automatic wire
   that is now background: should be blobs only, never a skipped wire).

Test photos never enter: `correct_training.py` and `build_e_data.py` refuse
them, and `score_bakeoff.py` also skips any photo in `train_E/manifest.json`.

```
..\Materials_ThangPham\03_Code\labelling: ..\..\..\labelling_env\Scripts\python correct_training.py --list
venv/Scripts/python bakeoff/build_e_data.py          # ~2 min; rerun after every hand fix
venv_dl/Scripts/python bakeoff/train_unet.py --name E --data E --copy-paste --skeleton-recall 1.0
venv_dl/Scripts/python bakeoff/predict.py unet --contender E --model results/bakeoff/models/E/final.pt
```

E trains on the laptop in about an hour (as D). On ARC, copy `results/bakeoff/train_E/` and
`copypaste_E/`, then `sbatch slurm/bakeoff_unet.sbatch E --data E --copy-paste --skeleton-recall 1.0`.
Look at **wire recall** first: that is where faint wires show up.

### E_thin: E's probabilities as a thinner mask (2026-11-19)

E's masks at prob > 0.5 are wide enough to fuse close parallel wires on the
dense photo. `holdout_E.sh` trains E's recipe twice, each without one
hand-labelled photo (copy-paste sprites only from the other), and
`thin_masks.py choose` scores 11 settings fixed in advance on the held-out
photos. Winner: **ridge_0.5_1** (the probability's across-wire maximum where
prob > 0.5, widened by 1 px): held-out wire F1 0.616 vs 0.589 for E as
scored; thr_0.9 0.610 was next; ridge with r 2 or 3 was far worse (0.47-0.50).
All scores: `results/bakeoff/holdout/thin_scores.csv`. `thin_masks.py apply`
writes `pred/E_thin/`, scored once on the test quarters:

| contender | wire F1 [95%] | vs A [95%] | pixel F1 | count err | junction recall |
|---|---|---|---|---|---|
| E_thin | 0.570 [0.541, 0.598] | +0.064 [+0.040, +0.087] | 0.625 | 34% | 0.30 |
| E | 0.529 [0.503, 0.554] | +0.023 [+0.000, +0.045] | 0.726 | 27% | 0.15 |
| A | 0.506 [0.481, 0.531] | – | 0.708 | 68% | 0.22 |

E_thin vs E (paired, same tile resamples): +0.041 [+0.017, +0.065]. Better
than A on 14/16 quarters and than E on 12/16; worse than E on the sparse
15-min photo 22 (3 of 4 quarters). Pixel F1 falls because the ~3 px mask
covers less of the 8 px label band; it is not the ranking metric.

## Dev set and crossing-aware extraction (2026-11-19)

The error deep dive (`results/bakeoff/deepdive_2026-11-19/`) found that E_thin's
mask reaches almost every wire (17 of 1401 never segmented); most of what is
lost happens in `wire_extract`. Wires touching end to end (V, L) skeletonise
with no branch point and are never split, and `--min-area 40` deletes wires
shorter than ~13 px in a 3 px mask. The GT lines themselves, as perfect 3 px
masks, score only wire F1 0.81 on test and 0.79 on the dev quarters.

**Dev set** (`dev_cv.py`, `dev_cv.sh`): the 5 complete-mode hand-fixed
training quarters (696 lines, drawn like the test labels). E's recipe is
trained twice, each run without one fold of those photos and their
copy-paste copies (fold A: 48, 20; fold B: 65, 70, 35, fixed before
training), and each quarter is predicted by the model that did not see it.
The copy-paste sprites come only from the 80k hand photos, so nothing leaks.

**`extract_plus.py`** wraps the unchanged `wire_extract`. The candidate grid
(22 settings, file sha256 `0c71f5a2…`) was frozen on 2026-11-19 at 03:25,
before any fold prediction existed:
`prod` (unchanged); `rdp` (split chains at RDP turns > 30/45°, not next to
junction bridges, min_area 40/20, rejoin 0/8 px); `arm` (split where the
10 px of path on either side differ by > 30/45°, rejoin 0/8); `fam` (split the
mask into the three growth families 0/60/123.5° by directional openings with
11 or 15 px lines, plus a residual, production extraction on each, minimum
piece 0/10 px); `lsd` (OpenCV line-segment detector on the blurred mask, moved
onto the ridge, duplicates suppressed, collinear rejoin, minimum piece
10/15/20/25/30 px). Rule: best pooled dev wire F1 for E_thin's recipe; ties
within 0.002 → smaller mean |count error| → simpler mode → grid order;
adopted only if it beats `prod` by ≥ 0.010 on dev, else nothing is scored.

This grid replaced a first one (min_area 10/20/40 × 30/45° RDP splits) after
dev-only evidence: perfect dev masks, A's real dev masks and the older 80k
hold-out E masks. min_area 10 added false wires and a 30° split over-split
real skeletons, while LSD and the family split did better on thin real
masks. Perfect dev masks: prod 0.794, best rdp 0.905, arm 0.904, fam 0.923,
lsd_m10 0.968. A's real dev masks: prod 0.498, arm_t45 0.522, lsd_m20 0.598.

**Honest framing.** The idea for these fixes came from the error analysis of
the TEST quarters (handover section 15); only the settings are chosen on dev.
So the +X rows are not a fully independent test. The dev set has no 15-min
photo: read the 15-min column of the +X rows separately.

**Declared test batch** (scored once, after `choose`): the chosen setting on
A, v3, E and E_thin (4 new test variants, named `<contender>+X`), next to the
same four with the production extraction. **Primary comparison, declared
before test: E_thin+X vs E_thin** (paired interval). A+X, v3+X and E+X are
secondary and cannot be promoted to the leader on this run. Output:
`results/bakeoff/summary_plus.csv` (paired intervals vs A, vs A+X and vs the
contender's own production row, per run and per photo) and
`scores_per_quarter_plus.csv`, both stamped with the code and choice hashes.
`test` refuses to run twice; `choose` refuses to run after `test`.
`summary.csv` is not touched.

**Result (2026-11-19).** Dev pick: `lsd_m20`. E_thin's dev wire F1 goes from 0.634 to 0.734
(+0.100 [+0.063, +0.139]), better on all 5 dev quarters. Test, scored once:

| contender | wire F1 [95%] | vs own prod | vs A |
|---|---|---|---|
| **E_thin+X** | **0.639 [0.611, 0.665]** | **+0.069 [+0.046, +0.092]** | **+0.132 [+0.105, +0.158]** |
| E+X | 0.591 [0.565, 0.616] | +0.062 | +0.085 |
| A+X | 0.554 [0.528, 0.578] | +0.048 | +0.048 |
| v3+X | 0.429 [0.405, 0.451] | −0.026 | −0.078 |

E_thin+X beats E_thin and A on every test photo (whole-photo bootstrap
vs E_thin [+0.051, +0.092], vs A [+0.061, +0.220]). Mean |median-length error|
per quarter falls from 32% to 20%, and pooled count error from +20% to +13%.

**Labelling convention confirmed (user, 2026-11-19):** a zig-zag is several
nanowires, not one bent wire. So splitting at corners is physically right,
and production `wire_extract` was wrong to fuse them.

### Round 2 (2026-11-19, started ~11:30): training variants + piece verifier

- **R**: E's recipe with rotation augmentation ±25° (was ±5°; steep wires had
  recall 0.47). `bash bakeoff/dev_cv_variant.sh R --data E --rot-deg 25`.
- **RB** (R + background on wire flanks, `build_rb_data.py`): **rejected before
  training**. Simulated inside the 5 complete-mode quarters, 11.1% of the
  pixels it would make background lie within 2.5 px of a hand-drawn wire
  (limit 5%; 22-25% in two quarters). These are real wires that neither A nor v3
  found, next to ones they did. The rule was not loosened to pass, because
  that would mean tuning it on the dev quarters.
- **Verifier V**: a per-piece check on the LSD pieces (keep confident short
  pieces, drop weak long ones), designed on the dev quarters with fold
  cross-validation.
- **Selection, declared before any round-2 dev score:** pooled dev wire F1
  (cross-validated for V) over {E, R} × {lsd_m20, lsd + V}. A change is adopted
  only if it beats E + lsd_m20 (0.734) by ≥ 0.010 on dev. If R wins, R is
  retrained on all training data (as E was) before test.
- **Final test batch, scored once:** the adopted pipeline. **Primary
  comparison: adopted pipeline vs E_thin+X** (0.639, `summary_plus.csv`), with
  the paired tile-bootstrap and whole-photo intervals. If nothing beats
  E + lsd_m20 on dev, nothing new is scored on test.

**Dev result (13:23), pooled wire F1 over the 5 dev quarters** (V cross-validated
over the folds, primary V = GradientBoosting, label y_match, nested threshold):

| network | lsd_m20 | lsd + V | V vs lsd_m20 |
|---|---|---|---|
| E | 0.734 | **0.796** | +0.062 [+0.041, +0.083] |
| R (rot ±25°) | 0.713 | 0.742 | +0.028 [+0.009, +0.046] |

So V is adopted (+0.062 over E + lsd_m20, past the +0.010 bar). R is rejected,
and E stays the network. R's ±25° rotation lowers dev F1 by 0.021 with plain
LSD and by 0.054 with V.

**E2, decided ~13:25 and written down 13:27-13:30, before any E2 dev or test
number existed.** E2_cvA had started training at 13:26:48 and had written only
training-loss lines by then. After the dev result, this replaced "the adopted
pipeline" (E_thin+V) as the primary, under the guard below. E2 is E's recipe and
data plus the user's 2 new complete-mode quarters, drawn after round 1:
SF251211N_50000x_59 br (15 min, 12 lines) and SF251213M_50000x_38 bl (60 min,
89 lines + 5 unsure). Adding hand labels is a data decision, not a dev
comparison. E_cv's fold models trained on those photos' pseudo-labels, so they
have no fair score on the new quarters.
- The dev set grows to 7 quarters: 59 br joins fold A, 38 bl joins fold B
  (`dev_cv.py prepare`). Retrain E2_cvA/B
  (`dev_cv_variant.sh E2 --data E` on the rebuilt train_E). Refit V on their
  7-quarter predictions by the same protocol (`PV_MODEL=E2_cv`
  prepare/build_table/train_cv/final_fit), then train the final E2 on all
  data. The round-1 train_E/copypaste_E are kept as `*_r1`.
- Fallback guard: if E2_cv + V (CV, pooled over the original 5 quarters
  only) falls more than 0.020 below E_cv + V (0.796), the primary becomes
  E_thin+V, i.e. the round-1 E model with the V fitted on E_cv.
- **Final batch, scored once** (`final_round2.py`): primary **E2_thin+V vs
  E_thin+X**. The batch also scores E2_thin+X (data effect alone) and E_thin+V
  (V alone, on the round-1 E). The references E_thin+X and A are scored too.

```
bash bakeoff/round2_e2.sh > results/bakeoff/logs/round2_e2.log 2>&1   # E2_cvA/B, E2, test probs (~3-5 h)
PV_MODEL=E2_cv venv/Scripts/python -B bakeoff/verifier/prepare.py     # then build_table, train_cv, final_fit
venv/Scripts/python bakeoff/round2_choose.py                           # applies the guard -> devcv/round2_choice.json
venv/Scripts/python bakeoff/final_round2.py results/bakeoff/devcv/round2_choice.json   # once
```
`final_round2.py` was smoke-tested on the 5 dev quarters with patched inputs,
reading no test labels. It reproduced E_thin+X at 0.732 (0.734 with unsure
lines dropped) and A at 0.495 (0.498).

**E2 dev (16:17):** E2_cv + V scores 0.769 pooled over the 7 quarters, +0.058
[+0.040, +0.076] over lsd_m20 (0.710), and 0.7956 on the original 5. That
passes the guard against E_cv + V's 0.7958, so the primary is E2_thin+V. The
new 15-min quarter 59 br is poor (F1 0.12). The drawn wires are under the mask
(79% of line pixels within 4 px), but only 22% of the mask lies near a wire:
the network marks clutter in sparse 15-min images.

**Round 2 test result (16:19, scored once, `summary_round2.csv`):**

| pipeline | wire F1 | vs E_thin+X (tile bootstrap) | photo 60 / 37 / 15 / 22 | count err pooled | mean abs median-length err |
|---|---|---|---|---|---|
| **E2_thin+V** (primary) | **0.716** [0.689, 0.740] | **+0.077** [+0.059, +0.096] | 0.82 / 0.77 / 0.65 / 0.55 | +2% | 16% |
| E_thin+V | 0.714 [0.688, 0.738] | +0.075 [+0.063, +0.088] | 0.84 / 0.78 / 0.64 / 0.51 | +9% | 21% |
| E_thin+X (round 1) | 0.639 [0.611, 0.665] | – | 0.77 / 0.72 / 0.55 / 0.48 | +13% | 20% |
| E2_thin+X | 0.633 | −0.005 [−0.022, +0.011] | 0.73 / 0.72 / 0.55 / 0.49 | +14% | 21% |
| A | 0.506 | −0.132 | 0.58 / 0.51 / 0.50 / 0.29 | +25% | 40% |

- **The gain is the verifier V, not E2's extra labels.** E2_thin+V beats
  E_thin+V by only +0.0015 and wins 8 of 16 quarters. The new quarters help
  the 15-min photo 22 (+0.038; count error +54% → +37%) and cost a little on
  photos 60 and 37. V was +0.062 on dev and gives +0.075 on test.
- **Every test photo improves** (+0.051 to +0.105), as do 15 of 16 quarters
  (the worst is 22 tr at −0.029). With 4 photos from 3 runs, the
  whole-photo bootstrap [+0.051, +0.101] is just the range of the
  per-photo gains. It is not a calibrated 95% interval for new photos.
- **Not a fully independent test.** The ideas behind R, RB and V came from
  the test-set error deep dive (round 1's "honest framing" applies again).
  Every setting was fitted on dev only, and the primary was declared first.
  11 distinct pipelines have now been scored on these 16 quarters: A, v3, E,
  E_thin, each also +X, and the 3 round-2 rows.
- The intervals cover one trained network and one fitted verifier. The
  verifier's dev CV F1 varies 0.737–0.769 over seeds 0–4. Seed 0 was fixed in
  advance and happens to be the highest.
- The `wire_recall` and `wire_precision` columns are quarter means, and the
  `d_vs_*` columns are bootstrap means. Round 1 used the same convention.
- An independent 4-agent check (leakage, protocol timeline, scoring, code)
  found nothing fatal or major. The reference rows reproduce
  `scores_per_quarter_plus.csv` exactly, per quarter. make_mask on pred/E
  reproduces pred/E_thin pixel for pixel. Each dev prediction comes from the
  fold that held its photo out. Neither verifier saw a test quarter.
  CP_SF251213M_50000x_39_0 also changed in the rebuild, because copy_paste.py
  was edited on 18 Nov after E trained. The effect is negligible.

```
venv/Scripts/python bakeoff/dev_cv.py prepare
bash bakeoff/dev_cv.sh                                  # ~2 h, laptop GPU
venv/Scripts/python bakeoff/extract_plus.py ceiling     # optional
venv/Scripts/python bakeoff/extract_plus.py choose      # dev only
venv/Scripts/python bakeoff/extract_plus.py test        # once
```

## Early check: D on the hand-labelled photos (2026-11-05)

Before any test labels existed, `holdout_check.sh` trained D twice on the
laptop (RTX 4070, 20k steps, ~57 min each). Each run left out one of the two
hand-labelled 80k photos, and that run was scored on the photo it left out.
The scorer and bootstrap are the same as in `compare/`.

| contender | pixel F1 | wire F1 | marked / labelled | count err | length err | width err | wire F1 vs A |
|---|---|---|---|---|---|---|---|
| A path_open_clean | 0.627 | 0.544 | 1.59 | 9% | 6% | 27% | – |
| v3 forest (trained on the other photo) | 0.569 | 0.442 | 1.15 | 4% | 28% | 38% | −0.112 [−0.160, −0.066] |
| D U-Net | 0.651 | 0.567 | 1.60 | 12% | 7% | 43% | +0.029 [−0.002, +0.061], P>A 0.97 |

- D is a little better than A, but the 95% interval just touches zero. Its
  masks overlap A's with IoU 0.87 on both photos, so mostly it has learned A.
- D misses the same faint wires A misses (the long blue wire in
  `results/bakeoff/holdout/overlays/TiS3_80000x_2_*.png`). It never saw those
  wires in training, because the agreement pseudo-labels call them background.
- D marks wires wider than A does: width error 43% vs 27%.
- The score is slightly optimistic. The pseudo-labels came from A (tuned on
  both photos) and the forest (trained on both).

What this means: pseudo-labels from A cap D at about A's level. To get past
it, a network needs labels A cannot make: the test labels, or hand-corrected
pseudo-labels on a few photos where A misses wires.

## Caveats

- The pseudo-labels inherit what *both* filters miss. Faint wires that neither
  marks are labelled background. The test labels will show whether that costs
  recall.
- Copy-paste wires all come from the two 80k photos of the 30-min run. On the
  15-min photos, whose wires are longer and darker, they may look less real.
- Whole-wire F1 runs every mask through Nikhil's `wire_extract`, exactly as
  in `compare/`. A network that separates close wires only gets credit if
  `wire_extract` keeps them apart too.
- The intervals come from resampling tiles within the labelled quarters. They
  show how sure the ranking is on these quarters, not how it would change on
  a new growth run.
