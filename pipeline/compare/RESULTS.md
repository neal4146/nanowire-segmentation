# Filter race on the two labelled TiS₃ photos (20 Oct 2026)

All rows went through `compare/` with the same ground truth and the same wire step.
Every setting was chosen on one labelled photo and scored on the other, then
the photos swapped. The table shows test-photo means. Full numbers are in
`results/compare/results.csv`; 95% intervals are in `results/compare/bootstrap.csv`.

**What the scores are measured against.** The same answer key Nikhil uses: the two
hand-labelled photos, read with his `gt_labels.py`. *Pixel F1* compares marked pixels
with the filled hand outlines, using his `tune_pseudo_mask.score()`. *Wire F1* compares
the wires his `wire_extract` traces in each mask with the wires it traces in the
hand-labelled mask, the same reference as `--validate-gt`. The errors compare count,
length and width from that same step.

**Scorer check.** Sato v1 reproduces the README exactly (recall 0.517, precision 0.484,
F1 0.500), and so does v2 (0.570 / 0.505 / 0.534). RF v3, trained on one photo and
tested on the other, gives 0.614 / 0.550 / 0.569 (README 0.617 / 0.550 / 0.571).

| Method | Settings chosen by | Pixel F1 | Marked ÷ labelled | Wire F1 (95% interval) | Count error | Median length error | Width error* | Junction recall |
|---|---|---|---|---|---|---|---|---|
| **Path openings, cleaned** (Sato σ 1–2 + 25 px line opening + closing) | best pixel F1 | **0.625** | 1.58 | **0.547** (0.51–0.61) | 9% | 9% | 27% | 0.42 |
| **Path openings, cleaned** | best wire F1 | 0.591 | 1.26 | 0.534 (0.50–0.59) | 6% | **6%** | 32% | 0.40 |
| Path openings (Sato σ 1–3 + line opening + closing) | best pixel F1 | 0.612 | 1.66 | 0.530 (0.48–0.59) | 6% | 12% | 26% | 0.41 |
| Forest + path-opening inputs | best wire F1 | 0.600 | 1.31 | 0.520 (0.47–0.58) | **3%** | 22% | 33% | 0.42 |
| Sato dark, v1 | Nikhil's (thr 0.10) | 0.500 | 1.07 | 0.498 (0.46–0.55) | 10% | 12% | 26% | 0.36 |
| Sato dark + closing, v2 | Nikhil's (thr 0.14) | 0.534 | 1.13 | 0.480 (0.43–0.54) | 9% | 14% | 27% | 0.33 |
| Sato dark + closing, v2 | best wire F1 | 0.583 | 1.42 | 0.498 (0.46–0.55) | 8% | 20% | 29% | 0.41 |
| Sato, dark + bright | best wire F1 | 0.513 | 1.15 | 0.500 (0.46–0.55) | 6% | 9% | 30% | 0.32 |
| Frangi dark | best pixel F1 | 0.450 | 0.79 | 0.479 (0.43–0.55) | 15% | 24% | 12% | 0.39 |
| Forest, then path opening | best pixel F1 | 0.565 | 1.09 | 0.486 (0.44–0.54) | 1% | 21% | 32% | 0.40 |
| Random forest, v3 | Nikhil's (p > 0.70) | 0.569 | 1.15 | 0.442 (0.39–0.50) | 4% | 28% | 38% | 0.38 |
| Local std (high) | best pixel F1 | 0.549 | 1.92 | 0.374 (0.34–0.43) | 41% | 46% | 50% | 0.38 |
| Steger line detector | best pixel F1 | 0.490 | 1.29 | 0.335 (0.29–0.38) | 40% | 36% | 17% | 0.29 |
| Sato + line fitting (review E1b) | best pixel F1 | 0.462 | 1.51 | 0.299 (0.25–0.34) | 23% | 28% | n/a | 0.15 |
| Sato bright | best pixel F1 | 0.246 | 0.71 | 0.225 (0.17–0.25) | 35% | 43% | 46% | 0.16 |

\*Width error is against the width the wire step measures on the hand outlines.
That is the wire's *outer* width (see "How wide are the wires" below). Line fitting
draws lines of a fixed width, so it has no width measurement. The 95% intervals
come from resampling 160 px tiles of the test photos (paired across methods). They
cover variation within these two photos only. Both photos come from the same
30-minute growth run, so the intervals cannot show how results change between
growth runs.

## What the race says

1. **Cleaned path openings is the best method, and the first to beat Sato v1 on whole
   wires with the whole interval above zero.** Tuned for pixel F1, it is ahead of Sato v1
   by +0.050 wire F1 (interval +0.008 to +0.090; ahead in 99% of resamples) and +0.128
   pixel F1. It is ahead of the forest by +0.116 wire F1 (+0.069 to +0.165).
   - **What it does:** Sato on dark lines with sigmas 1–2 (narrower than v1's 1–3), a
     grayscale opening with 25 px straight lines along the three growth directions
     (a pixel stays only if a straight run through it is bright too), then Nikhil's
     v2 directional closing. Of the 6 tuning choices (3 protocols × 2 photos), all picked
     sigmas 1–2 and 4 picked 25 px lines (the others 17 px).
   - **The other options matter less.** Removing bright particles was picked in 2 of 6
     choices and 1 px thinning in 3 of 6; on their own they did not raise the score in a
     first test. Removing particles costs labelled pixels, because some bright compact shapes are
     outlined as wires in the hand labels. Whether those are wires is a question for
     Dr. Chen.
   - **Weak spots:** tuned for pixel F1 it marks 1.6× the labelled pixels. Tuned for
     wire F1 it marks 1.3×, has the lowest length error (6%), and stays ahead of v1 in
     93% of resamples (+0.032, interval −0.010 to +0.074).
2. **Path openings also fixes the forest's main weakness.** With the two path-opening
   maps as extra inputs, the forest improves by +0.083 wire F1 (interval +0.047 to
   +0.120) and keeps its 3% count error. The 25 px opening became one of its top-3
   inputs on one tuning photo. It is still slightly behind the cleaned filter on whole wires.
3. **The plain forest wins on wire count, but loses on whole wires.** Against Sato v1
   it is +0.068 pixel F1 and −0.066 wire F1 (−0.123 to −0.014), and it makes wires
   about 28% shorter than the reference.
4. **Sato v1, v2, both-polarity Sato and Frangi are statistically tied on whole wires**
   in the settings shown (differences within ±0.05, intervals overlap zero).
5. **Steger, line fitting and bright Sato are clearly worse** (intervals entirely below
   Sato v1). For line fitting, the port gives identical output to the review's
   original. A wider search and full 80k resolution were also tried; neither brings it
   level with Sato v1 on unseen photos.
6. **Local std works, but in the opposite direction to the README.** Wire pixels have about 2× the
   background's local variation (median 0.050 vs 0.022, σ = 2).

## How wide are the wires? (`width_check.py`)

On two 200,000× photos of the same 60-minute run (1.35 nm/px), 20 straight wire
stretches were picked by hand. 19 passed the consistency check; one had a
particle on it. A wire there is a body between two dark edge lines. Averaged across-wire
profiles give:

| Width measure | Median | Middle half |
|---|---|---|
| Edge to edge (between the two dark edge lines) | **29 nm** | 26–31 nm |
| Outer (including the dark edges) | **45 nm** | 41–50 nm |

What the methods report at 50k, on the eight in-band 50k photos of the same run:

| Source | Median width |
|---|---|
| Hand outlines (80k photos, 30-min run) | 45 / 38 nm |
| Nikhil's v3 pipeline | **47.6 nm**, close to the outer width |
| Steger | 18.9 nm, below both widths |

So the hand outlines are **not twice too wide**: they trace the outer width, and the v3
pipeline measures that same outer width to within about 7%. Which of the two is the
physical width (sidewall to sidewall, or including the dark edge contrast) can't be
decided from SEM images alone. **Ask Dr. Chen / Dr. Pham which one the lab wants.**
Steger needs a calibration factor (~1.5× for edge-to-edge) before its widths mean
anything. Each profile picture with its marked edges is in
`results/compare/width_check/profiles_*.png`.

## On the real photos (`run_production.py`)

The winner was run through the full production pipeline: `load_frame_50k`, then the
mask, then Nikhil's unchanged `wire_extract`. It ran on the eight in-band 50k photos of
the 60-minute run, with one fixed setting (Sato σ 1–2, 25 px opening, closing, thr 0.0566).
That setting scores pixel F1 0.627 and wire F1 0.544 on the labelled photos
(cross-frame). Per-photo numbers are in `results/compare/production_vs_v3.csv`;
side-by-side pictures are in `results/compare/v3_vs_new_*.png`.

| | v3 (Nikhil) | Path openings, cleaned |
|---|---|---|
| Wires, 8 photos | 4,103 | 3,647 (−11%) |
| Median wire length (median over photos) | 173 nm | 211 nm (+22%) |
| Median width | 45–51 nm | 39–51 nm |
| Mask overlap with v3 (Dice) | – | 0.72–0.77 |
| Whole wires found by both (F1) | – | 0.37–0.56 |

- **Longer wires and lower counts point the right way.** On the labelled photos v3 made
  wires ~28% too short, while the new method was within ~9%.
- **Wires only v3 finds** (2,182) are mostly short: a median 64–98 nm, against 260–400 nm
  for shared wires, and they are no fainter. Most are v3's fragments and the round
  particles it marks as wires, plus some genuinely short thin wires the new method misses.
- **Wires only the new method finds** (1,726) are fainter, medium-length wires (100–170 nm)
  that v3 misses.
- **Dense photo (_15):** where several parallel wires run close together, the new method
  fuses them into one wide band (coverage 26% vs v3's 21%). This is its over-marking
  showing up as undercounting.
- **Which method is right on the disputed wires can only be decided with labels on these
  photos** (next section).

## Testing on a new photo (`external.py`)

The tools to label a 50k photo from the 60-minute run are in
`Materials_ThangPham/labelling/` (see `HOW_TO_LABEL.md` there). After labelling,
`venv/Scripts/python compare/external.py` scores every method on that photo, using
only settings learned on the two 80k photos. This shows whether the ranking above
holds on the photos the pipeline actually measures.

## Caveats

- Only two labelled photos, both from one growth run. The labelling tools above
  exist to fix this.
- Tuning by wire F1 on one photo often transfers worse than a fixed threshold
  (Sato v1: 0.441 vs 0.498). With two photos, a careful single setting beats
  aggressive tuning.
- Wire F1 is measured against wires that the same wire step extracts from the
  hand mask. Against the raw hand-drawn pieces the maximum is about 0.56, because
  outlines get cut where one wire was drawn over another.
- Run times on this laptop varied up to 5× between identical runs.
