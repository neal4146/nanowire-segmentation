#!/bin/bash
# Contender E's hold-out runs, used ONLY to choose how E's probability map is
# turned into a mask (bakeoff/thin_masks.py). From the repo root in Git Bash:
#   bash bakeoff/holdout_E.sh            (~2 h on the RTX 4070)
# Each run trains E's recipe without one hand-labelled photo, with a copy-paste
# set whose sprites come only from the OTHER hand photo (no leak), and predicts
# the held-out photo. The test photos are not touched.
set -e
export PYTHONIOENCODING=utf-8
venv/Scripts/python bakeoff/holdout_check.py prepare
for k in 1 2; do
    other=$((3 - k))
    venv/Scripts/python bakeoff/copy_paste.py --src results/bakeoff/train_E \
        --dst results/bakeoff/copypaste_E_ho$k --faint 15 --events 30 \
        --sprite-photos TiS3_80000x_$other
    venv_dl/Scripts/python -u bakeoff/train_unet.py --name E_holdout$k --data E --copy-paste \
        --cp-dir results/bakeoff/copypaste_E_ho$k --skeleton-recall 1.0 \
        --exclude HAND_TiS3_80000x_$k
    venv_dl/Scripts/python bakeoff/predict.py unet --contender E_holdout$k \
        --model results/bakeoff/models/E_holdout$k/final.pt \
        --images results/bakeoff/holdout/images --out results/bakeoff/holdout/pred/E_holdout$k
done
echo "hold-out runs done: venv/Scripts/python bakeoff/thin_masks.py choose"
