#!/bin/bash
# The whole hold-out check (bakeoff/holdout_check.py) on the laptop, from the
# repo root in Git Bash: two U-Nets, each trained without one hand-labelled
# photo, predicted on that photo, then scored. About 2-3 hours on the RTX 4070.
#   bash bakeoff/holdout_check.sh
set -e
export PYTHONIOENCODING=utf-8
venv/Scripts/python bakeoff/holdout_check.py prepare
for k in 1 2; do
    venv_dl/Scripts/python bakeoff/train_unet.py --name D_holdout$k \
        --exclude HAND_TiS3_80000x_$k --workers 4
    venv_dl/Scripts/python bakeoff/predict.py unet --contender D_holdout$k \
        --model results/bakeoff/models/D_holdout$k/final.pt \
        --images results/bakeoff/holdout/images --out results/bakeoff/holdout/pred/D_holdout$k
done
venv/Scripts/python bakeoff/holdout_check.py score
