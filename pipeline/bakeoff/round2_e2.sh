#!/bin/bash
# Round 2, E2 (README "Round 2", declared 13:25): E's recipe on the rebuilt train_E
# (+ SF251211N_50000x_59 br, SF251213M_50000x_38 bl). Run from the repo root:
#   bash bakeoff/round2_e2.sh > results/bakeoff/logs/round2_e2.log 2>&1     (~3 h)
# 1) dev folds E2_cvA/B on the 7-quarter dev set, 2) the final E2 on all data,
# 3) E2's test probabilities into results/bakeoff/pred/E2/ (scored later, once,
#    by final_round2.py; nothing here reads a test label).
set -e
export PYTHONIOENCODING=utf-8
bash bakeoff/dev_cv_variant.sh E2 --data E
venv_dl/Scripts/python -u bakeoff/train_unet.py --name E2 --data E --copy-paste \
    --skeleton-recall 1.0 --workers 3
venv_dl/Scripts/python bakeoff/predict.py unet --contender E2 \
    --model results/bakeoff/models/E2/final.pt
echo "round 2 E2 training done"
