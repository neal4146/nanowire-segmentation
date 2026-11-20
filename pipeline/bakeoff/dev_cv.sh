#!/bin/bash
# Dev set in the test's style (see bakeoff/dev_cv.py): E's recipe trained twice,
# each run without one fold of the complete-mode hand-fixed training photos,
# then predicted on all 5 (each fold model is scored only on its own held-out
# photos). The test photos are not touched. From the repo root in Git Bash:
#   venv/Scripts/python bakeoff/dev_cv.py prepare
#   bash bakeoff/dev_cv.sh            (~2 h on the RTX 4070)
set -e
export PYTHONIOENCODING=utf-8
for k in A B; do
    venv_dl/Scripts/python -u bakeoff/train_unet.py --name E_cv$k --data E --copy-paste \
        --skeleton-recall 1.0 --workers 3 --exclude $(cat results/bakeoff/devcv/exclude_$k.txt)
    venv_dl/Scripts/python bakeoff/predict.py unet --contender E_cv$k \
        --model results/bakeoff/models/E_cv$k/final.pt \
        --images results/bakeoff/devcv/images --out results/bakeoff/devcv/pred/E_cv$k
done
echo "dev CV runs done"
