#!/bin/bash
# A training variant through the same dev folds as dev_cv.sh (see bakeoff/dev_cv.py):
# trained twice, each without one fold of the complete-mode hand-fixed photos,
# predicted on all 5 dev photos. The test photos are not touched.
#   bash bakeoff/dev_cv_variant.sh <name> <train_unet args, e.g. --data E --rot-deg 25>
# Outputs: models/<name>_cv{A,B}/, devcv/pred/<name>_cv{A,B}/.
# If <name> needs built data (ERB), set REQUIRE=<file>: the run is skipped unless it exists.
set -e
export PYTHONIOENCODING=utf-8
name=$1; shift
if [ -n "$REQUIRE" ] && [ ! -f "$REQUIRE" ]; then
    echo "skipping $name: $REQUIRE does not exist"; exit 0
fi
for k in A B; do
    venv_dl/Scripts/python -u bakeoff/train_unet.py --name ${name}_cv$k --copy-paste \
        --skeleton-recall 1.0 --workers 3 "$@" --exclude $(cat results/bakeoff/devcv/exclude_$k.txt)
    venv_dl/Scripts/python bakeoff/predict.py unet --contender ${name}_cv$k \
        --model results/bakeoff/models/${name}_cv$k/final.pt \
        --images results/bakeoff/devcv/images --out results/bakeoff/devcv/pred/${name}_cv$k
done
echo "dev CV runs for $name done"
