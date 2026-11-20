#!/bin/bash
#SBATCH --job-name=bakeoff_prep
#SBATCH --partition=t4_normal_q
#SBATCH --qos=fal_t4_normal_short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/bakeoff_prep-%j.out
#SBATCH --error=logs/bakeoff_prep-%j.err
#SBATCH --account=YOUR_ACCOUNT

# Bake-off training data (bakeoff/README.md, step 2): agreement pseudo-labels
# on every in-band TiS3 photo, the copy-paste photos, then the nnU-Net datasets
# (601 = B, 602 = C) and the test images. CPU work in the main venv; the GPU
# idles (Falcon has no CPU-only partition, as in ridge_full.sbatch).
# Photos are read from data/real/TiS3 on sapphire/ (or $BAKEOFF_PHOTOS).
#   sbatch slurm/bakeoff_prep.sbatch
cd "$SLURM_SUBMIT_DIR"
if [ ! -f bakeoff/pseudo_labels.py ]; then
    echo "error: submit from the repo root (got $SLURM_SUBMIT_DIR)" >&2
    exit 1
fi
mkdir -p logs

if [ -z "$PYTHON" ]; then
    if [ -x venv/bin/python ]; then PYTHON=venv/bin/python; else PYTHON=python3; fi
fi
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export nnUNet_raw=${nnUNet_raw:-$PWD/results/bakeoff/nnunet/raw}

"$PYTHON" bakeoff/pseudo_labels.py --previews || exit $?
"$PYTHON" bakeoff/copy_paste.py || exit $?
"$PYTHON" bakeoff/export_nnunet.py || exit $?
echo "bakeoff_prep done"
