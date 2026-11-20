#!/bin/bash
#SBATCH --job-name=bakeoff_unet
#SBATCH --partition=t4_normal_q
#SBATCH --qos=fal_t4_normal_short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/bakeoff_unet-%j.out
#SBATCH --error=logs/bakeoff_unet-%j.err
#SBATCH --account=YOUR_ACCOUNT

# Contender D (bakeoff/README.md, step 3): U-Net with an ImageNet-pretrained
# encoder (bakeoff/train_unet.py), then the test photos are predicted into
# results/bakeoff/pred/<name>/. Extra arguments go to train_unet.py.
#   sbatch slurm/bakeoff_unet.sbatch D                      # B's data
#   sbatch slurm/bakeoff_unet.sbatch D_cp --copy-paste      # C's data
# If the job hits its time limit, submit the same command again: training
# resumes from results/bakeoff/models/<name>/last.pt.
# Needs venv_dl (bash bakeoff/setup_dl_env.sh, which also caches the encoder
# weights) and slurm/bakeoff_prep.sbatch done.

NAME=${1:?usage: sbatch slurm/bakeoff_unet.sbatch NAME [train_unet.py options]}
shift
cd "$SLURM_SUBMIT_DIR"
if [ ! -f bakeoff/train_unet.py ]; then
    echo "error: submit from the repo root (got $SLURM_SUBMIT_DIR)" >&2
    exit 1
fi
mkdir -p logs
PY=${PYTHON_DL:-venv_dl/bin/python}

"$PY" bakeoff/train_unet.py --name "$NAME" --workers $((SLURM_CPUS_PER_TASK - 1)) "$@" || exit $?
"$PY" bakeoff/predict.py unet --contender "$NAME" \
    --model "results/bakeoff/models/$NAME/final.pt" || exit $?
echo "bakeoff_unet $NAME done"
