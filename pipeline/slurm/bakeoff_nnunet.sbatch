#!/bin/bash
#SBATCH --job-name=bakeoff_nnunet
#SBATCH --partition=t4_normal_q
#SBATCH --qos=fal_t4_normal_short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/bakeoff_nnunet-%j.out
#SBATCH --error=logs/bakeoff_nnunet-%j.err
#SBATCH --account=YOUR_ACCOUNT

# Contenders B and C (bakeoff/README.md, step 3): nnU-Net v2, 2D, trained on
# every training photo (fold "all"), then the test photos are predicted into
# results/bakeoff/pred/<B|C>/.
#   sbatch slurm/bakeoff_nnunet.sbatch B                                   # dataset 601
#   sbatch slurm/bakeoff_nnunet.sbatch C                                   # dataset 602
#   TRAINER=nnUNetTrainer_250epochs sbatch slurm/bakeoff_nnunet.sbatch B   # quicker first pass
# The default nnUNetTrainer runs 1000 epochs, about a day or more on a T4. If
# the job hits its time limit, submit the same command again: training resumes
# from the last checkpoint (nnU-Net saves one every 50 epochs). If the QOS
# refuses 24 h, lower --time; resubmitting still finishes the run.
# Needs venv_dl (bash bakeoff/setup_dl_env.sh) and slurm/bakeoff_prep.sbatch done.

KEY=${1:?usage: sbatch slurm/bakeoff_nnunet.sbatch B|C}
case $KEY in
    B) DID=601 ;;
    C) DID=602 ;;
    *) echo "error: contender must be B or C (got $KEY)" >&2; exit 1 ;;
esac
DS=Dataset${DID}_TiS3_${KEY}
TRAINER=${TRAINER:-nnUNetTrainer}
NAME=$KEY
[ "$TRAINER" != nnUNetTrainer ] && NAME=${KEY}_${TRAINER#nnUNetTrainer_}

cd "$SLURM_SUBMIT_DIR"
if [ ! -f bakeoff/predict.py ]; then
    echo "error: submit from the repo root (got $SLURM_SUBMIT_DIR)" >&2
    exit 1
fi
mkdir -p logs
PY=${PYTHON_DL:-venv_dl/bin/python}
BIN=$(dirname "$PY")
export nnUNet_raw=${nnUNet_raw:-$PWD/results/bakeoff/nnunet/raw}
export nnUNet_preprocessed=${nnUNet_preprocessed:-$PWD/results/bakeoff/nnunet/preprocessed}
export nnUNet_results=${nnUNet_results:-$PWD/results/bakeoff/nnunet/results}
export nnUNet_n_proc_DA=$((SLURM_CPUS_PER_TASK - 1))

if [ ! -d "$nnUNet_preprocessed/$DS/nnUNetPlans_2d" ]; then
    "$BIN/nnUNetv2_plan_and_preprocess" -d $DID -c 2d -np $SLURM_CPUS_PER_TASK \
        --verify_dataset_integrity || exit $?
fi
OUTDIR=$nnUNet_results/$DS/${TRAINER}__nnUNetPlans__2d/fold_all
if [ ! -f "$OUTDIR/checkpoint_final.pth" ]; then
    CONT=""
    [ -f "$OUTDIR/checkpoint_latest.pth" ] && CONT="--c"
    "$BIN/nnUNetv2_train" $DID 2d all -tr $TRAINER $CONT || exit $?
fi
"$PY" bakeoff/predict.py nnunet --contender "$NAME" --dataset $DID --trainer $TRAINER || exit $?
echo "bakeoff_nnunet $NAME done"
