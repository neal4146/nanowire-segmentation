#!/bin/bash
# Build venv_dl (PyTorch + nnU-Net v2 + segmentation_models_pytorch) for the
# bake-off's GPU jobs. Run once from the repo root on an ARC LOGIN node, which
# has internet (compute nodes may not), so the encoder weights get cached too:
#   bash bakeoff/setup_dl_env.sh
# On another machine set PY to any Python 3.12/3.13.
set -e
PY=${PY:-/apps/arch/software/Python/3.13.5-GCCcore-14.3.0/bin/python3.13}
"$PY" -m venv venv_dl
venv_dl/bin/pip install --upgrade pip
venv_dl/bin/pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
venv_dl/bin/pip install -r bakeoff/requirements-dl.txt
venv_dl/bin/python bakeoff/train_unet.py --fetch-weights
venv_dl/bin/python -c "import torch; print('torch', torch.__version__, '| CUDA build', torch.version.cuda)"
