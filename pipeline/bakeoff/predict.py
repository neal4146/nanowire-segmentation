"""Predict the test photos with a trained contender and write its masks for
bakeoff/score_bakeoff.py.

    venv_dl/Scripts/python bakeoff/predict.py nnunet --contender B --dataset 601
    venv_dl/Scripts/python bakeoff/predict.py nnunet --contender C --dataset 602
    venv_dl/Scripts/python bakeoff/predict.py unet --contender D --model results/bakeoff/models/D/final.pt

Input: results/bakeoff/test_images/<case>_0000.png (written by export_nnunet.py),
or any folder of such PNGs with --images.
Output: results/bakeoff/pred/<contender>/<case>.png, 255 = wire, at the
50k-equivalent scale. nnU-Net's mask is its argmax (wire vs background); the
U-Net's is probability > 0.5, with the probability also saved as
<case>_prob.png (0-255). No threshold is tuned: the test photos choose nothing.
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from common import OUT, PRED, TEST_IMAGES, read_png, save_u8


def predict_nnunet(images: Path, dest: Path, dataset: int, trainer: str, plans: str,
                   checkpoint: str) -> None:
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

    results = Path(os.environ.get("nnUNet_results", OUT / "nnunet" / "results"))
    folder = results / maybe_convert_to_dataset_name(dataset) / f"{trainer}__{plans}__2d"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                device=dev, verbose=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(str(folder), use_folds=("all",),
                                                   checkpoint_name=checkpoint)
    with tempfile.TemporaryDirectory() as tmp:
        predictor.predict_from_files(str(images), tmp, save_probabilities=False, overwrite=True,
                                     num_processes_preprocessing=1,
                                     num_processes_segmentation_export=1)
        for f in sorted(Path(tmp).glob("*.png")):
            save_u8(dest / f.name, (read_png(f) == 1) * 255)
            print(f"  {f.stem}", flush=True)


def _window(n: int) -> np.ndarray:
    w = np.hanning(n + 2)[1:-1]
    return np.outer(w, w).astype(np.float32) + 1e-3


def load_unet(model_path: Path):
    """(model, device) for unet_prob: a train_unet.py checkpoint, in eval mode."""
    import torch
    from train_unet import build_model

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(model_path, map_location=dev, weights_only=False)
    model = build_model(ck["args"]["encoder"], weights=None).to(dev).eval()
    model.load_state_dict(ck["model"])
    return model, dev


def unet_prob(model, dev, img: np.ndarray, tile: int = 512, overlap: int = 128,
              tta: bool = True) -> np.ndarray:
    """Wire probability (float32, 0-1) of one photo. img: the u8 photo / 255 at the
    50k-equivalent scale (as read from the *_0000.png files). Tiles with a Hann
    window, averaged over flips. Also used by measure/measure_wires.py."""
    import torch

    win = _window(tile)
    step = tile - overlap
    flips = [(), (-1,), (-2,), (-2, -1)] if tta else [()]
    x = (img - img.mean()) / (img.std() + 1e-6)
    H, W = x.shape
    Hp, Wp = max(H, tile), max(W, tile)
    xp = np.pad(x, ((0, Hp - H), (0, Wp - W)), mode="reflect")
    acc = np.zeros((Hp, Wp), np.float32)
    wsum = np.zeros((Hp, Wp), np.float32)
    rows = sorted({*range(0, Hp - tile + 1, step), Hp - tile})
    cols = sorted({*range(0, Wp - tile + 1, step), Wp - tile})
    with torch.no_grad(), torch.autocast(dev.type, dtype=torch.float16,
                                         enabled=dev.type == "cuda"):
        for r in rows:
            for c in cols:
                t = torch.from_numpy(np.ascontiguousarray(xp[r:r + tile, c:c + tile]))[None, None].to(dev)
                p = 0
                for fl in flips:
                    tt = torch.flip(t, fl) if fl else t
                    o = torch.sigmoid(model(tt).float())
                    p = p + (torch.flip(o, fl) if fl else o)
                p = (p / len(flips))[0, 0].cpu().numpy()
                acc[r:r + tile, c:c + tile] += p * win
                wsum[r:r + tile, c:c + tile] += win
    return (acc / wsum)[:H, :W]


def predict_unet(images: Path, dest: Path, model_path: Path, tile: int, overlap: int,
                 tta: bool) -> None:
    model, dev = load_unet(model_path)
    for f in sorted(images.glob("*_0000.png")):
        img = read_png(f).astype(np.float32) / 255
        prob = unet_prob(model, dev, img, tile, overlap, tta)
        case = f.name[:-len("_0000.png")]
        save_u8(dest / f"{case}.png", (prob > 0.5) * 255)
        save_u8(dest / f"{case}_prob.png", np.round(prob * 255))
        print(f"  {case}: wire {np.mean(prob > 0.5) * 100:.1f}% of pixels", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="kind", required=True)
    for kind in ("nnunet", "unet"):
        s = sub.add_parser(kind)
        s.add_argument("--contender", required=True, help="name of the pred/ folder (B, C, D, ...)")
        s.add_argument("--images", type=Path, default=TEST_IMAGES)
        s.add_argument("--out", type=Path, default=None,
                       help="output folder (default results/bakeoff/pred/<contender>)")
    n = sub.choices["nnunet"]
    n.add_argument("--dataset", type=int, required=True)
    n.add_argument("--trainer", default="nnUNetTrainer")
    n.add_argument("--plans", default="nnUNetPlans")
    n.add_argument("--checkpoint", default="checkpoint_final.pth")
    u = sub.choices["unet"]
    u.add_argument("--model", type=Path, required=True)
    u.add_argument("--tile", type=int, default=512)
    u.add_argument("--overlap", type=int, default=128)
    u.add_argument("--no-tta", action="store_true", help="skip the flip averaging")
    args = ap.parse_args()

    dest = args.out or PRED / args.contender
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    print(f"predicting {args.images} -> {dest}")
    if args.kind == "nnunet":
        predict_nnunet(args.images, dest, args.dataset, args.trainer, args.plans, args.checkpoint)
    else:
        predict_unet(args.images, dest, args.model, args.tile, args.overlap, not args.no_tta)


if __name__ == "__main__":
    main()
