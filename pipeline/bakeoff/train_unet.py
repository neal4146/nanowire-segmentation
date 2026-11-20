"""Contender D: a U-Net with an ImageNet-pretrained encoder, trained on the
same data as nnU-Net.

    venv_dl/Scripts/python bakeoff/train_unet.py --name D                  # B's data
    venv_dl/Scripts/python bakeoff/train_unet.py --name D_cp --copy-paste  # C's data
    venv_dl/Scripts/python bakeoff/train_unet.py --fetch-weights           # download only
    venv_dl/Scripts/python bakeoff/train_unet.py --name E --data E --copy-paste --skeleton-recall 1.0

Why it is in the race: with very few labels a pretrained encoder has beaten a
U-Net trained from scratch (ESRVS, 2026: one labelled image, Dice 0.80 vs
0.71), while with plenty of labels nnU-Net wins (OmniCrack30k). We have few
hand labels but ~40 pseudo-labelled photos, so neither result settles it.
The decoder works back up to full resolution through skip connections, so
8 px wires are not lost the way they are in 16 px vision-transformer patches.

Training mirrors nnU-Net where it matters for a fair race:
  data     results/bakeoff/train (+ copypaste/ with --copy-paste); each photo is
           drawn uniformly, the two hand-labelled photos --hand-copies times each
           (the same mix as the nnU-Net datasets); a third of crops are centred
           on a labelled wire pixel (nnU-Net's foreground oversampling)
  input    each photo z-scored with its own mean and std
  loss     BCE + soft Dice over labelled pixels (ignore = 2 adds nothing),
           plus optional skeleton recall (--skeleton-recall WEIGHT)
  augment  flips, rotation +-5 deg (wires keep their growth directions),
           scale 0.85-1.15, brightness/contrast/gamma, noise, blur
Checkpoints go to results/bakeoff/models/<name>/: last.pt every 500 steps (a
rerun resumes from it, so a Slurm job can be resubmitted) and final.pt.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from common import (COPYPASTE, COPYPASTE_E, IGNORE, MODELS, OUT, TRAIN, TRAIN_E, WIRE, read_json,
                    read_png)

DATA = {"B": (TRAIN, COPYPASTE), "E": (TRAIN_E, COPYPASTE_E),
        "ERB": (OUT / "train_ERB", OUT / "copypaste_ERB")}   # (base set, copy-paste set); ERB = E + flank background (build_rb_data.py)


def build_model(encoder: str, weights: str | None = "imagenet") -> torch.nn.Module:
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=encoder, encoder_weights=weights, in_channels=1, classes=1)


def load_cases(copy_paste: bool, exclude=(), data: str = "B", cp_dir=None) -> list[dict]:
    out, hit = [], set()
    base, cp = DATA[data]
    cp = Path(cp_dir) if cp_dir else cp
    for folder in [base] + ([cp] if copy_paste else []):
        for c in read_json(folder / "manifest.json")["cases"]:
            if c["case"] in exclude or c.get("from") in exclude:
                hit.update({c["case"], c.get("from")})
                continue
            img = read_png(folder / "images" / f"{c['case']}.png")
            lab = read_png(folder / "labels" / f"{c['case']}.png")
            wire = np.flatnonzero(lab == WIRE)
            out.append({"case": c["case"], "source": c["source"], "img": img, "lab": lab,
                        "mean": float(img.mean()) / 255, "std": float(img.std()) / 255 + 1e-6,
                        "wire_idx": wire[:: max(1, len(wire) // 20000)]})
    missing = set(exclude) - hit
    if missing:      # a typo or a renamed case would otherwise train on a held-out photo
        raise SystemExit(f"--exclude names matched no training case: {sorted(missing)}")
    return out


class Crops(torch.utils.data.Dataset):
    """Random augmented crops; every __getitem__ draws a fresh one."""

    def __init__(self, cases, draw, crop: int, epoch_len: int, skeleton: bool, rot_deg: float = 5.0):
        self.cases, self.draw, self.crop = cases, draw, crop
        self.epoch_len, self.skeleton, self.rot_deg = epoch_len, skeleton, rot_deg

    def __len__(self) -> int:
        return self.epoch_len

    def __getitem__(self, _):
        rng = np.random.default_rng()
        c = self.cases[self.draw[rng.integers(len(self.draw))]]
        H, W = c["lab"].shape
        if rng.random() < 1 / 3 and len(c["wire_idx"]):
            cy, cx = divmod(int(c["wire_idx"][rng.integers(len(c["wire_idx"]))]), W)
        else:
            cy, cx = rng.uniform(0, H), rng.uniform(0, W)
        s, th = rng.uniform(0.85, 1.15), np.radians(rng.uniform(-self.rot_deg, self.rot_deg))
        a, b, n = np.cos(th) / s, np.sin(th) / s, self.crop / 2
        A = np.array([[a, -b, cx - a * n + b * n], [b, a, cy - b * n - a * n]], np.float32)
        size = (self.crop, self.crop)
        x = cv2.warpAffine(c["img"].astype(np.float32) / 255, A, size,
                           flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_REFLECT)
        y = cv2.warpAffine(c["lab"], A, size, flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=IGNORE)
        if rng.random() < 0.5:                       # flips map 58 deg <-> 122 deg: still growth directions
            x, y = x[:, ::-1], y[:, ::-1]
        if rng.random() < 0.5:
            x, y = x[::-1], y[::-1]
        if rng.random() < 0.3:
            x = np.clip(x, 0, 1) ** rng.uniform(0.7, 1.5)
        x = (x - c["mean"]) / c["std"]
        x = x * rng.uniform(0.75, 1.25) + rng.uniform(-0.2, 0.2)
        if rng.random() < 0.2:
            x = cv2.GaussianBlur(np.ascontiguousarray(x), (0, 0), rng.uniform(0.5, 1.0))
        if rng.random() < 0.15:
            x = x + rng.normal(0, rng.uniform(0, 0.1), x.shape)
        y = np.ascontiguousarray(y)
        out = {"x": torch.from_numpy(np.ascontiguousarray(x, np.float32))[None],
               "y": torch.from_numpy(y.astype(np.int64))[None]}
        if self.skeleton:
            from skimage.morphology import skeletonize
            out["skel"] = torch.from_numpy(skeletonize(y == WIRE).astype(np.float32))[None]
        return out


def loss_fn(logits, y, skel=None, w_skel: float = 0.0) -> tuple[torch.Tensor, dict]:
    logits = logits.float()
    valid = (y != IGNORE).float()
    t = (y == WIRE).float()
    bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    bce = (bce * valid).sum() / valid.sum().clamp(min=1)
    p = torch.sigmoid(logits)
    pv = p * valid
    dice = 1 - (2 * (pv * t).sum() + 1) / (pv.sum() + (t * valid).sum() + 1)   # batch Dice
    loss, parts = bce + dice, {"bce": bce.item(), "dice": dice.item()}
    if skel is not None and w_skel > 0:
        sr = 1 - (p * skel).sum() / (skel.sum() + 1)
        loss = loss + w_skel * sr
        parts["skel"] = sr.item()
    return loss, parts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", default="D")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--copy-paste", action="store_true", help="also train on copypaste/")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="training cases to leave out (holdout_check.py: one hand-labelled photo)")
    ap.add_argument("--data", default="B", choices=sorted(DATA),
                    help="B: train/ (+ copypaste/), as B, C, D; E: train_E/ (+ copypaste_E/), "
                         "built by build_e_data.py")
    ap.add_argument("--cp-dir", default=None,
                    help="copy-paste set to use instead of the default (E hold-out runs)")
    ap.add_argument("--hand-copies", type=int, default=8)
    ap.add_argument("--corrected-copies", type=int, default=3,
                    help="draws per photo with hand-fixed quarters (data E)")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--skeleton-recall", type=float, default=0.0)
    ap.add_argument("--rot-deg", type=float, default=5.0,
                    help="random rotation range, +- degrees (E used 5)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--restart", action="store_true", help="ignore an existing last.pt")
    ap.add_argument("--fetch-weights", action="store_true",
                    help="only download the pretrained encoder (run on a node with internet)")
    args = ap.parse_args()

    if args.fetch_weights:
        build_model(args.encoder)
        print(f"{args.encoder} ImageNet weights cached")
        return

    out = MODELS / args.name
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cases = load_cases(args.copy_paste, set(args.exclude), args.data, args.cp_dir)
    copies = {"hand": args.hand_copies, "corrected": args.corrected_copies}
    draw = [i for i, c in enumerate(cases) for _ in range(copies.get(c["source"], 1))]
    n = lambda s: sum(c["source"] == s for c in cases)          # noqa: E731
    print(f"data {args.data}: {len(cases)} training photos ({n('hand')} hand-labelled "
          f"x{args.hand_copies}, {n('corrected')} hand-fixed x{args.corrected_copies}, "
          f"{n('pseudo')} pseudo, {n('copypaste')} copy-paste); device {dev}", flush=True)

    model = build_model(args.encoder).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warm = min(500, args.iters // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 *
                                              (1 + math.cos(math.pi * min(s, args.iters) / args.iters)))
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    step = 0
    if (out / "last.pt").exists() and not args.restart:
        ck = torch.load(out / "last.pt", map_location=dev, weights_only=False)
        old = ck.get("args", {})
        for k in ("data", "copy_paste", "cp_dir", "exclude", "encoder", "iters", "batch", "crop",
                  "lr", "skeleton_recall", "hand_copies", "corrected_copies", "rot_deg"):
            a, b = old.get(k, 5.0 if k == "rot_deg" else None), getattr(args, k)
            if (sorted(a or []) != sorted(b or [])) if k == "exclude" else (a != b):
                raise SystemExit(f"{out / 'last.pt'} was written with {k}={a!r}, this run has {b!r}: "
                                 "use another --name or --restart")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        step = ck["step"]
        print(f"resumed from step {step}", flush=True)
    if step >= args.iters:
        print("already finished")
        return

    data = Crops(cases, draw, args.crop, (args.iters - step) * args.batch,
                 skeleton=args.skeleton_recall > 0, rot_deg=args.rot_deg)
    loader = torch.utils.data.DataLoader(data, batch_size=args.batch, num_workers=args.workers,
                                         pin_memory=dev.type == "cuda",
                                         persistent_workers=args.workers > 0)

    def save(path):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(), "step": step,
                    "args": vars(args)}, path)

    log = open(out / "log.txt", "a", encoding="utf-8")
    model.train()
    t0, run = time.perf_counter(), {}
    for batch in loader:
        x, y = batch["x"].to(dev, non_blocking=True), batch["y"].to(dev, non_blocking=True)
        skel = batch["skel"].to(dev) if "skel" in batch else None
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            logits = model(x)
        loss, parts = loss_fn(logits, y, skel, args.skeleton_recall)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        for k, v in {"loss": loss.item(), **parts}.items():
            run[k] = run.get(k, 0.0) + v
        if step % 100 == 0 or step == args.iters:
            n = 100 if step % 100 == 0 else step % 100
            msg = (f"step {step:6d}/{args.iters}  " +
                   "  ".join(f"{k} {v / n:.4f}" for k, v in run.items()) +
                   f"  lr {sched.get_last_lr()[0]:.2e}  {n / (time.perf_counter() - t0):.1f} it/s")
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
            t0, run = time.perf_counter(), {}
        if step % 500 == 0:
            save(out / "last.pt")
        if step >= args.iters:
            break
    save(out / "last.pt")
    save(out / "final.pt")
    (out / "train_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"done: {out / 'final.pt'}")


if __name__ == "__main__":
    main()
