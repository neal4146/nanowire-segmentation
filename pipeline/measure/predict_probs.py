"""Wire probabilities for measure_wires.py --method unet.

    venv_dl/Scripts/python measure/predict_probs.py <jobs.json>

measure_wires.py calls this itself: in-process when its Python has torch,
otherwise through venv_dl (the bake-off's deep-learning environment).
jobs.json is {"model": <train_unet.py checkpoint>, "items": [{"photo": <.tif>,
"out": <png>}, ...]}. Each photo is loaded exactly as measure_wires.py loads it
(load_frame_50k, at the 50,000x scale) and rounded to 8 bits as in the bake-off.
It is predicted with bakeoff/predict.py's unet_prob (512 px tiles, 128 px
overlap, flip averaging), and the probability is saved as a 0-255 PNG. These
are the same steps that produced the scored test predictions.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for _sub in ("scripts/ridge", "compare", "bakeoff"):
    if str(ROOT / _sub) not in sys.path:
        sys.path.insert(0, str(ROOT / _sub))


def predict_items(model_path, items: list[dict]) -> None:
    from predict import load_unet, unet_prob
    from ridge_nanowire_images2 import load_frame_50k

    model, dev = load_unet(Path(model_path))
    print(f"network on {dev.type}: {len(items)} photo(s)", flush=True)
    t0 = time.perf_counter()
    for i, it in enumerate(items, start=1):
        try:
            gray, _ = load_frame_50k(Path(it["photo"]))
            # = common.img_to_u8(img01), the bake-off's *_0000.png images, read back / 255
            img = np.clip(np.round((gray / 255.0) * 255.0), 0, 255).astype(np.uint8).astype(np.float32) / 255
            prob = unet_prob(model, dev, img)
        except Exception as e:                         # noqa: BLE001  (one bad photo must not stop the run)
            print(f"  [{i}/{len(items)}] {Path(it['photo']).name}: skipped, "
                  f"{type(e).__name__}: {e}", flush=True)
            continue
        out = Path(it["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.round(prob * 255).astype(np.uint8), mode="L").save(out)
        print(f"  [{i}/{len(items)}] {Path(it['photo']).name} "
              f"({time.perf_counter() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    jobs = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    predict_items(jobs["model"], jobs["items"])
