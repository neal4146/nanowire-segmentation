"""Cache the 5 dev quarters: E_thin mask, prob crop, image crop, ALL LSD pieces
(min_len_px 0), GT instances, unsure raster, frame; and reproduce lsd_m20.

    venv/Scripts/python -B <this> (from the repo root)
Writes only into this script's folder.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core  # noqa: E402  (sets sys.path, PREFIX, WORK)

import pickle  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import extract_plus as xp  # noqa: E402
from common import read_png  # noqa: E402
from score import extract_params  # noqa: E402


def main():
    p = extract_params()
    qs = xp.dev_quarters()
    cache, per_q, counts = [], [], {}
    for q in qs:
        m = core.thin_mask(prob_full := (read_png(xp.DEV / "pred" / f"{core.PREFIX}{q['fold']}" / f"{q['case']}_prob.png").astype(np.float32) / 255))[q["crop"]]
        f = xp.DEV / "pred" / f"{core.PREFIX}{q['fold']}" / f"{q['case']}_prob.png"
        prob = (read_png(f).astype(np.float32) / 255)[q["crop"]]
        img = q["img01"][q["crop"]].astype(np.float32)
        nm = q["frame"].nm_per_px
        allp = xp.extract(m, nm, p, {"mode": "lsd", "min_len_px": 0})
        m20 = xp.extract(m, nm, p, {"mode": "lsd", "min_len_px": 20})
        s, c = xp.score_pieces(m20, m, q["frame"], q["ref"], drop_inst=q["unsure"])
        name = q["frame"].name
        per_q.append({"quarter": name, "fold": q["fold"], "contender": "E_thin", "variant": "lsd_m20", **s})
        counts.setdefault(("E_thin", "lsd_m20"), {})[name] = c
        cache.append({"name": name, "case": q["case"], "fold": q["fold"], "mask": m, "prob": prob,
                      "img": img, "pieces": allp, "frame": q["frame"], "ref": q["ref"],
                      "unsure": q["unsure"], "nm": nm})
        print(f"{name} fold {q['fold']}: {len(allp)} pieces (>=20: {len(m20)}), "
              f"gt {q['ref']['count']}, F1 {s['wire_f1']:.4f}", flush=True)
    rows = pd.DataFrame(xp.pooled_rows(per_q, counts))
    print(rows.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    with open(core.WORK / "dev_cache.pkl", "wb") as fh:
        pickle.dump(cache, fh, protocol=4)


if __name__ == "__main__":
    main()
