"""Round 2's declared test batch, scored ONCE (see bakeoff/README.md, "Round 2").

    venv/Scripts/python bakeoff/final_round2.py results/bakeoff/devcv/round2_choice.json

The choice file is written from DEV results only, before this runs. It lists
the rows to score, e.g.
  {"primary": ["E2_thin+V", "E_thin+X"],
   "rows": [{"name": "E2_thin+V", "pred": "E2", "extract": "verifier",
             "verifier": "results/bakeoff/devcv/verifier_E2_cv/verifier_learned.joblib"},
            {"name": "E2_thin+X", "pred": "E2", "extract": "lsd_m20"}]}
Every row: E_thin's recipe (thin_masks ridge choice) on results/bakeoff/pred/<pred>/<case>_prob.png,
then LSD pieces; 'lsd_m20' drops pieces < 20 px (round 1's pick), 'verifier' keeps
the pieces bakeoff/verifier's learned model accepts (fitted on dev only).
References always scored alongside: E_thin+X (round 1's leader, 0.639) and A.
Writes results/bakeoff/summary_round2.csv (tile-bootstrap paired intervals with
score_bakeoff's draws, plus whole-photo bootstrap) and scores_per_quarter_round2.csv,
both stamped with code and choice hashes. Refuses to run twice.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

from common import OUT, PRED, read_json, read_png

SUMMARY = OUT / "summary_round2.csv"


def main(choice_path: str) -> None:
    import pandas as pd
    from bootstrap import pooled
    from external import lines_to_frame, read_lines
    from pseudo_labels import path_open_clean_fixed
    from score import extract_params
    from score_bakeoff import REF, label_sets
    from thin_masks import CHOICE as THIN_CHOICE, make_mask
    from common import TRAIN, TRAIN_E, case_name, find_photo, load_photo, photo_name, run_of, test_photos
    import extract_plus as X
    sys.path.insert(0, str(Path(__file__).resolve().parent / "verifier"))

    if SUMMARY.exists():
        raise SystemExit(f"{SUMMARY} exists: round 2's batch was already scored once")
    choice_path = Path(choice_path)
    ch = read_json(choice_path)
    th = json.loads(THIN_CHOICE.read_text(encoding="utf-8"))
    rows_cfg = ch["rows"]
    verifiers = {}
    for r in rows_cfg:
        if r["extract"] == "verifier":
            import joblib
            verifiers[r["name"]] = joblib.load(r["verifier"])
    import verifier_learned as VL
    print(f"declared batch: {[r['name'] for r in rows_cfg]} + references E_thin+X, A; primary {ch['primary']}")

    tests = set(test_photos())
    trained_on = {c["photo"] for t in (TRAIN, TRAIN_E) if (t / "manifest.json").exists()
                  for c in read_json(t / "manifest.json")["cases"]}
    p = extract_params()
    per_q, counts, quarters = [], {}, []
    for csv_path, meta in label_sets():
        photo = photo_name(meta["photo"])
        if photo not in tests or photo in trained_on:
            continue
        tag = f"{case_name(photo)}_{meta['quarter']}"
        lines = read_lines(csv_path)
        img01, fm = load_photo(find_photo(photo))
        if abs(fm["rescale_to_50k"] - 1) > 1e-3:
            raise SystemExit(f"{photo} is not a 50,000x photo")
        r0, c0 = meta["crop_origin_row_col"]
        h, w = meta["crop_shape_row_col"]
        crop = (slice(r0, r0 + h), slice(c0, c0 + w))
        nm = fm["pixel_size_m"] * 1e9 / fm["rescale_to_50k"]
        frame, ref = lines_to_frame(lines, (h, w), nm, tag, img01[crop])
        ref = X._with_total(ref, lines, nm)
        out = {}
        m_a = path_open_clean_fixed(img01)[crop]
        out["A_path_open_clean"] = (m_a, X.extract(m_a, nm, p, X.PROD))
        full = read_png(PRED / "E_thin" / f"{case_name(photo)}.png")
        if full.shape != img01.shape:
            raise SystemExit(f"E_thin: shape {full.shape}, photo {img01.shape}")
        m_e = (full > 127)[crop]
        out["E_thin+X"] = (m_e, X.extract(m_e, nm, p, {"mode": "lsd", "min_len_px": 20}))
        for r in rows_cfg:
            prob_full = read_png(PRED / r["pred"] / f"{case_name(photo)}_prob.png").astype(np.float32) / 255
            if prob_full.shape != img01.shape:
                raise SystemExit(f"{r['pred']}: shape {prob_full.shape}, photo {img01.shape}")
            m = make_mask(prob_full, th["kind"], th["t"], th["r"])[crop]
            if r["extract"] == "lsd_m20":
                pcs = X.extract(m, nm, p, {"mode": "lsd", "min_len_px": 20})
            elif r["extract"] == "verifier":
                allp = X.extract(m, nm, p, {"mode": "lsd", "min_len_px": 0})
                pcs = VL.verify(allp, prob_full[crop], img01[crop].astype(np.float32),
                                model=verifiers[r["name"]])
            else:
                raise SystemExit(f"unknown extract {r['extract']}")
            out[r["name"]] = (m, pcs)
        for name, (m, pcs) in out.items():
            s, tc = X.score_pieces(pcs, m, frame, ref)
            per_q.append({"quarter": tag, "photo": photo, "run": run_of(photo), "contender": name, **s})
            counts.setdefault(name, {})[tag] = tc
        quarters.append(tag)
        print(f"  {tag}: scored", flush=True)

    quarters = sorted(quarters)
    df = pd.DataFrame(per_q)
    stamp = {"final_round2_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "choice_sha256": hashlib.sha256(choice_path.read_bytes()).hexdigest()}
    df.assign(**stamp).to_csv(OUT / "scores_per_quarter_round2.csv", index=False, float_format="%.4f")
    rng = np.random.default_rng(0)
    ntiles = {q: len(counts[REF][q]["pix_tp"]) for q in quarters}
    draws = [{q: rng.integers(0, k, k) for q, k in ntiles.items()} for _ in range(2000)]
    boot = {n: np.array([pooled(per, d)[1] for d in draws]) for n, per in counts.items()}
    photos = sorted(df.photo.unique())

    q_photo = {q: df[df.quarter == q].photo.iloc[0] for q in quarters}
    keys = ("wire_gt_tp", "wire_gt", "wire_pred_tp", "wire_pred")
    tot = {n: {ph: {k: sum(float(counts[n][q][k].sum()) for q in quarters if q_photo[q] == ph) for k in keys}
               for ph in sorted(set(q_photo.values()))} for n in counts}

    def photo_f1(n, sel):
        """Pooled wire F1 over the photos in sel, a photo drawn twice counting twice."""
        t = {k: sum(tot[n][ph][k] for ph in sel) for k in keys}
        R = t["wire_gt_tp"] / max(t["wire_gt"], 1)
        P = t["wire_pred_tp"] / max(t["wire_pred"], 1)
        return 2 * P * R / max(P + R, 1e-9)

    rows = []
    for n, per in counts.items():
        sub = df[df.contender == n]
        row = {"contender": n, "wire_f1": pooled(per)[1],
               "wire_f1_lo": np.percentile(boot[n], 2.5), "wire_f1_hi": np.percentile(boot[n], 97.5),
               "wire_recall": sub.wire_recall.mean(), "wire_precision": sub.wire_precision.mean(),
               "count_err_pct_pooled": 100 * (sub.n_wires_pred.sum() - sub.n_wires_gt.sum()) / sub.n_wires_gt.sum(),
               "count_abs_err_pct": sub.count_err_pct.abs().mean(),
               "length_median_abs_err_pct": sub.length_median_nm_err_pct.abs().mean(),
               "length_total_abs_err_pct": sub.length_total_err_pct.abs().mean()}
        for ref_name in ("E_thin+X", REF):
            if ref_name == n:
                continue
            d = boot[n] - boot[ref_name]
            row[f"d_vs_{ref_name}"] = float(d.mean())
            row[f"d_vs_{ref_name}_lo"] = float(np.percentile(d, 2.5))
            row[f"d_vs_{ref_name}_hi"] = float(np.percentile(d, 97.5))
            dp = [photo_f1(n, s) - photo_f1(ref_name, s) for s in itertools.product(photos, repeat=len(photos))]
            row[f"d_vs_{ref_name}_photo_lo"] = float(np.percentile(dp, 2.5))
            row[f"d_vs_{ref_name}_photo_hi"] = float(np.percentile(dp, 97.5))
        for ph in photos:
            row[f"wire_f1_{case_name(ph).rsplit('_', 1)[-1]}"] = photo_f1(n, [ph])
        rows.append(row)
    s = pd.DataFrame(rows).sort_values("wire_f1", ascending=False).assign(**stamp)
    s.to_csv(SUMMARY, index=False, float_format="%.4f")
    print(s.drop(columns=list(stamp)).T.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nwrote {SUMMARY}")


if __name__ == "__main__":
    main(sys.argv[1])
