"""Fold cross-validation of the learned piece verifier, end to end with the official matcher.

Declared before any CV number was looked at:
  primary model : GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                  subsample=0.8, min_samples_leaf=10, random_state=0) on all FEATURES
  primary label : y_match (the piece is matched when all scored pieces are kept; one per GT line)
  threshold     : nested. Inside the training fold, leave-one-quarter-out out-of-fold
                  probabilities; threshold = argmax of the training fold's pooled wire F1
                  (fast exact scorer) over 0.05..0.95 step 0.025; ties within 0.002 -> median
                  of the tied thresholds. The evaluated fold never chooses anything.
  outer         : fit on fold A, evaluate fold B, and vice versa; pooled over all 5 quarters
                  with extract_plus.score_pieces (unsure dropped) + pooled_rows.
Other rows (reported, not used to pick the primary): fixed threshold 0.5, logistic regression,
label y_spec, in-sample.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import extract_plus as xp  # noqa: E402
import verifier_learned as vl  # noqa: E402

THR_GRID = np.round(np.arange(0.05, 0.951, 0.025), 3)


def gb():
    return GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                                      min_samples_leaf=10, random_state=0)


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))


MODELS = {"gb": gb, "lr": lr}


def fit(make, tab, quarters, label, feats):
    d = tab[tab.quarter.isin(quarters) & ~tab.uns]
    return make().fit(d[feats].values, d[label].values)


def predict(model, tab, quarter, feats):
    d = tab[tab.quarter == quarter]
    return model.predict_proba(d[feats].values)[:, 1]


def pick_threshold(probs: dict, preps_by: dict):
    """argmax pooled fast F1 over THR_GRID; ties within 0.002 -> median tied threshold."""
    qs = sorted(probs)
    f1 = np.array([core.fast_pooled([preps_by[q] for q in qs], [probs[q] >= t for q in qs])[0]
                   for t in THR_GRID])
    tied = THR_GRID[f1 >= f1.max() - 0.002]
    return float(np.median(tied)), float(f1.max()), f1


def inner_oof(make, tab, train_qs, label, feats):
    out = {}
    for u in train_qs:
        rest = [v for v in train_qs if v != u]
        out[u] = predict(fit(make, tab, rest, label, feats), tab, u, feats)
    return out


def run_cv(make, tab, preps_by, label, feats, fixed_thr=None):
    """Outer fold CV. Returns {quarter: (probs, threshold)} and per-fold info."""
    res, info = {}, {}
    for ev, tr in (("A", "B"), ("B", "A")):
        tr_qs, ev_qs = core.FOLDS[tr], core.FOLDS[ev]
        if fixed_thr is None:
            oof = inner_oof(make, tab, tr_qs, label, feats)
            thr, f_in, _ = pick_threshold(oof, preps_by)
        else:
            thr, f_in = fixed_thr, float("nan")
        model = fit(make, tab, tr_qs, label, feats)
        for q in ev_qs:
            res[q] = (predict(model, tab, q, feats), thr)
        info[ev] = {"trained_on": tr, "threshold": thr, "train_fold_inner_f1": f_in}
    return res, info


def evaluate(qs, res, variant):
    kept = []
    for q in qs:
        p, thr = res[q["name"]]
        kept.append([pc for pc, s in zip(q["pieces"], p) if s >= thr])
    row, per_q, counts = core.official(qs, kept, variant)
    return row, per_q, counts


def main():
    qs = core.load_cache()
    tab = pd.read_pickle(core.WORK / "piece_table.pkl")
    tab["uns"] = tab.uns.astype(bool)
    with open(core.WORK / "preps.pkl", "rb") as fh:
        preps = pickle.load(fh)
    # the match label is recomputed from preps (cardinality-first matching, see core.prep_quarter)
    preps_by = {q["name"]: pr for q, pr in zip(qs, preps)}
    feats = list(vl.FEATURES)

    base_kept = [[pc for pc in q["pieces"] if pc["length_px"] >= 20] for q in qs]
    base_row, _, base_counts = core.official(qs, base_kept, "lsd_m20")
    print(f"baseline lsd_m20 (same harness): F1 {base_row['wire_f1']:.4f}", flush=True)

    configs = [("V_gb_match_nested", "gb", "y_match", None),
               ("V_gb_match_thr0.5", "gb", "y_match", 0.5),
               ("V_lr_match_nested", "lr", "y_match", None),
               ("V_gb_spec_nested", "gb", "y_spec", None)]
    rows, all_per_q, store = [], [], {}
    for name, mk, label, thr in configs:
        res, info = run_cv(MODELS[mk], tab, preps_by, label, feats, thr)
        row, per_q, counts = evaluate(qs, res, name)
        d = xp._paired(counts[("E_thin", name)], base_counts[("E_thin", "lsd_m20")])
        per_f1 = {p["quarter"]: p["wire_f1"] for p in per_q}
        row.update({"variant": name, "d_vs_lsd_m20": d[0], "d_lo": d[1], "d_hi": d[2],
                    "thr_A": info["A"]["threshold"], "thr_B": info["B"]["threshold"],
                    "inner_f1_trainB": info["A"]["train_fold_inner_f1"],
                    "inner_f1_trainA": info["B"]["train_fold_inner_f1"]})
        rows.append(row)
        all_per_q += per_q
        store[name] = {q: (p.tolist(), t) for q, (p, t) in res.items()}
        print(f"{name}: CV F1 {row['wire_f1']:.4f} R {row['wire_recall']:.4f} P {row['wire_precision']:.4f} "
              f"d {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}] thr A {info['A']['threshold']} B {info['B']['threshold']} "
              f"per-q {json.dumps({k[:22]: round(v, 3) for k, v in per_f1.items()})}", flush=True)

    # in-sample (fit on all 5, threshold on the same 5, in-sample probs): optimistic reference
    model = fit(gb, tab, [q["name"] for q in qs], "y_match", feats)
    ins = {q["name"]: predict(model, tab, q["name"], feats) for q in qs}
    thr, _, _ = pick_threshold(ins, preps_by)
    row, per_q, _ = evaluate(qs, {k: (v, thr) for k, v in ins.items()}, "V_gb_match_insample")
    row.update({"variant": "V_gb_match_insample", "thr_A": thr, "thr_B": thr})
    rows.append(row)
    all_per_q += per_q
    print(f"in-sample: F1 {row['wire_f1']:.4f} thr {thr}", flush=True)

    base_row.update({"variant": "lsd_m20"})
    out = pd.DataFrame([base_row] + rows)
    out.to_csv(core.WORK / "cv_scores.csv", index=False, float_format="%.4f")
    pd.DataFrame(all_per_q).to_csv(core.WORK / "cv_per_quarter.csv", index=False, float_format="%.4f")
    with open(core.WORK / "cv_probs.json", "w") as fh:
        json.dump(store, fh)
    print(out.drop(columns=["contender"]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
