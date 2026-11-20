"""Refit the primary verifier on all dev quarters in devcv/manifest.json (5 in round 1, 7 for E2; threshold by leave-one-quarter-out
out-of-fold probabilities over them, the same nested recipe as inside each CV fold),
save it with joblib, check the module reproduces the table, and print diagnostics
(seed stability of the CV number, what is kept by length, feature importances)."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier  # noqa: E402

import train_cv as tc  # noqa: E402
import verifier_learned as vl  # noqa: E402


def main():
    qs = core.load_cache()
    tab = pd.read_pickle(core.WORK / "piece_table.pkl")
    tab["uns"] = tab.uns.astype(bool)
    with open(core.WORK / "preps.pkl", "rb") as fh:
        preps = pickle.load(fh)
    preps_by = {q["name"]: pr for q, pr in zip(qs, preps)}
    feats = list(vl.FEATURES)
    names = [q["name"] for q in qs]

    oof = tc.inner_oof(tc.gb, tab, names, "y_match", feats)
    thr, f_oof, curve = tc.pick_threshold(oof, preps_by)
    print(f"final threshold (LOQO over {len(names)}): {thr}  (LOQO pooled fast F1 at it {f_oof:.4f})")
    print("LOQO curve:", {float(t): round(float(f), 4) for t, f in zip(tc.THR_GRID, curve)})
    clf = tc.fit(tc.gb, tab, names, "y_match", feats)
    model = {"clf": clf, "threshold": thr, "features": feats, "label": "y_match",
             "trained_on": names, "n_train": int((~tab.uns).sum()),
             "sklearn": sklearn.__version__,
             "params": clf.get_params(),
             "note": "input = ALL LSD pieces (extract_plus lsd, min_len_px 0) of E_thin's mask"}
    vl.MODEL_FILE = core.WORK / "verifier_learned.joblib"
    joblib.dump(model, vl.MODEL_FILE)
    print(f"saved {vl.MODEL_FILE}")

    # the module, loaded fresh, must reproduce the table's probabilities
    vl._MODEL = None
    for q in qs:
        s_mod = vl.piece_scores(q["pieces"], q["prob"], q["img"])
        s_tab = tc.predict(clf, tab, q["name"], feats)
        kept = vl.verify(q["pieces"], q["prob"], q["img"])
        print(f"{q['name']}: max |module - table| {np.abs(s_mod - s_tab).max():.2e}, kept {len(kept)}"
              f"/{len(q['pieces'])} (in-sample)")

    # LOQO (5-fold) official score with the final threshold: another out-of-sample view
    row, per_q, _ = tc.evaluate(qs, {k: (v, thr) for k, v in oof.items()}, "V_gb_match_LOQO5")
    print(f"LOQO-5 official pooled F1 {row['wire_f1']:.4f} R {row['wire_recall']:.4f} P {row['wire_precision']:.4f}"
          " (threshold picked on these same OOF probs: slightly optimistic)")

    # seed stability of the primary CV (diagnostic only, not a selection)
    seeds = {}
    for seed in (1, 2, 3, 4):
        def mk(seed=seed):
            return GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                              subsample=0.8, min_samples_leaf=10, random_state=seed)
        res, info = tc.run_cv(mk, tab, preps_by, "y_match", feats)
        f = core.fast_pooled([preps_by[n] for n in names], [res[n][0] >= res[n][1] for n in names])
        seeds[seed] = (round(f[0], 4), info["A"]["threshold"], info["B"]["threshold"])
    print("seed stability (fast = official scorer) CV F1, thrA, thrB:", seeds)

    # what the CV verifier keeps, by length bin
    cv = json.load(open(core.WORK / "cv_probs.json"))["V_gb_match_nested"]
    rows = []
    for q in qs:
        p, t = cv[q["name"]]
        d = tab[tab.quarter == q["name"]].copy()
        d["keep"] = np.asarray(p) >= t
        rows.append(d)
    d = pd.concat(rows)
    d = d[~d.uns]
    d["bin"] = pd.cut(d.length_px, [0, 10, 15, 20, 30, 60, 1e4], right=False)
    g = d.groupby("bin", observed=True).agg(n=("keep", "size"), pos=("y_match", "sum"),
                                            kept=("keep", "sum"),
                                            kept_pos=("y_match", lambda s: int((s & d.loc[s.index, "keep"]).sum())))
    g["m20_kept"] = [int(b.left >= 20) * n for b, n in zip(g.index, g.n)]
    print(g.to_string())
    cc = d[d.corner_contact > 0]
    print(f"corner-contact pieces (zig-zag/V arms): {len(cc)}, positives {cc.y_match.sum()}, "
          f"kept {cc.keep.sum()}, kept positives {(cc.keep & (cc.y_match == 1)).sum()}")
    imp = pd.Series(clf.feature_importances_, index=feats).sort_values(ascending=False)
    print("feature importances (final model):")
    print(imp.round(3).head(20).to_string())


if __name__ == "__main__":
    main()
