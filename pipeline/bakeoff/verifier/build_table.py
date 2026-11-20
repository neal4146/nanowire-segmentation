"""Features + labels for every LSD piece of the 5 dev quarters; check the fast scorer
against the official one on the min-length rules and on the label oracles."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402
import time  # noqa: E402
import pickle  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import verifier_learned as vl  # noqa: E402


def main():
    qs = core.load_cache()
    preps, tabs = [], []
    for q in qs:
        t = time.time()
        X = vl.piece_features(q["pieces"], q["prob"], q["img"])
        pr = core.prep_quarter(q)
        dup = X[:, vl.FEATURES.index("dup_flag")] > 0
        # spec label: >= 70% within 4 px of one GT line and not a parallel duplicate of a longer piece
        fwd_line = pr["y_spec"]  # longest-per-line version, kept for reference
        df = pd.DataFrame(X, columns=vl.FEATURES)
        df["quarter"] = q["name"]
        df["fold"] = q["fold"]
        df["piece"] = np.arange(len(X))
        df["uns"] = pr["uns"]
        df["y_match"] = pr["y_match"]
        df["y_spec_longest"] = fwd_line
        preps.append(pr)
        tabs.append(df)
        print(f"{q['name']}: {len(X)} pieces, {pr['uns'].sum()} unsure, match+ {pr['y_match'].sum()}, "
              f"n_gt {pr['n_gt']}, {time.time() - t:.1f}s", flush=True)
    # spec label (task text): near70 on one line and not a duplicate of a longer piece
    for q, pr, df in zip(qs, preps, tabs):
        fwd, bwd, gl, near70 = core.coverage(q["pieces"], q["ref"]["_instances"])
        on_line = near70.max(0) >= 0.7
        df["y_spec"] = (on_line & (df.dup_flag.values == 0)).astype(int)
        df["maxC"] = pr["C"].max(0) if pr["C"].shape[0] else 0.0
    tab = pd.concat(tabs, ignore_index=True)
    tab.to_pickle(core.WORK / "piece_table.pkl")          # exact features (CSV only to read)
    tab.to_csv(core.WORK / "piece_table.csv", index=False, float_format="%.5g")
    with open(core.WORK / "preps.pkl", "wb") as fh:
        pickle.dump(preps, fh, protocol=4)

    # fast vs official on fixed rules
    rows = []
    for name, rule in [("lsd_m10", lambda d: d.length_px >= 10), ("lsd_m15", lambda d: d.length_px >= 15),
                       ("lsd_m20", lambda d: d.length_px >= 20), ("lsd_m25", lambda d: d.length_px >= 25),
                       ("lsd_m0", lambda d: d.length_px >= 0),
                       ("oracle_match", lambda d: d.y_match == 1), ("oracle_spec", lambda d: d.y_spec == 1),
                       ("oracle_cover", lambda d: d.maxC >= 0.5)]:
        keeps = [rule(df).values for df in tabs]
        f = core.fast_pooled(preps, keeps)
        kept = [[pc for pc, k in zip(q["pieces"], kp) if k] for q, kp in zip(qs, keeps)]
        off, _, _ = core.official(qs, kept, name)
        rows.append({"rule": name, "fast_f1": f[0], "fast_R": f[1], "fast_P": f[2],
                     "off_f1": off["wire_f1"], "off_R": off["wire_recall"], "off_P": off["wire_precision"]})
        print(rows[-1], flush=True)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(tab.groupby("quarter")[["y_match", "y_spec", "uns"]].sum())


if __name__ == "__main__":
    main()
