"""Write results/bakeoff/devcv/round2_choice.json from DEV results only (README "Round 2", E2 declared 13:25).

    venv/Scripts/python bakeoff/round2_choose.py

Rule: primary = E2_thin+V unless E2_cv + V (cross-validated, pooled over the ORIGINAL 5 dev
quarters, devcv/manifest_r1.json) is more than GUARD below E_cv + V on the same quarters;
then primary = E_thin+V. All rows are scored either way: E2_thin+V, E2_thin+X, E_thin+V
(+ references E_thin+X and A, always added by final_round2.py).
"""
from __future__ import annotations

import json

import pandas as pd

from common import OUT, PRED, read_json

DEV = OUT / "devcv"
GUARD = 0.020
VARIANT = "V_gb_match_nested"


def pooled_f1(csv_path, quarters) -> float:
    d = pd.read_csv(csv_path)
    d = d[(d.variant == VARIANT) & d.quarter.isin(quarters)]
    if set(d.quarter) != set(quarters):
        raise SystemExit(f"{csv_path}: missing quarters {set(quarters) - set(d.quarter)}")
    R = (d.wire_recall * d.n_wires_gt).sum() / d.n_wires_gt.sum()
    P = (d.wire_precision * d.n_wires_pred).sum() / d.n_wires_pred.sum()
    return 2 * P * R / (P + R)


def main() -> None:
    q5 = [f"{c['case']}_{c['quarter']}" for c in read_json(DEV / "manifest_r1.json")["cases"]]
    q7 = [f"{c['case']}_{c['quarter']}" for c in read_json(DEV / "manifest.json")["cases"]]
    e5 = pooled_f1(DEV / "verifier_E_cv" / "cv_per_quarter.csv", q5)
    e2_5 = pooled_f1(DEV / "verifier_E2_cv" / "cv_per_quarter.csv", q5)
    e2_7 = pooled_f1(DEV / "verifier_E2_cv" / "cv_per_quarter.csv", q7)
    print(f"E_cv + V, original 5 quarters: {e5:.4f}")
    print(f"E2_cv + V, original 5 quarters: {e2_5:.4f} (guard: >= {e5 - GUARD:.4f})")
    print(f"E2_cv + V, all 7 quarters: {e2_7:.4f}")
    v_e = DEV / "verifier_E_cv" / "verifier_learned.joblib"
    v_e2 = DEV / "verifier_E2_cv" / "verifier_learned.joblib"
    for f in (v_e, v_e2, *(PRED / n for n in ("E", "E2"))):
        if not f.exists():
            raise SystemExit(f"{f} missing")
    primary = "E2_thin+V" if e2_5 >= e5 - GUARD else "E_thin+V"
    choice = {"primary": [primary, "E_thin+X"],
              "rule": f"E2_thin+V unless E2_cv+V < E_cv+V - {GUARD} on the original 5 dev quarters",
              "dev": {"E_cv+V_5q": e5, "E2_cv+V_5q": e2_5, "E2_cv+V_7q": e2_7},
              "rows": [{"name": "E2_thin+V", "pred": "E2", "extract": "verifier", "verifier": str(v_e2)},
                       {"name": "E2_thin+X", "pred": "E2", "extract": "lsd_m20"},
                       {"name": "E_thin+V", "pred": "E", "extract": "verifier", "verifier": str(v_e)}]}
    out = DEV / "round2_choice.json"
    if out.exists():
        raise SystemExit(f"{out} exists")
    out.write_text(json.dumps(choice, indent=1), encoding="utf-8")
    print(f"primary: {primary}; wrote {out}")


if __name__ == "__main__":
    main()
