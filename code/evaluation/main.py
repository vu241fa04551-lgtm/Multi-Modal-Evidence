"""
code/evaluation/main.py — Offline Local Accuracy Analyzer (Zero API Cost Calls)
"""

import sys
import json
import argparse
import pathlib
import pandas as pd

HERE      = pathlib.Path(__file__).parent.resolve()   # code/evaluation/
CODE_DIR  = HERE.parent                               # code/
ROOT      = CODE_DIR.parent                           # repo root

DEFAULT_GT   = ROOT / "dataset"    / "sample_claims.csv"
DEFAULT_PRED = ROOT / "evaluation" / "sample_output.csv"
DIFF_CSV     = ROOT / "evaluation" / "eval_diff.csv"
REPORT_JSON  = ROOT / "evaluation" / "eval_report.json"

PRIMARY   = ["claim_status", "issue_type", "severity"]
SECONDARY = ["evidence_standard_met", "valid_image", "object_part"]

def norm(val) -> str:
    if pd.isna(val): return ""
    return str(val).strip().lower()

def evaluate_predictions(gt_path: pathlib.Path, pred_path: pathlib.Path, no_diff: bool = False) -> dict:
    if not gt_path.exists() or not pred_path.exists():
        print("Error: Ground truth or prediction source tables are missing.")
        return {}

    gt = pd.read_csv(gt_path)
    pred = pd.read_csv(pred_path)

    merged = pd.merge(gt, pred, on=["user_id", "claim_object"], suffixes=("_gt", "_pred"))
    total = len(merged)
    
    if total == 0:
        print("Error: Unable to align matching record index mappings across columns.")
        return {}

    report = {"total_rows": total, "metrics": {}}

    print("\n" + "=" * 58)
    print("📋 HACKERRANK EVALUATION ACCURACY MATRIX REPORT")
    print("=" * 58)

    for metric in PRIMARY + SECONDARY:
        gt_col, pred_col = f"{metric}_gt", f"{metric}_pred"
        if gt_col in merged and pred_col in merged:
            matches = sum(norm(r[gt_col]) == norm(r[pred_col]) for _, r in merged.iterrows())
            acc = matches / total
            report["metrics"][metric] = {"matches": matches, "total": total, "accuracy": acc}
            if metric in PRIMARY:
                print(f"  • {metric:<25} -> {matches}/{total} matches | Accuracy: {acc*100:.2f}%")

    if not no_diff:
        rows = []
        for _, row in merged.iterrows():
            dr = {"user_id": row.get("user_id", "?")}
            for m in PRIMARY + SECONDARY:
                gc, pc = m + "_gt", m + "_pred"
                if gc in merged and pc in merged:
                    dr[m + "_match"] = (norm(row[gc]) == norm(row[pc]))
                    dr[m + "_gt"]    = row[gc]
                    dr[m + "_pred"]  = row[pc]
            rows.append(dr)
        pd.DataFrame(rows).to_csv(str(DIFF_CSV), index=False)

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(str(REPORT_JSON), "w") as f:
        json.dump(report, f, indent=2)
    return report

def main() -> None:
    ap = argparse.ArgumentParser(description="Offline evaluation score counter")
    ap.add_argument("--ground-truth", type=pathlib.Path, default=DEFAULT_GT)
    ap.add_argument("--predictions",  type=pathlib.Path, default=DEFAULT_PRED)
    ap.add_argument("--no-diff", action="store_true")
    args = ap.parse_args()
    evaluate_predictions(args.ground_truth, args.predictions, args.no_diff)

if __name__ == "__main__":
    main()