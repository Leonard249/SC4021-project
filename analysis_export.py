"""
analysis_export.py
------------------
Reads classified_eval_new.json and exports 4 grouped analysis JSON files
into data/analysis/:

  positive_analysis.json
  negative_analysis.json
  neutral_analysis.json
  irrelevant_analysis.json

Each file has the structure:
{
  "true_label": "<class>",
  "total": N,
  "correct": { "count": N, "records": [...] },
  "wrong": {
    "total": N,
    "<predicted_class>": { "count": N, "records": [...] },
    ...
  }
}
"""

import json
from pathlib import Path
from collections import defaultdict

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent
INPUT_PATH  = BASE / "data" / "results" / "classified_eval_new.json"
OUTPUT_DIR  = BASE / "data" / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["positive", "negative", "neutral", "irrelevant"]

# ─── Label resolution (same logic as evaluate_pipeline.py) ────────────────────

def get_predicted(record: dict) -> str:
    if record.get("Subjectivity") == "Irrelevant":
        return "irrelevant"
    return (record.get("Overall_Document_Polarity") or "neutral").lower()

def get_true(record: dict) -> str:
    return (record.get("label") or "").lower()

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    with open(INPUT_PATH, encoding="utf-8") as f:
        records = json.load(f)

    print(f"Loaded {len(records)} records.\n")

    # Group records by true label
    by_true: dict[str, list] = defaultdict(list)
    for r in records:
        by_true[get_true(r)].append(r)

    for true_cls in CLASSES:
        cls_records = by_true.get(true_cls, [])

        correct_records = []
        wrong_by_pred: dict[str, list] = defaultdict(list)

        for r in cls_records:
            pred = get_predicted(r)
            if pred == true_cls:
                correct_records.append(r)
            else:
                wrong_by_pred[pred].append(r)

        # Build wrong section — one key per predicted class (only those that appear)
        wrong_section = {"total": sum(len(v) for v in wrong_by_pred.values())}
        for pred_cls in CLASSES:
            if pred_cls == true_cls:
                continue
            if pred_cls in wrong_by_pred:
                wrong_section[f"predicted_{pred_cls}"] = {
                    "count":   len(wrong_by_pred[pred_cls]),
                    "records": wrong_by_pred[pred_cls],
                }

        output = {
            "true_label": true_cls,
            "total":      len(cls_records),
            "correct": {
                "count":   len(correct_records),
                "records": correct_records,
            },
            "wrong": wrong_section,
        }

        out_path = OUTPUT_DIR / f"{true_cls}_analysis.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        # Summary
        wrong_total = wrong_section["total"]
        print(f"  {true_cls}_analysis.json")
        print(f"    Total : {len(cls_records)}")
        print(f"    ✓ Correct   : {len(correct_records)}")
        print(f"    ✗ Wrong     : {wrong_total}")
        for pred_cls in CLASSES:
            key = f"predicted_{pred_cls}"
            if key in wrong_section:
                print(f"        → {pred_cls:<12}: {wrong_section[key]['count']}")
        print()

    print(f"✅ Written to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
