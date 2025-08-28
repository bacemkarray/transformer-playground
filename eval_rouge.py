import argparse, json, sys
from pathlib import Path

import evaluate


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser(description="Compute ROUGE on predictions JSONL.")
    ap.add_argument("--pred", required=True, help="Path to predictions JSONL with fields: id, prediction, reference")
    ap.add_argument("--out", required=True, help="Where to write metrics JSON")
    args = ap.parse_args()

    pred_path = Path(args.pred)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(pred_path)
    if not rows:
        print("No rows found in predictions file.", file=sys.stderr)
        sys.exit(1)

    preds = [r["prediction"] for r in rows]
    refs  = [r["reference"]  for r in rows]

    rouge = evaluate.load("rouge")
    scores = rouge.compute(predictions=preds, references=refs, use_stemmer=True)

    # Keep ROUGE-L (F1) as the headline metric
    result = {"Rouge-L": float(f"{scores['rougeL']:.4f}")}

    # Write the rouge metrics
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote ROUGE metrics to {out_path}: {result}")


if __name__ == "__main__":
    main()