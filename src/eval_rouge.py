import json, os, sys
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
    pred_path = Path(os.environ["PRED"]) # Where to ingest the data from
    out_path = Path(os.environ["OUT"]) # Where to write the rouge score
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(pred_path)
    if not rows:
        print("No rows found in predictions file.", file=sys.stderr)
        sys.exit(1)

    preds = [r["prediction"] for r in rows]
    refs  = [r["reference"]  for r in rows]

    rouge = evaluate.load("rouge")
    scores = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    result = {"Rouge-L": float(f"{scores['rougeL']:.4f}")}

    # Write the rouge metrics
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote ROUGE metric to {out_path}: {result}")


if __name__ == "__main__":
    main()