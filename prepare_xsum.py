import json, os, random
from datasets import load_dataset
from pathlib import Path

OUT = Path("data")
OUT.mkdir(parents=True, exist_ok=True)

# Choose how big the eval set is:
N_SAMPLES = int(os.environ.get("N_SAMPLES", "500"))  # set to 11333 for full val
SEED = int(os.environ.get("SEED", "42"))


PROMPT_TEMPLATE = (
    "You are a helpful assistant. Summarize the following news article into one, "
    "concise sentence capturing the main point.\n\nArticle:\n{document}\n\nSummary:"
)

def main():
    ds = load_dataset("EdinburghNLP/xsum", split="validation", trust_remote_code=True)
    ids = list(range(len(ds)))
    random.Random(SEED).shuffle(ids)
    ids = ids[:N_SAMPLES]

    out_path = OUT / "eval.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for i in ids:
            ex = ds[i]
            article = ex["document"].strip()
            reference = ex["summary"].strip()
            prompt = PROMPT_TEMPLATE.format(document=article)
            rec = {
                "id": str(i),
                "document": article,
                "reference": reference,
                "prompt": prompt,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {N_SAMPLES} examples to {out_path}")

if __name__ == "__main__":
    main()