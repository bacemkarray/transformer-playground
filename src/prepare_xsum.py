import json, os, random
from datasets import load_dataset
from pathlib import Path

OUT = Path("data")
OUT.mkdir(parents=True, exist_ok=True)

# Sizes (override if you want)
TRAIN_N = int(os.environ.get("TRAIN_N", "50000"))
VAL_N   = int(os.environ.get("VAL_N", "2000"))
EVAL_N  = int(os.environ.get("EVAL_N", "11334"))
SEED    = int(os.environ.get("SEED", "42"))



PROMPT_TEMPLATE = (
    "You are a helpful assistant. Summarize the following news article into one, "
    "concise sentence capturing the main point.\n\nArticle:\n{document}\n\nSummary:"
)


def prepare_subset(split_name, ds, n, out_path, seed):
    ids = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(ids)
    ids = ids[:n]

    with out_path.open("w", encoding="utf-8") as f:
        for idx in ids:
            ex = ds[idx]
            article = ex["document"].strip()
            reference = ex["summary"].strip()
            prompt = PROMPT_TEMPLATE.format(document=article)
            rec = {
                "id": f"{split_name}-{idx}",
                "document": article,
                "reference": reference,
                "prompt": prompt,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")



def main():
    # XSum provides dedicated splits
    train_ds = load_dataset("EdinburghNLP/xsum", split="train", trust_remote_code=True)
    val_ds = load_dataset("EdinburghNLP/xsum", split="validation", trust_remote_code=True)
    test_ds = load_dataset("EdinburghNLP/xsum", split="test", trust_remote_code=True)

    prepare_subset("train", train_ds, TRAIN_N, OUT / "train.jsonl", SEED)
    prepare_subset("val",   val_ds,   VAL_N,   OUT / "val.jsonl",   SEED)
    prepare_subset("eval",  test_ds,  EVAL_N,  OUT / "eval.jsonl",  SEED)


    print(f"Wrote {TRAIN_N} -> {OUT/'train.jsonl'}")
    print(f"Wrote {VAL_N}   -> {OUT/'val.jsonl'}")
    print(f"Wrote {EVAL_N}  -> {OUT/'eval.jsonl'}")

if __name__ == "__main__":
    main()