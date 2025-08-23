import os, json, time
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

RUNS = Path("runs")
DATA = Path("data/eval.jsonl")

# Decoding params — keep these fixed across all runs for apples-to-apples
GEN_KW = {
    "max_new_tokens": 64,
    "do_sample": False,
    "temperature": 0.0,
    "top_p": 1.0,
    "no_repeat_ngram_size": 3,
}


def load_base(model_name: str, dtype: str = "bfloat16", device_map: str = "auto"):
    torch_dtype = dict(float16=torch.float16, bfloat16=torch.bfloat16, float32=torch.float32)[dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def attach_adapter(model, adapter_path: Optional[str]):
    if not adapter_path:
        return model
    # Load LoRA/QLoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    # Merge for faster inference and to avoid PEFT dependency at gen time
    try:
        model = model.merge_and_unload()
    except Exception:
        # Some PEFT configs may not support merge; fall back to non-merged
        pass
    return model


def main():
    """
    Required env vars:
      MODEL_NAME      - base or fine-tuned base (e.g., "mistralai/Mistral-7B-Instruct-v0.3")
      RUN_NAME        - a short tag for this generation run (e.g., base, lora, qlora)
    Optional:
      ADAPTER_PATH    - path to a PEFT adapter dir (LoRA/QLoRA)
      DTYPE           - float16 | bfloat16 | float32 (default bfloat16)
      DEVICE_MAP      - e.g., "auto"
      LIMIT           - cap examples for quick smoke tests
    """
    model_name = os.environ["MODEL_NAME"]
    run_name   = os.environ["RUN_NAME"]
    adapter    = os.environ.get("ADAPTER_PATH")
    dtype      = os.environ.get("DTYPE", "bfloat16")
    device_map = os.environ.get("DEVICE_MAP", "auto")
    limit      = int(os.environ.get("LIMIT", "0"))

    out_dir = RUNS / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_jsonl = out_dir / "predictions.jsonl"
    pred_csv   = out_dir / "predictions.csv"

    model, tok = load_base(model_name, dtype=dtype, device_map=device_map)
    # model = attach_adapter(model, adapter)
    model.eval()

    # Determinism
    torch.manual_seed(0)

    t0 = time.time()
    n = 0
    with open(DATA, "r", encoding="utf-8") as f, \
         open(pred_jsonl, "w", encoding="utf-8") as fout:

        for line in f:
            rec = json.loads(line)
            prompt = rec["prompt"]
            inputs = tok(prompt, return_tensors="pt", truncation=True,
                         max_length=tok.model_max_length).to(model.device)

            with torch.no_grad():
                out = model.generate(**inputs, **GEN_KW)

            # Only decode newly generated tokens
            gen_ids = out[0, inputs["input_ids"].shape[1]:]
            pred = tok.decode(gen_ids, skip_special_tokens=True).strip()

            fout.write(json.dumps({
                "id": rec["id"],
                "reference": rec["reference"],
                "prediction": pred,
                "prompt_len": int(inputs["input_ids"].shape[1]),
                "output_len": int(gen_ids.shape[0]),
            }, ensure_ascii=False) + "\n")

            n += 1
            if limit and n >= limit:
                break

    # Also write a CSV for eyeballing
    import csv
    with open(pred_jsonl, "r", encoding="utf-8") as f, open(pred_csv, "w", newline="", encoding="utf-8") as g:
        w = csv.writer(g)
        w.writerow(["id", "prediction", "reference"])
        for line in f:
            r = json.loads(line)
            w.writerow([r["id"], r["prediction"], r["reference"]])

    dur = time.time() - t0
    with open(out_dir / "gen_meta.json", "w", encoding="utf-8") as g:
        json.dump({
            "model_name": model_name,
            "adapter_path": adapter,
            "dtype": dtype,
            "device_map": device_map,
            "gen_kwargs": GEN_KW,
            "seconds": dur,
            "num_examples": n if not limit else min(n, limit),
        }, g, indent=2)

    print(f"Wrote {n} preds to {pred_jsonl} in {dur:.1f}s")

if __name__ == "__main__":
    main()