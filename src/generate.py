import os, json, time
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

RUNS = Path("runs")
DATA = Path("data/test.jsonl")

# fixed params
GEN_KW = {
    "max_new_tokens": 64,
    "do_sample": False,
    "no_repeat_ngram_size": 3,
}


def load_base(model_name: str, dtype: str = "bfloat16", device_map: str = "auto"):
    dtype_dict = {
    "float16":  torch.float16,
    "bfloat16": torch.bfloat16,
    "float32":  torch.float32
    }
    torch_dtype = dtype_dict[dtype]

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
      MODEL_NAME - base or fine-tuned base (e.g., "mistralai/Mistral-7B-Instruct-v0.3")
      RUN_NAME - a short tag for this generation run (e.g., base, lora, qlora)
    Optional:
      ADAPTER_PATH - path to a PEFT adapter dir (LoRA/QLoRA)
      DTYPE - float16 | bfloat16 | float32 (default bfloat16)
      DEVICE_MAP - e.g., "auto"
      LIMIT - cap examples for quick smoke tests
    """
    model_name = os.environ["MODEL_NAME"]
    run_name = os.environ["RUN_NAME"]
    adapter = os.environ.get("ADAPTER_PATH")
    dtype = os.environ.get("DTYPE", "bfloat16")
    device_map = os.environ.get("DEVICE_MAP", "auto")
    limit = int(os.environ.get("LIMIT", "11334"))

    out_dir = RUNS / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_jsonl = out_dir / "predictions.jsonl"

    model, tok = load_base(model_name, dtype=dtype, device_map=device_map)
    model = attach_adapter(model, adapter)
    model.eval()

    # Determinism
    torch.manual_seed(0)

    t0 = time.time()
    n = 0
    with open(DATA, "r", encoding="utf-8") as f, \
         open(pred_jsonl, "w", encoding="utf-8") as fout:
        
        # find list length for tqdm progress bar
        lines = f.readlines()
        if limit:
            lines = lines[:limit]

        for line in tqdm(lines, desc="Inference Progress", unit="ex"):
            rec = json.loads(line)
            prompt = rec["prompt"]

            # Max context window for Mistral is 8192
            ctx = model.config.max_position_embeddings # 8192
            gen_max = GEN_KW.get("max_new_tokens", 64)
            encode_len = ctx - gen_max # 8128

            inputs = tok(prompt, return_tensors="pt", truncation=True,
                         max_length=encode_len).to(model.device)

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

    # METADATA
    dur = (time.time() - t0)/60
    with open(out_dir / "gen_meta.json", "w", encoding="utf-8") as g:
        json.dump({
            "model_name": model_name,
            "adapter_path": adapter,
            "dtype": dtype,
            "device_map": device_map,
            "gen_kwargs": GEN_KW,
            "minutes": dur,
            "num_examples": n if not limit else min(n, limit),
        }, g, indent=2)

    print(f"Wrote {n} preds to {pred_jsonl} in {dur:.1f}s")
    

if __name__ == "__main__":
    main()