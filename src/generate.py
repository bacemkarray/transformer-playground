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

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


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
    batch_size = int(os.environ.get("BATCH_SIZE", "8"))

    out_dir = RUNS / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_jsonl = out_dir / "predictions.jsonl"

    model, tok = load_base(model_name, dtype=dtype, device_map=device_map)
    model = attach_adapter(model, adapter)
    model.eval()

    # Update GEN_KW to include tokenizer IDs
    GEN_KW.update({
    "pad_token_id": tok.eos_token_id,
    "eos_token_id": tok.eos_token_id,  # Mistral’s eos is 2
    })
    
    # Max context window for Mistral is 8192
    ctx = model.config.max_position_embeddings # 8192
    gen_max = GEN_KW.get("max_new_tokens", 64)
    encode_len = ctx - gen_max # 8128
    
    # Determinism
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    t0 = time.time()
    n = 0
    with open(DATA, "r", encoding="utf-8") as f, \
        open(pred_jsonl, "w", encoding="utf-8") as fout:
        
        # find list length for tqdm progress bar
        lines = f.readlines()
        lines.sort(key=lambda s: len(json.loads(s)["prompt"])) # sort to prevent major padding during inference
        if limit:
            lines = lines[:limit]

        num_batches = (len(lines) + batch_size - 1) // batch_size

        with torch.inference_mode():
            for batch_lines in tqdm(chunks(lines, batch_size), desc="Generating", unit="batch", total=num_batches):
                batch = [json.loads(s) for s in batch_lines]
                prompts = [r["prompt"] for r in batch]

                enc = tok(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=encode_len,
                ).to(model.device)

                out = model.generate(
                    **enc,
                    **GEN_KW,
                    return_dict_in_generate=True,
                )

                # lengths of each prompt in tokens (per row)
                in_lens = enc["attention_mask"].sum(-1).tolist()
                seqs = out.sequences  # (B, prompt+gen)

                preds = []
                out_token_lens = []
                for row, in_len in zip(seqs, in_lens):
                    gen_ids = row[in_len:]
                    out_token_lens.append(int(gen_ids.shape[0]))
                    preds.append(tok.decode(gen_ids, skip_special_tokens=True).strip())

                # write this batch
                buf = []
                for rec, pred, in_len, out_len in zip(batch, preds, in_lens, out_token_lens):
                    buf.append(json.dumps({
                        "id": rec["id"],
                        "reference": rec["reference"],
                        "prediction": pred,
                        "prompt_len": int(in_len),
                        "output_len": out_len,
                    }, ensure_ascii=False))
                fout.write("\n".join(buf) + "\n")

                n += len(batch)
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

    print(f"Wrote {n} preds to {pred_jsonl} in {dur:.1f}m")
    

if __name__ == "__main__":
    main()