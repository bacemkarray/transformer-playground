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