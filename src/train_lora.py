import os, json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, BitsAndBytesConfig, get_constant_schedule_with_warmup
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training, get_peft_model

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW




OUT_DIR = Path("adapters") / os.environ["OUT_DIR"]
MODEL_NAME = os.environ.get("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
USE_QLORA = os.environ.get("USE_QLORA", "false") == "true" # Leave False for bf16 LoRA
MAX_LEN = int(os.environ.get("MAX_LEN", 2048))

TRAIN_PATH = Path(os.environ.get("TRAIN_PATH", "data/train.jsonl"))
VAL_PATH = Path(os.environ.get("VAL_PATH", "data/val.jsonl"))

BATCH_SIZE = int(os.environ.get("per_device_batch_size", "4"))
EVAL_BS = int(os.environ.get("per_device_eval_batch_size", "4"))
ACCUM = int(os.environ.get("gradient_accumulation_steps", "8"))
EPOCHS = int(os.environ.get("num_train_epochs", "2"))
LR = float(os.environ.get("learning_rate", "2e-4"))
LOG_STEPS  = int(os.environ.get("logging_steps", "200"))

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

def build_tokenizer_and_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if USE_QLORA:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config, 
            device_map=None,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map=None,
        )
                                                                            
    model.config.use_cache = False
    model.enable_input_require_grads()
    model = get_peft_model(model, peft_config) #creates a peft model

    return model, tokenizer


raw = load_dataset(
    "json",
    data_files={
        "train": str(TRAIN_PATH),
        "validation": str(VAL_PATH),
    },
)

train_raw = raw["train"]
val_raw = raw["validation"]

def format_example(ex):
    # Supervised fine-tuning as causal LM: prompt + target + eos
    full = ex["prompt"] + " " + ex["reference"] + tokenizer.eos_token
    return full

def tokenize_fn(ex):
    text = format_example(ex)
    OUT_DIR = tokenizer(
        text,
        max_length=MAX_LEN,
        truncation=True,
        padding=False, # pack later via collator
    )
    return OUT_DIR

train_ds = train_raw.map(tokenize_fn, remove_columns=train_raw.column_names)
val_ds = val_raw.map(tokenize_fn, remove_columns=val_raw.column_names)

# Dynamic padding & label shifting handled by the collator
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training args
args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BS,
    gradient_accumulation_steps=ACCUM, # Accumulate 8 mini-batches of 4 to simulate size of 32
    group_by_length=True,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    logging_steps=LOG_STEPS,
    bf16=torch.cuda.is_available(),
    gradient_checkpointing=True,
    report_to="tensorboard",
    logging_dir=str(OUT_DIR / "tb")
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collator,
    tokenizer=tokenizer,
)

# Begin training loop
trainer.train()

# Save PEFT adapter only in output_dir
trainer.save_model()