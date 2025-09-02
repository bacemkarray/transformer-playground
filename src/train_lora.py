from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import LoraModel, LoraConfig, TaskType, prepare_model_for_kbit_training, get_peft_model
import torch
from datasets import load_dataset
import os, json
from pathlib import Path


out = Path("adapters") / os.environ["out"]
model_name = os.environ.get("model_name", "mistralai/Mistral-7B-Instruct-v0.3")
use_qlora = os.environ.get("use_qlora", "false") == "true" # Leave False for bf16 LoRA
max_len = int(os.environ.get("max_len", 2048))

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


tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if use_qlora:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config, 
        device_map="none",
    )
    model = prepare_model_for_kbit_training(model)
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="none",
    )
                                                                        

model.config.use_cache = False
model.enable_input_require_grads()
model = get_peft_model(model, peft_config) #creates a peft model


train_path = Path(os.environ.get("train_path", "data/train.jsonl"))
val_path = Path(os.environ.get("val_path", "data/val.jsonl"))

raw = load_dataset(
    "json",
    data_files={
        "train": str(train_path),
        "validation": str(val_path),
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
    out = tokenizer(
        text,
        max_length=max_len,
        truncation=True,
        padding=False, # pack later via collator
    )
    return out

train_ds = train_raw.map(tokenize_fn, remove_columns=train_raw.column_names)
val_ds = val_raw.map(tokenize_fn, remove_columns=val_raw.column_names)

# Dynamic padding & label shifting handled by the collator
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training args
args = TrainingArguments(
    output_dir=out,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8, # Accumulate 8 mini-batches of 4 to simulate size of 32
    group_by_length=True,
    num_train_epochs=2,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    logging_steps=200,
    bf16=torch.cuda.is_available(),
    gradient_checkpointing=True,
    report_to="tensorboard",
    logging_dir=str(out / "tb")
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