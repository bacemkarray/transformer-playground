import os, json, math, time
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
from torch.optim.lr_scheduler import CosineAnnealingLR




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

def load_json_ds():
    raw = load_dataset(
        "json",
        data_files={"train": str(TRAIN_PATH), "validation": str(VAL_PATH)},
    )
    return raw["train"], raw["validation"]


def format_example(ex, tokenizer):
    # Supervised fine-tuning as causal LM: prompt + target + eos
    full = ex["prompt"] + " " + ex["reference"] + tokenizer.eos_token
    return full


def make_tokenize_fn(tokenizer):
    def _fn(ex):
        text = format_example(ex, tokenizer)
        out = tokenizer(
            text,
            max_length=MAX_LEN,
            truncation=True,
            padding=False, # pack later via collator
        )
        return out
    return _fn


def evaluate(model, dataloader, accelerator):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            losses.append(accelerator.gather(loss.detach()).mean())
    model.train()
    if len(losses) == 0:
        return float("inf")
    return torch.stack(losses).mean().item()


def main():
    set_seed(42)

    project_config = ProjectConfiguration(project_dir=str(OUT_DIR), logging_dir=str(OUT_DIR/"tb"))
    accelerator = Accelerator(
        gradient_accumulation_steps=ACCUM,
        mixed_precision="bf16" if torch.cuda.is_available() else "no",
        log_with="tensorboard",
        project_config=project_config,
    )

    # Tokenizer / model
    tokenizer, model = build_tokenizer_and_model()

    # Data
    train_raw, val_raw = load_json_ds()
    tokenize_fn = make_tokenize_fn(tokenizer)
    train_ds = train_raw.map(tokenize_fn, remove_columns=train_raw.column_names)
    val_ds   = val_raw.map(tokenize_fn,   remove_columns=val_raw.column_names)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=EVAL_BS,   shuffle=False, collate_fn=collator, pin_memory=True)

    # Optimizer + scheduler
    optim = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    num_update_steps_per_epoch = math.ceil(len(train_dl) / ACCUM)
    t_total = num_update_steps_per_epoch * EPOCHS
    lr_sched = CosineAnnealingLR(optim, T_max=t_total)

    # Prepare for DDP
    model, optim, train_dl, val_dl, lr_sched = accelerator.prepare(
        model, optim, train_dl, val_dl, lr_sched
    )

    best_val = float("inf")
    global_step = 0
    t0 = time.time()

    for epoch in range(EPOCHS):
        model.train()
        for step, batch in enumerate(train_dl):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optim.step()
                lr_sched.step()
                optim.zero_grad()
                global_step += 1

                if accelerator.is_main_process and (global_step % LOG_STEPS == 0):
                    accelerator.log({"train/loss": loss.item(), "train/step": global_step}, step=global_step)
        
        # Eval at epoch end
        val_loss = evaluate(model, val_dl, accelerator)
        if accelerator.is_main_process:
            accelerator.log({"eval/loss": val_loss, "epoch": epoch + 1}, step=global_step)
            # Save best
            if val_loss < best_val:
                best_val = val_loss
                save_dir = OUT_DIR / "best"
                save_dir.mkdir(parents=True, exist_ok=True)
                # save PEFT adapter; unwrap to save from the real module
                accelerator.unwrap_model(model).save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)

    # Final save
    if accelerator.is_main_process:
        final_dir = OUT_DIR / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        with open(OUT_DIR / "train_meta.json", "w", encoding="utf-8") as f:
            json.dump({
                "model_name": MODEL_NAME,
                "use_qlora": USE_QLORA,
                "epochs": EPOCHS,
                "lr": LR,
                "accum": ACCUM,
                "batch_size": BATCH_SIZE,
                "updates_per_epoch": num_update_steps_per_epoch,
                "total_updates": t_total,
                "minutes": (time.time() - t0) / 60.0,
                "best_val_loss": best_val,
            }, f, indent=2)


if __name__ == "__main__":
    main()