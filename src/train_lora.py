from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraModel, LoraConfig, TaskType, prepare_model_for_kbit_training, get_peft_model
import torch

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



# model = prepare_model_for_kbit_training(model)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
model = get_peft_model(model, peft_config) #creates a peft model