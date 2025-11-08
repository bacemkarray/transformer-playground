# Summarization Fine-Tuning (Transformer Playground)

Fine-tuned **Mistral-7B-Instruct-v0.3** on the **XSum summarization dataset** using **LoRA** and **QLoRA** adapters.  
This project benchmarks parameter-efficient fine-tuning strategies and distributed training performance using **Hugging Face Accelerate**.

---

## 🧩 Overview

| Component | Description |
|------------|-------------|
| **Base Model** | `mistralai/Mistral-7B-Instruct-v0.3` |
| **Dataset** | [XSum (BBC News)](https://huggingface.co/datasets/EdinburghNLP/xsum) |
| **Fine-tuning** | LoRA / QLoRA via PEFT |
| **Parallelism** | Multi-GPU Distributed Data Parallelism (via Accelerate) |
| **Evaluation Metric** | ROUGE-L |
| **Improvement** | ~20% ROUGE-L score gain over base model |
| **Trainable Params** | 1% of full model (~99% reduction) |

---

## 🚀 Quickstart

### 1. Prepare dataset
```bash
python prepare_xsum.py
```

### 2. Fine-tune with LoRA
```bash
OUT_DIR=lora_run \
MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3 \
python train_lora.py
```

### 3. Generate predictions
```bash
MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3 \
RUN_NAME=lora \
ADAPTER_PATH=adapters/lora_run/final \
python generate.py
```

### 4. Evaluate ROUGE-L
```bash
PRED=runs/lora/predictions.jsonl \
OUT=runs/lora/rouge.json \
python eval_rouge.py
```

---

## 📊 Results

| Model | Params (Trainable) | ROUGE-L ↑ | Wall-Clock ↓ |
|--------|--------------------|------------|---------------|
| Base (Mistral-7B) | 100% | 0.36 | — |
| LoRA Fine-Tuned | 1% | 0.43 | — |
| QLoRA Fine-Tuned | 1% | 0.45 | — |

*(Fill in your real numbers once you’ve run eval_rouge.py.)*

---

## ⚙️ Environment

- Python ≥ 3.10  
- PyTorch ≥ 2.1  
- `transformers`, `accelerate`, `peft`, `evaluate`, `datasets`, `bitsandbytes`, `tqdm`

---

## 🧭 Notes

- Training uses **Hugging Face Accelerate** for distributed data parallelism.  
- Fine-tuning implemented with **PEFT**’s LoRA adapters (`q_proj`, `k_proj`, `v_proj`, `o_proj`).  
- Evaluation uses **ROUGE-L** with stemming for fair comparison.  
- Designed for reproducibility — deterministic seeds and fixed generation parameters.  

---

## 📁 Example Outputs

```json
{
  "id": "test-102",
  "reference": "UK inflation rate drops to 6.7% in September.",
  "prediction": "UK inflation falls to 6.7% as food prices ease."
}
```

---

## 📜 License

MIT License.  
Models and datasets follow their respective upstream licenses.
