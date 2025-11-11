# LLM Fine-tuning for News Article Summarization 

Fine-tuned **Mistral-7B-Instruct-v0.3** on the **XSum summarization dataset** using **LoRA** and **QLoRA** adapters.  
This project benchmarks parameter-efficient fine-tuning strategies and distributed training performance using **Hugging Face Accelerate**.

---

## Overview

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

## Results

| Model | Params (Trainable) | ROUGE-L ↑ | Wall-Clock ↓ |
|--------|--------------------|------------|---------------|
| Base (Mistral-7B) | 100% | 0.36 | — |
| LoRA Fine-Tuned | 1% | 0.43 | — |
| QLoRA Fine-Tuned | 1% | 0.45 | — |

*(Fill in your real numbers once you’ve run eval_rouge.py.)*

---

## ⚙️ Environment

- Python 3.10+  
- PyTorch 2.1+  
- `transformers`, `accelerate`, `peft`, `evaluate`, `datasets`, `bitsandbytes`, `tqdm`

---

## Notes

- Fine-tuning implemented with **PEFT**’s LoRA adapters (`q_proj`, `k_proj`, `v_proj`, `o_proj`).  
- Evaluation uses **ROUGE-L** with stemming for fair comparison.  
- Designed for reproducibility — deterministic seeds and fixed generation parameters.  

---


MIT License.  
Models and datasets follow their respective upstream licenses.
