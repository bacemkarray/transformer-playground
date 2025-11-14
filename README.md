# LLM Fine‑Tuning Benchmark: LoRA vs QLoRA on XSum

This project benchmarks **parameter‑efficient fine‑tuning** strategies on a real summarization task.  
The focus is on how much quality a 7B model can recover by training a very small fraction of its weights, and how quantized adapters (QLoRA) compare to standard LoRA in both quality and efficiency.

---

## 1. Purpose

Large language models are expensive to fully fine‑tune.  
This benchmark tests how much performance you can recover by freezing the base model and only training low-rank adapter layers.

The goals:

- Measure the quality gains achievable with LoRA and QLoRA under heavy parameter freezing.  
- Compare adapter methods directly on a strict task (BBC XSum summarization).  
- Quantify training efficiency improvements, including multi‑GPU scaling.

---

## 2. Why Adapters?

LoRA and QLoRA replace full‑model fine‑tuning with a small, trainable low‑rank decomposition injected into key transformer projections.

Key advantages:

- **Massive parameter reduction** - train <1% of a 7B model.  
- **Lower VRAM requirements** – frozen base weights and 4-bit quantization (QLoRA) reduce memory footprint.
- **Reduced optimization overhead** – only the adapter weights receive gradients or optimizer updates.
- **No degradation in downstream task quality** for many summarization workloads.

This makes adapters ideal for studying *how much performance you can recover with minimal compute*.

---

## 3. Benchmark Axes

This project evaluates two core dimensions:

### **Axis A – Parameter-Efficiency**
How much ROUGE-L improvement can be recovered by fine-tuning:
- **LoRA adapters** (bf16 base)
- **QLoRA adapters** (4-bit NF4 base)

while freezing ~99% of the model parameters.

### **Axis B – Training Efficiency**
How training speed and behavior differ between:
- **Single-GPU vs multi-GPU DDP**
- **LoRA vs QLoRA training throughput**

Together, these measurements show how adapter type and GPU configuration affect both throughput and final task quality.

---

## 4. Pipeline Overview

High‑level workflow:

1. **Dataset Preparation**  
   - XSum train/val/test splits preprocessed into JSONL.  
   - Prompts built using a consistent summarization instruction template.

2. **Adapter‑Based Fine‑Tuning**  
   - LoRA or QLoRA adapters applied to `q_proj`, `k_proj`, `v_proj`, and `o_proj`.  
   - Gradient checkpointing enabled.  
   - Base weights frozen.

3. **Distributed Training (Accelerate)**  
   - Multi‑GPU Data Parallelism.  
   - Cosine LR schedule.  
   - Mixed precision (bf16 / 4BUD‑bit compute).

4. **Batch‑Sorted Inference**  
   - Sorting by input token length minimizes padding and leads to higher throughput.

5. **Evaluation (ROUGE‑L)**  
   - Predictions compared to reference summaries using stemmed ROUGE‑L.

---

## 5. Results

### **Model Quality (ROUGE‑L)**  

| Model | Trainable Params | ROUGE‑L |
|-------|------------------|---------|
| **Mistral‑7B Base** | ~7.3 Billion | `0.1907` |
| **LoRA Fine‑Tuned** | 6,815,744 | `0.2289` |
| **QLoRA Fine‑Tuned** | 6,815,744 | `0.2283` |

---

### **Training Efficiency**

| Configuration | Wall‑Clock Time | Speedup |
|--------------|-----------------|---------|
| **1× GPU (LoRA)** | `8:57:37` | — |
| **2× GPU DDP (LoRA)** | `4:20:44` | ~52% |
| **QLoRA 1× GPU** | `11:07:59` | — |

---

### **VRAM Usage (1× GPU)**

| Configuration       | VRAM Usage     | Reduction |
|---------------------|----------------|-----------|
| **LoRA (bf16)**     | `21,576 MiB`   | —         |
| **QLoRA (4-bit NF4)**| `14,526 MiB`   | ~33%  |

QLoRA reduces VRAM usage by roughly **33%**, allowing the same 7B model to fine-tune comfortably on smaller GPUs, at the cost of longer training times.

---

## 6. Methodological Choices

Some key decisions that define this benchmark:

- **Adapter Targets:** `q_proj`, `k_proj`, `v_proj`, `o_proj`  
  These layers dominate attention transformations; adapting them yields meaningful behavior change without tuning the entire network.

- **NF4 Quantization (QLoRA):**  
  Chosen for its strong empirical performance and minimal degradation on summarization tasks.

- **Cosine LR Schedule:**  
  Safe, smooth decay for adapter‑based fine‑tuning.

- **Sorted Prompts During Inference:**  
  Reduces excessive padding tokens, leading to more efficient computations.

---

## 7. Insights

- LoRA delivers the highest task quality with minimal parameter updates.  
- QLoRA preserves nearly identical performance to LoRA’s, despite 4‑bit quantization. This comes at the cost of increased training time, however.  
- Distributed training significantly reduces end‑to‑end training time.  
- Summarization tasks benefit heavily from adapting attention projections.  
- Minimizing padding by sorting batches by length improves throughput in both training and inference.
