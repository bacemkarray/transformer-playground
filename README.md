# LLM Fine‑Tuning Benchmark: LoRA vs QLoRA on XSum

This project benchmarks **parameter‑efficient fine‑tuning** strategies on a real summarization task.  
The focus is on how much quality a 7B model can recover by training **<1%** of its weights, and how quantized adapters (QLoRA) compare to standard LoRA in both quality and efficiency.

---

## 1. Purpose

Large language models are expensive to fully fine‑tune.  
This benchmark tests how much performance you can recover by freezing the 7B base model and only training low-rank adapter layers.

The goals:

- Measure the quality gains achievable with LoRA and QLoRA under heavy parameter freezing.  
- Compare adapter methods directly on a strict task (BBC XSum summarization).  
- Quantify training efficiency improvements, including multi‑GPU scaling.

---

## 2. Why Adapters?

LoRA and QLoRA replace full‑model fine‑tuning with a small, trainable low‑rank decomposition injected into key transformer projections.

Key advantages:

- **Massive parameter reduction** - train <1% of a 7B model.  
- **Lower VRAM requirements** - especially with QLoRA’s 4‑bit NF4 base weights.  
- **Faster training** - smaller gradient updates, less communication overhead.  
- **No degradation in downstream task quality** for many summarization workloads.

This makes adapters ideal for studying *how much performance you can recover with minimal compute*.

---

## 3. Benchmark Axes

This project evaluates two core dimensions:

### **Axis A — Parameter‑Efficiency**
How much ROUGE‑L improvement can we extract by fine‑tuning:
- **LoRA adapters** (bf16 base weights)  
- **QLoRA adapters** (4‑bit NF4 base weights)  

while freezing ~99% of model parameters?

### **Axis B — Training Efficiency**
How much wall‑clock speed do we gain from:
- **Single‑GPU vs multi‑GPU DDP**  
- **Adapter weights vs full‑precision training**  
- **Sorted batch inference optimizations**

Results highlight the scaling behavior of a real 7B model on commodity multi‑GPU hardware.

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
   - Mixed precision (bf16 / 4‑bit compute).

4. **Batch‑Sorted Inference**  
   - Sorting by input token length minimizes padding leads to higher throughput.

5. **Evaluation (ROUGE‑L)**  
   - Predictions compared to reference summaries using stemmed ROUGE‑L.

---

## 5. Results

### **Model Quality (ROUGE‑L)**  

| Model | Trainable Params | ROUGE‑L | Notes |
|-------|------------------|---------|-------|
| **Mistral‑7B Base** | ~7.3 Billion (frozen at inference) | `0.1907` | Baseline |
| **LoRA Fine‑Tuned** | 6815744 | `0.2289` | Strongest task performance |
| **QLoRA Fine‑Tuned** | 6815744 | `0.2283` | Nearly matches LoRA despite 4‑bit base |

---

### **Training Efficiency**

| Configuration | Wall‑Clock Time | Speedup | Notes |
|--------------|-----------------|---------|-------|
| **1× GPU (LoRA)** | `8:57:37` | — | Baseline |
| **2× GPU DDP (LoRA)** | `4:20:44` | ~52% | Substantial scaling efficiency |
| **QLoRA 1× GPU** | `11:07:59` | — | Lower VRAM usage but longer training time |

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
- QLoRA preserves most of LoRA’s performance despite 4‑bit quantization.  
- Distributed training significantly reduces end‑to‑end training time.  
- Summarization tasks benefit heavily from adapting attention projections, even when almost the entire model is frozen.  
- Padding minimization (sorted batches) improves throughput in both training and inference.