transformer-playground

LoRA on one gpu is eating up 21576MiB of VRAM
QLoRA on one gpu is eating up 14526MiB of VRAM

This means a VRAM % reduction of 32.6751947% or just ~33% 



Total runtime for mistral lora was 8:57:37
Total runtime for mistral qlora was 11:07:59

LORA SCORE 0.1752
QLORA SCORE 0.1746

Note the discrepancies between base, lora, and qlora runs is due to how generate.py was programmed. Base wasn't running generation. With batches, hence why it took so long. I forgot to set the CUDA_VISIBLE_DEVICES env variable when I generated with lora and that sharded the model onto two GPU's. qlora I learned from my mistakes and only exposed one gpu, therefore less overhead time and quicker generation

Start time 11:40 pm