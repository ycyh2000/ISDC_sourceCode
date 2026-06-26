# SparseGPT – Reference Implementation

This repository contains a reference implementation of the paper  
**[SparseGPT: Massive Language Models Can be Accurately Pruned in One-shot](https://arxiv.org/abs/2301.00774)**.

SparseGPT enables **one‑shot pruning** of large GPT‑style models (e.g., OPT, BLOOM) to high sparsity (≥50%) without retraining. It supports unstructured, semi‑structured (2:4, 4:8), and combined sparse‑quantized compression.

The code is built upon the [official SparseGPT codebase](https://github.com/IST-DASLab/sparsegpt).

---

## Dependencies

Tested with:

- `torch` ≥ 1.10.1  
- `transformers` ≥ 4.21.2  
- `datasets` ≥ 1.17.0  

Install all required packages via `pip install -r requirements.txt` (you may create this file from your environment).

---

## Quick Start

### Prune an OPT model with 4:8 semi‑structured sparsity

```bash
# OPT‑125M
python opt_bfloat16.py ./facebook__opt-125m c4 --save ./result/opt-125m  --prunen 4 --prunem 8
# OPT‑350M
python opt_bfloat16.py ./facebook__opt-350m c4 --save ./result/opt-350m  --prunen 4 --prunem 8
# OPT‑1.3B
python opt_bfloat16.py facebook/opt-1.3b c4 --save ./result/opt-1.3b  --prunen 4 --prunem 8
```

