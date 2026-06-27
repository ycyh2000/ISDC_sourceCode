# ISDC_sourceCode
# ISDC: Index-Sorted Differential Compression for N:M Sparse Large Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of **ISDC (Index-Sorted Differential Compression)**, a lossless compression framework for efficient storage and deployment of N:M sparse large language models.

---

## Overview

Large language models (LLMs) contain billions of floating-point parameters, resulting in substantial storage and memory overhead. Existing lossless compression methods often struggle to effectively compress floating-point mantissa streams due to their inherently high entropy.

ISDC addresses this challenge by leveraging the structural characteristics of N:M sparsity. Specifically, ISDC utilizes sparse index information to reorder non-zero weights, transforming high-entropy mantissa values into low-entropy differential representations that are more amenable to compression.

The framework further introduces a GPU-oriented decompression pipeline that enables efficient parallel reconstruction of compressed sparse weights.

### Key Features

- Lossless compression for N:M sparse neural network weights
- Index-guided sorting strategy
- Differential mantissa encoding
- Shared bit-width representation
- GPU-parallel decompression kernels
- Exact reconstruction with zero accuracy degradation

---

## Framework
## Framework

ISDC consists of a compression stage and a GPU-oriented decompression stage.

### Compression

<p align="center">
<img src="figures/ISDC_compression.png" width="800">
</p>

The compression stage exploits sparsity-induced index information to transform high-entropy mantissa streams into compact differential representations. Specifically, ISDC:

1. Extracts non-zero values from N:M sparse matrices.
2. Reorders values according to sparse index information.
3. Applies differential encoding within each sparse group.
4. Compresses residuals using a shared bit-width representation.

### Decompression

<p align="center">
<img src="figures/ISDC_decompression.png" width="800">
</p>

The decompression stage leverages massively parallel GPU kernels to reconstruct sparse weights directly from compressed bitstreams, enabling efficient sparse model deployment while preserving exact numerical accuracy.


---

## Repository Structure

```text
ISDC/
│
├── compressor/
│   ├── compress.py
│   ├── decompress.py
│   ├── differential_encode.py
│   ├── differential_decode.py
│   └── bitpack.py
│
├── cuda/
│   ├── isdc_kernel.cu
│   ├── ring_buffer.cu
│   ├── prefix_sum.cu
│   └── CMakeLists.txt
│
├── benchmark/
│   ├── compression_ratio.py
│   ├── decompression_speed.py
│   ├── entropy_analysis.py
│   └── ablation_sorted_vs_unsorted.py
│
├── examples/
│   ├── opt125m.py
│   ├── opt350m.py
│   └── opt1.3b.py

├── figures/
│
├── data/
│
├── requirements.txt
├── LICENSE
└── README.md
```


## Installation

### 1. Create a Conda Environment

```bash
conda create -n ISDC python=3.11
conda activate ISDC


### 2. Install PyTorch

For CUDA 11.8:

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch; print(torch.__version__)"
```

Expected output:

```text
2.7.1
```

---

## System Requirements

* Ubuntu 22.04 / 24.04
* Python >= 3.10
* CUDA >= 11.8

The experiments in the paper were conducted on:

* CPU: Intel Xeon Platinum 8380
* GPU: NVIDIA RTX A6000
* CUDA: 12.0
* PyTorch: 2.7.1

```
```
