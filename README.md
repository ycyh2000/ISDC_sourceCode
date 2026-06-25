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
<p align="center">
<img src="figures/ISDC_compression.png" width="800">
</p>
<p align="center">
<img src="figures/ISDC_decompression.png" width="800">
</p>

The overall workflow consists of:

1. Extract non-zero weights from N:M sparse matrices.
2. Reorder values according to sparse index information.
3. Apply differential encoding within each sparse group.
4. Encode residuals using shared bit-width representation.
5. Reconstruct sparse weights using massively parallel GPU decompression.

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
│   ├── bloom.py
│   └── llama.py
│
├── figures/
│
├── data/
│
├── requirements.txt
├── LICENSE
└── README.md