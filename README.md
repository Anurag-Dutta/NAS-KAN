# NAS-KAN: Neural Architecture Search for Kolmogorov-Arnold Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> **Optimal Neural Architecture Search for Kolmogorov-Arnold Network-Based Image Classification**  


---

## 📋 Abstract

Kolmogorov-Arnold Networks (KAN) can achieve superior accuracy over Multi-Layer Perceptrons (MLP) in vision tasks. However, their extensive hyperparameter space (grid size, spline order, base activation, etc.) makes manual architecture selection difficult. 

This repository presents:
- **A Neural Architecture Search (NAS) benchmark dataset** of over **150,000 candidate Conv-KAN architectures**
- Combines convolutional feature extractors with terminal KAN layers
- Evaluated on **five benchmark vision datasets**: MNIST, CIFAR-10, CIFAR-100, STL-10, and SVHN
- Unlike existing NAS benchmarks that sample discrete architectures, our dataset **exhaustively covers all valid configurations** through principled constraints
- **Fine-tuned Mistral-7B-Instruct** with low-rank adapters (LoRA) that inverts the typical NAS structure: given hardware constraints, it directly generates best-candidate architectures
- Achieves **0.5529% better accuracy on average** with fewer floating-point operations than architectures selected using existing NAS strategies

---

## 🎯 Motivation

### Why KANs for Vision?

<p align="center">
  <img src="FIGS/kan_vs_mlp_motivation.png" alt="KAN vs MLP Performance Comparison" width="800"/>
</p>

**Key Observations:**
- **KANs achieve higher classification accuracy (mAP)** across all benchmark datasets
- **Trade-off**: KANs require orders of magnitude more parameters than MLPs
- **Challenge**: Complex hyperparameter space (grid size, spline order, base activation) makes manual design difficult
- **Solution**: Automated Neural Architecture Search specifically designed for KAN-based vision models

---

## 🎯 Key Contributions

### 1. **Exhaustive NAS Benchmark for Conv-KAN Architectures**
- **150,000+ evaluated architectures** across 5 vision datasets
- Each entry stores:
  - **Task metrics**: test accuracy, precision, recall, F1-score
  - **Resource metrics**: epoch time, parameter count, FLOPs
  - **Training dynamics**: learning curves for predictor-based methods
  - **Human-readable strings**: e.g., `Conv2d(k=3,p=1) -> ... -> KAN(order=3,grid=8,act=ReLU)`

### 2. **Constraint-Based Search Space Reduction**
- Reduces unconstrained search space (**5000× reduction**)
- **Constraint I (Spatial Validity)**: Ensures output dimensions remain valid (≥1) after convolution + pooling
- **Constraint II (Receptive Field Ordering)**: Enforces non-decreasing kernel sequences $K_1 \leq K_2 \leq \ldots \leq K_d$ for hierarchical feature extraction

### 3. **LM-Based Inverse Architecture Generation**
- Fine-tuned **Mistral-7B-Instruct** with LoRA in 4-bit mode
- **Input**: Hardware constraints (params, FLOPs, time) + accuracy-efficiency preference (α ∈ [0, 1])
- **Output**: Valid architecture string directly queryable from benchmark
- Deployed via **Ollama** for efficient inference

### 4. **Superior Performance**
- **Average accuracy improvement**: +0.5529% across all datasets
- **Notable improvements**: +1.2261% on CIFAR-10, +2.2680% on STL-10
- **2.6% reduction** in parameter count compared to baselines

