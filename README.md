# alzheimer-mri-vit-classification

# Alzheimer's MRI Staging: Hybrid CNN-Transformer Architecture

## Executive Summary
This repository contains a deep learning pipeline designed to classify the progression stages of Alzheimer's Disease from Magnetic Resonance Imaging (MRI) scans. Moving beyond standard transfer learning, this project implements a custom Hybrid Architecture that couples a Convolutional Neural Network (CNN) backbone with a Vision Transformer (ViT) head. 

The objective is to capture both fine-grained local textures (cortical atrophy, ventricular enlargement) via convolutions, and global spatial dependencies across the brain volume via self-attention mechanisms.

## Architecture Design
The core engine (`models/hybrid_vit.py`) replaces the traditional Global Average Pooling layer of a standard CNN with a custom-built Vision Transformer.

* **Feature Extractor (Backbone):** MobileNetV2 (pre-trained on ImageNet). Selected for its parameter efficiency and depthwise separable convolutions, acting as a high-level feature map generator.
* **Self-Attention Head (Custom ViT):** Built from scratch using TensorFlow/Keras subclassing. The CNN feature maps are flattened into "patches" and passed through a projection layer. Learnable class tokens and positional embeddings are injected before routing the sequence through multiple Multi-Head Attention blocks.
* **Precision & Hardware Optimization:** The entire computational graph operates under a `mixed_float16` global policy. This prevents VRAM bottlenecking on consumer-grade GPUs (e.g., 16GB VRAM) and accelerates tensor operations during the Multi-Head Attention dot products.

## Data Engineering & Pipeline
The dataset utilized is the Alzheimer's MRI 4-Classes Dataset (Mild, Moderate, Non, VeryMild), automatically ingested via the Kaggle API.

* **Ingestion:** Automated dataset resolution and mapping to ensure environment reproducibility.
* **tf.data Optimization:** The data pipeline is built for high throughput. It utilizes `tf.data.Dataset` mapping, dynamic caching, and prefetching (`tf.data.AUTOTUNE`) to ensure the GPU is never starved for batches by the CPU.
* **Stratification:** Strict stratified splitting guarantees balanced class distributions across training and validation sets, preventing majority-class collapse during the fine-tuning phase.

## Experimental Framework (Benchmarking)
To validate the architectural decisions, the pipeline includes an automated benchmarking script (`benchmark_runner.py`) that executes and evaluates four distinct topological configurations in a single run:

1.  **Baseline Hybrid:** CNN Backbone + Default ViT Head (256 embed dim, 4 heads, 2 layers).
2.  **Augmented Hybrid:** Baseline + Spatial Data Augmentation (Rotation, Zoom, Flip).
3.  **Heavier ViT:** CNN Backbone + Deep ViT Head (256 embed dim, 8 heads, 1024 MLP dim, 4 layers) to test parameter scaling vs. overfitting limits.
4.  **Standard CNN:** CNN Backbone + Dense layers (No Transformer mechanics) to serve as the control group.

The framework automatically tracks inference times, parameter counts, and outputs comparative Confusion Matrices and Accuracy/Loss curves.

## Local Execution

1. Clone the repository and install the strict dependencies:
```bash
git clone [https://github.com/yourusername/alzheimer-hybrid-vit.git](https://github.com/yourusername/alzheimer-hybrid-vit.git)
cd alzheimer-hybrid-vit
pip install -r requirements.txt
```

2. Run the benchmarking suite (this will automatically download the Kaggle dataset, run all 4 scenarios, and generate the evaluation artifacts):
```bash
python benchmark_runner.py
```

Note: Ensure you have a CUDA-capable GPU configured. The script is heavily optimized for parallel execution and mixed precision.

