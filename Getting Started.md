## Getting Started

To begin exploring these architectures, ensure your environment is configured for high-performance JAX execution and Hugging Face model access.

1. Hugging Face Access: Many of the SOTA models included require explicit permission from the model authors or are gated. Hugging Face Token: Create a "Read" access token at huggingface.co/settings/tokens. CLI Login: Run huggingface-cli login in your terminal and paste your token to allow the notebooks to download weights directly. Gated Models: Ensure you have visited the specific model cards on Hugging Face and accepted the terms of use before attempting to download.

2. Compute & GPU Environment: While smaller models like ResNet or ViT can run on consumer-grade hardware, the larger models and FSDP training logic require significant VRAM. This project was developed and validated using NVIDIA 3090 GPUs. The notebooks demonstrate how to leverage JAX sharding and FSDP to fit large models within the VRAM constraints of the 3090.

3. Installation: Ensure you have the latest JAX version (13) configured for your specific CUDA version to fully leverage XLA acceleration.

## The Anatomy of a Notebook: From Weights to Architecture

Each notebook in this repository follows a consistent, transparent pipeline designed to demystify the transition from a "Black Box" Model to a "White Box" JAX implementation.

- **Ingestion & Inspection:** We begin by pulling the raw weights from Hugging Face. Before writing a single line of JAX, we inspect the original state dictionary to understand the naming conventions and tensor shapes.

- **Config-Driven Construction:** Using a lightweight configuration object, we define the hyperparameters. This ensures that our JAX architecture is a precise mirror of the original SOTA model.

- **Modular, Cell-Based Logic:** To avoid "spaghetti code," the architecture is segmented into logical blocks. Each concept lives in its own cell, allowing you to run, inspect, and verify the output of individual components in isolation.

- **Functional Forward Pass:** Following the JAX philosophy, we implement a purely functional forward pass. There are no hidden states or global variables, just data flowing through a series of mathematical transformations.

- **Weight Mapping & Parity:** We manually map the loaded weights into our JAX structures. This is the "Aha!" moment where you see exactly how a PyTorch tensor corresponds to a functional JAX parameter.

## Structure

Each model is implemented in a dedicated notebook:

```
/models
    /resnet
    /vit
    /mamba
    /qwen
    /gemma
    ...
```

Each notebook typically includes:

- Model architecture implementation
- Forward pass logic
- Weight loading (from Hugging Face where applicable)
- Minimal validation / test runs
