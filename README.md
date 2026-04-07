# JAX Model Garden Notebooks (From Scratch)

This repository serves as a "Rosetta Stone" for modern deep learning architectures. It contains **from-scratch implementations of modern deep learning architectures using JAX**, with a focus on **understanding model internals, modular design, and architectural patterns** across them.

Rather than relying on high-level frameworks, each model is implemented using **core JAX primitives**, with weights optionally loaded from Hugging Face for validation and experimentation.

### Why JAX?

JAX allows for a near 1:1 mapping between mathematical notation and code. Unlike the imperative nature of PyTorch, JAX's functional paradigm forces a deeper understanding of the math and memory.

### Why Notebooks?

Notebooks allow the reader to see the math (LaTeX), the code (JAX), and the output (Tensors/Visuals) in a single vertical flow. This format is ideal for educational deep dives where seeing the intermediate shapes of a model is more valuable than a hidden script.

---

## Motivation

Most practitioners interact with models through high-level APIs.
This project was built to go deeper to:

- Understand how modern architectures actually work under the hood
- Reconstruct models across multiple modalities including vision, language (autoregressive and state space) models.
- Identify **common design patterns** across seemingly different architectures
- Build intuition for **scaling, efficiency, and modularity**

---

### The Anatomy of a Notebook: From Weights to Architecture

Each notebook in this repository follows a consistent, transparent pipeline designed to demystify the transition from a "Black Box" Model to a "White Box" JAX implementation.

- **Ingestion & Inspection:** We begin by pulling the raw weights from Hugging Face. Before writing a single line of JAX, we inspect the original state dictionary to understand the naming conventions and tensor shapes.

- **Config-Driven Construction:** Using a lightweight configuration object, we define the hyperparameters. This ensures that our JAX architecture is a precise mirror of the original SOTA model.

- **Modular, Cell-Based Logic:** To avoid "spaghetti code," the architecture is segmented into logical blocks. Each concept lives in its own cell, allowing you to run, inspect, and verify the output of individual components in isolation.

- **Functional Forward Pass:** Following the JAX philosophy, we implement a purely functional forward pass. There are no hidden states or global variables, just data flowing through a series of mathematical transformations.

- **Weight Mapping & Parity:** We manually map the loaded weights into our JAX structures. This is the "Aha!" moment where you see exactly how a PyTorch tensor corresponds to a functional JAX parameter.

### A Playground for Architectural Innovation

- The primary goal of this repository is to lower the barrier to entry for low-level deep learning experimentation. By separating the Architecture from the Training Logic, we give you the freedom to play without the overhead of a massive framework.

- Mix, Match, and Mutate: Want to see what happens if you swap a standard Attention block in Gemma with a Mamba 2 selective scan? Because the code is modular and functional, you can "hot-swap" components between notebooks with minimal friction.

- Plug-and-Play Training: While the core notebooks focus on the forward pass and architecture, we provide dedicated FSDP (Distributed Training) and LoRA (Fine-tuning) notebooks. If you are interested in training you can refer to the notebooks on how to extend your architecture to support training.

This project is an attempt or an invitation to understand the machinery of the models shaping the world as we move forward.

---

### Getting Started

To begin exploring these architectures, ensure your environment is configured for high-performance JAX execution and Hugging Face model access.

1. Hugging Face Access: Many of the SOTA models included require explicit permission from the model authors or are gated. Hugging Face Token: Create a "Read" access token at huggingface.co/settings/tokens. CLI Login: Run huggingface-cli login in your terminal and paste your token to allow the notebooks to download weights directly. Gated Models: Ensure you have visited the specific model cards on Hugging Face and accepted the terms of use before attempting to download.

2. Compute & GPU Environment: While smaller models like ResNet or ViT can run on consumer-grade hardware, the larger models and FSDP training logic require significant VRAM. This project was developed and validated using NVIDIA 3090 GPUs. The notebooks demonstrate how to leverage JAX sharding and FSDP to fit large models within the VRAM constraints of the 3090.

3. Installation: Ensure you have the latest JAX version (13) configured for your specific CUDA version to fully leverage XLA acceleration.

---

## Model Coverage

### Vision Models

- ResNet
- VGG
- EfficientNet
- ConvNeXt (v2)
- Vision Transformer (ViT)
- DINO (self-supervised learning)
- Segment Anything Model (SAM)

---

### Language Models / Transformers

- Gemma
- Qwen (30B architecture)
- UMT5

---

### Speech

- Whisper

---

### Generative Models

- Variational Autoencoder (VAE)
- UNet

---

### State Space Models

- Mamba (selective state space model)

---

### Diffusion Models

- Llada

---

### Training & Fine Tuning

- FSDP (Fully Sharded Data Parallel)
- LoRA (Low-Rank Adaptation for parameter-efficient fine-tuning)

---

## Key Features

- **Pure JAX implementations** (no PyTorch model wrappers)
- Modular, reusable building blocks
- Compatibility with **Hugging Face pretrained weights**
- Notebook-based explorations for each model

---

## Architectural Focus

Across implementations, the repository explores:

- Transformer architectures and attention mechanisms
- Convolutional design patterns (ResNet → ConvNeXt evolution)
- Representation learning (e.g., DINO)
- State space models vs attention-based models (Mamba vs Transformers)
- Generative modeling primitives (VAE, U-Net)
- Parameter-efficient training (LoRA)
- Distributed training concepts (FSDP)

---

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

---

## Future Work

- Implement additional architectures, innovations and advancements in deep learning

---

## Why This Matters

This project reflects an effort to move beyond using models to **understanding and reconstructing them**.

It is particularly relevant for:

- People interested in systems-level understanding
- People interested in exploring architectural design
- Practitioners working on **LLMs, generative AI, and scalable ML systems**

---

## Contributions

This is primarily a learning-driven project, but we welcome contributions! If you're interested in adding new models, improving existing implementations, or enhancing documentation

---

## Contact

If you’d like to discuss this work or collaborate, feel free to reach out.

---

## ⭐ If you find this useful, consider starring the repo!
