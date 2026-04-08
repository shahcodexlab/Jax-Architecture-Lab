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

### Getting Started

For setup instructions, see the [Getting Started Guide](getting-started.md).

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
- Tiny Recursion Models (TRM)

---

### Speech

- Whisper
- Mimi (neural audio codec)

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
