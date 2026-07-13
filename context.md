# Context: GPT-From-Scratch

This file stores context, notes, and the current state of the `gpt-from-scratch` project. It acts as a memory bank so that any future AI assistant can quickly understand the project architecture, mechanics, codebase organization, and roadmap without needing to read the entire conversation history.

## Project Overview
This project is a fully functional **Decoder-only Transformer language model**, built completely from scratch natively in PyTorch. It aims to implement the core mathematical and structural concepts powering modern Large Language Models (like GPT) without relying on high-level wrapper libraries. The model comprises **~10.7 million parameters**.

## Codebase Organization & Architecture
The project is organized to separate concerns effectively into data prep, core model math, and execution scripts:

### Source Code (`src/`)
*   **`tokenizer.py`**: Implements a foundational character-level tokenizer trained specifically on the `tinyshakespeare.txt` dataset. The vocabulary size is 65.
*   **`dataset.py`**: Defines the PyTorch Dataset class. It handles data loading logic and correctly batches context windows (input strings) to matching targets for the GPU/CPU.
*   **`model.py`**: The heart of the project containing the PyTorch modules:
    *   `GPTLanguageModel`: The primary model combining embeddings, blocks, and the linear head.
    *   `Block`: Custom Transformer blocks (x6) applying Layer Normalization, Multi-Head Attention, and FeedForward networks with skip/residual connections.
    *   `MultiHeadAttention`: Implements multi-head self-attention. Crucially, it has been optimized to use **Flash Attention** (via PyTorch’s native `scaled_dot_product_attention`), heavily reducing memory overhead and providing massive speedups during training.
    *   `Head`: Single scaled dot-product attention head.
    *   `FeedForward`: A simple multilayer perceptron used inside the Transformer blocks.
*   **`config.py`**: Centralized configuration and hyperparameters. 
    *   **Context Window (`block_size`)**: 256
    *   **Batch Size**: 64
    *   **Embedding Dimension (`n_embd`)**: 384
    *   **Attention Heads (`n_head`)**: 6
    *   **Transformer Layers (`n_layer`)**: 6
    *   **Learning Rate**: 3e-4
*   **`download_data.py`**: Script to fetch datasets.

### Pipeline Execution Scripts (Root)
*   **`train.py`**: A robust training loop equipped to automatically detect hardware (Apple MPS or CUDA). 
    *   It manages training optimization using **Mixed-Precision Training** (`torch.amp`). 
    *   It inherently saves progress every 500 steps and **auto-resumes** intelligently from existing checkpoints (`checkpoints/gpt_model.pt`). 
    *   It tracks and visualizes training and validation loss using `matplotlib`, outputting to `outputs/loss_plot.png`.
*   **`generate.py`**: An autoregressive inference script used to generate text from a saved checkpoint. It supports tunable sampling techniques like temperature scaling (`--temperature`) and top-k sampling (`--top_k`).

## Current State & Performance
The project architecture has been fully upgraded to support modern primitives. We are currently running an ablation study (`eval_suite.py`) to benchmark the architectural improvements:

| Config | Tokenizer | Pos. Embedding | Val Perplexity | Params |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | char-level | learned | ~1.48 (at 5k steps) | ~10.7M |
| **+BPE** | BPE (5000 vocab) | learned | *TBD (Training)* | ~7.34M |
| **+BPE+RoPE** | BPE (5000 vocab) | RoPE | *TBD (Training)* | ~7.27M |

## How to use the project
1.  **To Train**: Run `python train.py`. It will load `checkpoints/gpt_model.pt` if it exists and resume training, saving every 500 steps.
2.  **To Generate**: Run `python generate.py`. It uses the trained model to generate Shakespearean text with tunable constraints.

## Roadmap & Future Steps
-   **Tokenizer Upgrade**: Transitioned from a basic character-level tokenizer to an advanced subword tokenizer (BPE). *(Completed)*
-   **Data Scaling**: Train the model on a moderately larger, more diverse dataset (e.g., a few hundred MBs of text) to evaluate generalization and reduce overfitting, measured via validation perplexity.
