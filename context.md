# Context: GPT-From-Scratch

This file stores context, notes, and the current state of the `gpt-from-scratch` project. It acts as a memory bank so that any future AI assistant can quickly understand the project architecture and resume work without needing to read the entire conversation history.

## Project State & Progress
We have completed Phases 1 through 5 of our roadmap. The project is a fully functional, Decoder-only Transformer language model built entirely from scratch in PyTorch.

- **Phase 1 (Data Prep)**: We are using the `tinyshakespeare.txt` dataset. `src/tokenizer.py` implements a character-level tokenizer (vocab size 65). `src/dataset.py` handles batching data for the GPU/CPU.
- **Phase 2 (Architecture)**: `src/model.py` contains the core Transformer math (`Head`, `MultiHeadAttention`, `FeedForward`, `Block`, and `GPTLanguageModel`).
- **Phase 3 (Training)**: `train.py` is located in the root directory. It automatically detects Apple MPS or CUDA, runs the training loop, periodically saves checkpoints to `checkpoints/gpt_model.pt`, and plots the loss using `matplotlib` to `outputs/loss_plot.png`. **Crucially, `train.py` auto-resumes from existing checkpoints if they are present.**
- **Phase 4 (Generation)**: `generate.py` is located in the root directory. It loads the saved checkpoint and generates 500 characters of text based on a hardcoded prompt (e.g., `"ROMEO:\n"`).
- **Phase 5 (Scaling)**: Hyperparameters have been scaled up for better performance.

## Current Hyperparameters (`src/config.py`)
- **batch_size**: 64
- **block_size** (Context Window): 256
- **n_embd** (Embedding Dim): 384
- **n_head** (Attention Heads): 6
- **n_layer** (Transformer Blocks): 6
- **max_iters**: 5000
- **learning_rate**: 3e-4

## How to use the project
1. **To Train**: Run `python train.py`. It will load `checkpoints/gpt_model.pt` if it exists and resume training, saving every 500 steps.
2. **To Generate**: Run `python generate.py`. It uses the trained model to generate Shakespearean text.

## Future / Next Steps (Ideas)
- Switch from a character-level tokenizer to a subword tokenizer like `tiktoken` (BPE).
- Implement Flash Attention for significantly faster training speeds.
- Train the model on a much larger dataset (e.g., OpenWebText) to see emergent capabilities.
