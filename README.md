# GPT From Scratch

A fully functional, Decoder-only Transformer language model built entirely from scratch in PyTorch. This project is a foundational implementation designed for learning and experimenting with the core concepts behind models like GPT.

## Features

- **Built from Scratch**: Core components like Multi-Head Attention, FeedForward networks, and Transformer Blocks are implemented natively in PyTorch.
- **Character-Level Tokenization**: Currently uses a character-level tokenizer trained on the `tinyshakespeare.txt` dataset.
- **Hardware Agnostic**: Automatically detects and uses Apple MPS (Metal Performance Shaders) or CUDA for GPU acceleration if available, falling back to CPU otherwise.
- **Resumable Training**: The training script automatically detects existing checkpoints and resumes training seamlessly.
- **Text Generation**: Includes a dedicated script for autoregressive text generation using the trained model.

## Model Architecture & Hyperparameters

The model is a standard Decoder-only Transformer. The current configuration (`src/config.py`) is set as follows:

- **Batch Size**: 64
- **Context Window (block_size)**: 256
- **Embedding Dimension (n_embd)**: 384
- **Attention Heads (n_head)**: 6
- **Transformer Blocks (n_layer)**: 6
- **Max Iterations**: 5000
- **Learning Rate**: 3e-4

## Getting Started

### Prerequisites
Make sure you have Python installed along with the required dependencies. You can install the dependencies using:
```bash
pip install -r requirements.txt
```

### Training the Model
To start training the model, simply run:
```bash
python train.py
```
This will:
- Load the dataset and tokenizer.
- Initialize the model or load an existing checkpoint from `checkpoints/gpt_model.pt`.
- Run the training loop, saving checkpoints every 500 steps.
- Plot and save the training loss to `outputs/loss_plot.png`.

### Generating Text
Once you have a trained checkpoint, you can generate text by running:
```bash
python generate.py
```
This script loads the latest checkpoint and generates characters sequentially based on a hardcoded prompt (e.g., `"ROMEO:\n"`).

## Project Structure

- `src/`
  - `tokenizer.py`: Character-level tokenizer implementation.
  - `dataset.py`: Data loading and batching logic.
  - `model.py`: Core Transformer architecture (Head, MultiHeadAttention, FeedForward, Block, GPTLanguageModel).
  - `config.py`: Hyperparameters and configuration.
- `train.py`: Main training loop with auto-resume functionality.
- `generate.py`: Inference script for generating text.
- `checkpoints/`: Directory for saving model weights (`gpt_model.pt`).
- `outputs/`: Directory for saving training artifacts (e.g., loss plots).

## Roadmap & Future Steps

- [ ] Switch from a character-level tokenizer to a subword tokenizer like `tiktoken` (BPE).
- [ ] Implement Flash Attention for significantly faster training speeds.
- [ ] Train the model on a much larger dataset (e.g., OpenWebText) to observe emergent capabilities.
- [ ] Implement mixed-precision training.
