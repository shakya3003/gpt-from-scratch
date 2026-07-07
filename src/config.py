from dataclasses import dataclass

@dataclass
class GPTConfig:
    # Hyperparameters for TinyShakespeare (Tuned for better performance)
    batch_size: int = 64      # Number of independent sequences to process in parallel
    block_size: int = 256     # Maximum context length for predictions
    max_iters: int = 5000     # Total training iterations
    eval_interval: int = 500  # How often to evaluate the loss
    learning_rate: float = 3e-4
    device: str = "cpu"       # We will set this to 'mps' or 'cuda' in the training script if available
    eval_iters: int = 200     # Number of iterations to run during evaluation
    
    # Model Architecture Hyperparameters (Scaled up)
    n_embd: int = 384         # Embedding dimension
    n_head: int = 6           # Number of attention heads
    n_layer: int = 6          # Number of transformer blocks
    dropout: float = 0.2      # Dropout rate for regularization
    
    # Vocabulary size will be determined by the tokenizer
    vocab_size: int = 0
