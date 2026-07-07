import os
import torch
import matplotlib.pyplot as plt
from src.config import GPTConfig
from src.tokenizer import CharacterTokenizer
from src.dataset import get_batch
from src.model import GPTLanguageModel

def main():
    # 1. Setup configuration
    config = GPTConfig()
    # Check if Metal Performance Shaders (MPS) is available on Mac, otherwise use CPU
    if torch.backends.mps.is_available():
        config.device = 'mps'
    elif torch.cuda.is_available():
        config.device = 'cuda'
    else:
        config.device = 'cpu'
    
    print(f"Using device: {config.device}")

    # 2. Load Dataset and Tokenizer
    data_path = os.path.join("data", "tinyshakespeare.txt")
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Please run src/download_data.py first.")
        return

    tokenizer = CharacterTokenizer(data_path)
    config.vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {config.vocab_size}")

    # Encode entire text dataset into integers and split into Train / Val
    data = torch.tensor(tokenizer.encode(tokenizer.text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    print(f"Train data size: {len(train_data)} tokens, Val data size: {len(val_data)} tokens")

    # 3. Initialize Model and Optimizer
    model = GPTLanguageModel(config)
    checkpoint_path = os.path.join("checkpoints", "gpt_model.pt")
    
    # Try to load existing checkpoint to resume training
    if os.path.exists(checkpoint_path):
        print(f"Loading existing model weights from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
    else:
        print("No existing checkpoint found. Starting from scratch...")

    model = model.to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # 4. Training Loop
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval() # Set model to evaluation mode
        for split_name, split_data in [('train', train_data), ('val', val_data)]:
            losses = torch.zeros(config.eval_iters)
            for k in range(config.eval_iters):
                X, Y = get_batch(split_data, config.block_size, config.batch_size, config.device)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split_name] = losses.mean()
        model.train() # Set model back to training mode
        return out

    # Lists to store metrics for plotting
    tracked_iters = []
    tracked_train_losses = []
    tracked_val_losses = []

    print("Starting training...")
    for iter in range(config.max_iters):
        
        # Every once in a while evaluate the loss on train and val sets
        if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            # Record for plotting
            tracked_iters.append(iter)
            tracked_train_losses.append(losses['train'])
            tracked_val_losses.append(losses['val'])

            # Periodically save the model so we don't lose progress if interrupted
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)

        # sample a batch of data
        xb, yb = get_batch(train_data, config.block_size, config.batch_size, config.device)

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("Training finished!")
    
    # 5. Plot and save the loss graph
    plt.figure(figsize=(10, 6))
    plt.plot(tracked_iters, tracked_train_losses, label="Train Loss")
    plt.plot(tracked_iters, tracked_val_losses, label="Validation Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Time")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join("outputs", "loss_plot.png")
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(plot_path)
    print(f"Loss plot saved to {plot_path}")
    print(f"Final Model saved to {checkpoint_path}")

if __name__ == "__main__":
    main()
