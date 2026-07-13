import os
import torch
import matplotlib.pyplot as plt
from src.config import GPTConfig
from src.tokenizer import BPETokenizer
from src.dataset import get_batch
from src.model import GPTLanguageModel

def train_model(use_rope: bool, name: str, data_path: str, max_iters: int = 3000):
    print(f"\n{'='*50}\nStarting Training: {name} (use_rope={use_rope})\n{'='*50}")
    
    # 1. Setup config
    config = GPTConfig()
    config.use_rope = use_rope
    config.max_iters = max_iters
    
    if torch.backends.mps.is_available():
        config.device = 'mps'
    elif torch.cuda.is_available():
        config.device = 'cuda'
    else:
        config.device = 'cpu'
        
    print(f"Using device: {config.device}")

    # 2. Setup Tokenizer and Data
    tokenizer = BPETokenizer(data_path, vocab_size=5000)
    config.vocab_size = tokenizer.vocab_size
    
    data = torch.tensor(tokenizer.encode(tokenizer.text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    # Set seed for reproducible ablation comparison
    # Important to set right before training loop so both models see identical batches
    torch.manual_seed(1337)
    
    # 3. Model & Optimizer
    model = GPTLanguageModel(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    use_amp = (config.device == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    device_type = 'cuda' if 'cuda' in config.device else 'cpu'

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split_name, split_data in [('train', train_data), ('val', val_data)]:
            losses = torch.zeros(config.eval_iters)
            for k in range(config.eval_iters):
                X, Y = get_batch(split_data, config.block_size, config.batch_size, config.device)
                with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split_name] = losses.mean()
        model.train()
        return out

    tracked_iters = []
    tracked_train_losses = []
    tracked_val_losses = []

    for iter in range(config.max_iters):
        if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
            losses = estimate_loss()
            train_ppl = torch.exp(torch.tensor(losses['train'])).item()
            val_ppl = torch.exp(torch.tensor(losses['val'])).item()
            print(f"[{name}] step {iter}: train loss {losses['train']:.4f} (ppl: {train_ppl:.2f}), val loss {losses['val']:.4f} (ppl: {val_ppl:.2f})")
            
            tracked_iters.append(iter)
            tracked_train_losses.append(losses['train'].item())
            tracked_val_losses.append(losses['val'].item())

        xb, yb = get_batch(train_data, config.block_size, config.batch_size, config.device)
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
            logits, loss = model(xb, yb)
            
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return tracked_iters, tracked_train_losses, tracked_val_losses

def main():
    data_path = os.path.join("data", "tinyshakespeare.txt")
    if not os.path.exists(data_path):
        print("Data not found. Please run src/download_data.py first.")
        return
        
    # Run Baseline
    baseline_iters, baseline_train, baseline_val = train_model(use_rope=False, name="Baseline", data_path=data_path, max_iters=3000)
    
    # Run RoPE
    rope_iters, rope_train, rope_val = train_model(use_rope=True, name="RoPE", data_path=data_path, max_iters=3000)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(baseline_iters, baseline_train, label="Baseline (Learned PosEmb) - Train", linestyle='-', color='blue')
    plt.plot(baseline_iters, baseline_val, label="Baseline (Learned PosEmb) - Val", linestyle='--', color='blue')
    
    plt.plot(rope_iters, rope_train, label="RoPE - Train", linestyle='-', color='red')
    plt.plot(rope_iters, rope_val, label="RoPE - Val", linestyle='--', color='red')
    
    plt.xlabel("Iterations")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Ablation Study: Baseline vs Rotary Positional Embeddings (RoPE)")
    plt.legend()
    plt.grid(True)
    
    os.makedirs("outputs", exist_ok=True)
    plot_path = os.path.join("outputs", "rope_ablation_plot.png")
    plt.savefig(plot_path)
    print(f"\nAblation study complete! Combined plot saved to {plot_path}")

if __name__ == "__main__":
    main()
