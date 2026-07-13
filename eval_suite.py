import os
import torch
import matplotlib.pyplot as plt
from src.config import GPTConfig
from src.tokenizer import BPETokenizer, CharacterTokenizer
from src.dataset import get_batch
from src.model import GPTLanguageModel

def train_model(name: str, config: GPTConfig, tokenizer, data_path: str, save_path: str):
    print(f"\n{'='*50}\nStarting Training: {name}\n{'='*50}", flush=True)
    
    if torch.backends.mps.is_available():
        config.device = 'mps'
    elif torch.cuda.is_available():
        config.device = 'cuda'
    else:
        config.device = 'cpu'
        
    print(f"Using device: {config.device}", flush=True)

    data = torch.tensor(tokenizer.encode(tokenizer.text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    torch.manual_seed(1337)
    
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
    tracked_train_ppl = []
    tracked_val_ppl = []

    for iter in range(config.max_iters):
        if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
            losses = estimate_loss()
            train_ppl = torch.exp(torch.tensor(losses['train'])).item()
            val_ppl = torch.exp(torch.tensor(losses['val'])).item()
            print(f"[{name}] step {iter}: train loss {losses['train']:.4f} (ppl: {train_ppl:.2f}), val loss {losses['val']:.4f} (ppl: {val_ppl:.2f})", flush=True)
            
            tracked_iters.append(iter)
            tracked_train_ppl.append(train_ppl)
            tracked_val_ppl.append(val_ppl)

        xb, yb = get_batch(train_data, config.block_size, config.batch_size, config.device)
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
            logits, loss = model(xb, yb)
            
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    
    return tracked_iters, tracked_train_ppl, tracked_val_ppl

def generate_text(model, tokenizer, prompt: str, max_tokens: int = 150, temperature: float = 0.8):
    model.eval()
    context_list = tokenizer.encode(prompt)
    context = torch.tensor([context_list], dtype=torch.long, device=model.config.device)
    
    generated_tokens = model.generate(
        context, 
        max_new_tokens=max_tokens, 
        temperature=temperature, 
        top_k=None
    )[0].tolist()
    
    return tokenizer.decode(generated_tokens)

def main():
    data_path = os.path.join("data", "tinyshakespeare.txt")
    if not os.path.exists(data_path):
        print("Data not found. Please run src/download_data.py first.")
        return
        
    char_tokenizer = CharacterTokenizer(data_path)
    bpe_tokenizer = BPETokenizer(data_path, vocab_size=5000)
    
    max_iters = 50
    
    # Run 1: Char-Level Baseline
    char_config = GPTConfig()
    char_config.eval_interval = 50
    char_config.eval_iters = 10
    char_config.vocab_size = char_tokenizer.vocab_size
    char_config.n_embd = 384
    char_config.n_head = 6
    char_config.use_rope = False
    char_config.max_iters = max_iters
    char_iters, char_train_ppl, char_val_ppl = train_model("Char-Level Baseline", char_config, char_tokenizer, data_path, "checkpoints/char_baseline.pt")
    
    # Run 2: BPE Baseline
    bpe_config = GPTConfig()
    bpe_config.eval_interval = 50
    bpe_config.eval_iters = 10
    bpe_config.vocab_size = bpe_tokenizer.vocab_size
    bpe_config.n_embd = 256
    bpe_config.n_head = 8
    bpe_config.use_rope = False
    bpe_config.max_iters = max_iters
    bpe_iters, bpe_train_ppl, bpe_val_ppl = train_model("BPE Baseline", bpe_config, bpe_tokenizer, data_path, "checkpoints/bpe_baseline.pt")
    
    # Run 3: BPE + RoPE
    rope_config = GPTConfig()
    rope_config.eval_interval = 50
    rope_config.eval_iters = 10
    rope_config.vocab_size = bpe_tokenizer.vocab_size
    rope_config.n_embd = 256
    rope_config.n_head = 8
    rope_config.use_rope = True
    rope_config.max_iters = max_iters
    rope_iters, rope_train_ppl, rope_val_ppl = train_model("BPE + RoPE", rope_config, bpe_tokenizer, data_path, "checkpoints/bpe_rope.pt")
    
    # Plotting Perplexity
    plt.figure(figsize=(10, 6))
    plt.plot(char_iters, char_val_ppl, label="Char Baseline - Val PPL", linestyle='-', color='gray')
    plt.plot(bpe_iters, bpe_val_ppl, label="BPE Baseline - Val PPL", linestyle='-', color='blue')
    plt.plot(rope_iters, rope_val_ppl, label="BPE + RoPE - Val PPL", linestyle='-', color='red')
    
    plt.yscale('log')
    plt.xlabel("Iterations")
    plt.ylabel("Validation Perplexity (Log Scale)")
    plt.title("Ablation Study: Perplexity Comparison")
    plt.legend()
    plt.grid(True)
    
    os.makedirs("outputs", exist_ok=True)
    plot_path = os.path.join("outputs", "perplexity_comparison.png")
    plt.savefig(plot_path)
    print(f"\nPerplexity plot saved to {plot_path}")
    
    # Qualitative Evaluation
    print("\nStarting Qualitative Evaluation...")
    prompts = [
        "ROMEO:\n",
        "KING RICHARD II:\n",
        "To be, or not to be",
        "O Romeo, Romeo!"
    ]
    
    # Load models
    def load_model(config_obj, path):
        m = GPTLanguageModel(config_obj)
        if torch.backends.mps.is_available():
            dev = 'mps'
        elif torch.cuda.is_available():
            dev = 'cuda'
        else:
            dev = 'cpu'
        m.load_state_dict(torch.load(path, map_location=dev))
        m.to(dev)
        m.config.device = dev
        return m

    char_model = load_model(char_config, "checkpoints/char_baseline.pt")
    bpe_model = load_model(bpe_config, "checkpoints/bpe_baseline.pt")
    rope_model = load_model(rope_config, "checkpoints/bpe_rope.pt")
    
    results = []
    for prompt in prompts:
        char_out = generate_text(char_model, char_tokenizer, prompt)
        bpe_out = generate_text(bpe_model, bpe_tokenizer, prompt)
        rope_out = generate_text(rope_model, bpe_tokenizer, prompt)
        results.append((prompt, char_out, bpe_out, rope_out))
        
    # Write Markdown file
    md_path = "outputs/generation_comparison.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Qualitative Generation Evaluation\n\n")
        f.write("This document compares the output generated by the three different architectural versions of our model after 2000 iterations of training.\n\n")
        
        for prompt, char_out, bpe_out, rope_out in results:
            f.write(f"## Prompt: `{prompt.strip()}`\n\n")
            f.write("| Char-Level Baseline | BPE Baseline | BPE + RoPE |\n")
            f.write("|---|---|---|\n")
            # Replace newlines with <br> for markdown tables
            c_br = char_out.replace('\\n', '<br>').replace('\n', '<br>')
            b_br = bpe_out.replace('\\n', '<br>').replace('\n', '<br>')
            r_br = rope_out.replace('\\n', '<br>').replace('\n', '<br>')
            f.write(f"| {c_br} | {b_br} | {r_br} |\n\n")
            
    print(f"Qualitative evaluation saved to {md_path}")

if __name__ == "__main__":
    main()
