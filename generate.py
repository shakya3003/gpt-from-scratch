import os
import argparse
import torch
from src.config import GPTConfig
from src.tokenizer import BPETokenizer
from src.model import GPTLanguageModel

def main():
    parser = argparse.ArgumentParser(description="Generate text from the trained GPT model.")
    parser.add_argument("--prompt", type=str, default="ROMEO:\n", help="The starting prompt for generation.")
    parser.add_argument("--max_tokens", type=int, default=500, help="Maximum number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling. Lower is more deterministic.")
    parser.add_argument("--top_k", type=int, default=None, help="Top-K sampling to restrict to top K most likely tokens.")
    args = parser.parse_args()

    config = GPTConfig()
    
    if torch.backends.mps.is_available():
        config.device = 'mps'
    elif torch.cuda.is_available():
        config.device = 'cuda'
    else:
        config.device = 'cpu'

    print(f"Using device: {config.device}")

    data_path = os.path.join("data", "tinyshakespeare.txt")
    tokenizer = BPETokenizer(data_path, vocab_size=5000)
    config.vocab_size = tokenizer.vocab_size

    # Initialize model and load weights
    model = GPTLanguageModel(config)
    checkpoint_path = os.path.join("checkpoints", "gpt_model.pt")
    
    if not os.path.exists(checkpoint_path):
        print("No trained model found! Please run train.py first.")
        return
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
    model.to(config.device)
    model.eval()

    print(f"Model loaded successfully. Generating {args.max_tokens} tokens with temperature {args.temperature}...\n")
    print("-" * 50)
    
    # Encode the text into integers
    context_list = tokenizer.encode(args.prompt)
    
    # Convert to a PyTorch tensor with shape (1, T)
    context = torch.tensor([context_list], dtype=torch.long, device=config.device)
    
    # Generate new tokens
    generated_tokens = model.generate(
        context, 
        max_new_tokens=args.max_tokens, 
        temperature=args.temperature, 
        top_k=args.top_k
    )[0].tolist()
    
    # Decode and print the entire sequence (prompt + generated text)
    print(tokenizer.decode(generated_tokens))
    print("-" * 50)

if __name__ == "__main__":
    main()
