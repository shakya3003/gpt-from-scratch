import os
import torch
from src.config import GPTConfig
from src.tokenizer import CharacterTokenizer
from src.model import GPTLanguageModel

def main():
    config = GPTConfig()
    
    if torch.backends.mps.is_available():
        config.device = 'mps'
    elif torch.cuda.is_available():
        config.device = 'cuda'
    else:
        config.device = 'cpu'

    print(f"Using device: {config.device}")

    data_path = os.path.join("data", "tinyshakespeare.txt")
    tokenizer = CharacterTokenizer(data_path)
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

    print("Model loaded successfully. Generating text...\n")
    print("-" * 50)

    # Set your custom prompt here
    start_text = "Romeo\n"
    
    # Encode the text into integers
    context_list = tokenizer.encode(start_text)
    
    # Convert to a PyTorch tensor with shape (1, T)
    context = torch.tensor([context_list], dtype=torch.long, device=config.device)
    
    # Generate 500 new tokens on top of your prompt
    generated_tokens = model.generate(context, max_new_tokens=500)[0].tolist()
    
    # Decode and print the entire sequence (prompt + generated text)
    print(tokenizer.decode(generated_tokens))
    print("-" * 50)

if __name__ == "__main__":
    main()
