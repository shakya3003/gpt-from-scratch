import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

class BPETokenizer:
    def __init__(self, data_path: str, vocab_size: int = 5000):
        self.data_path = data_path
        self._vocab_size = vocab_size
        
        # We will save the tokenizer to a file next to the data path
        self.save_path = data_path.replace(".txt", "_bpe.json")
        
        if os.path.exists(self.save_path):
            # Load the existing tokenizer
            print(f"Loading existing BPE tokenizer from {self.save_path}")
            self.tokenizer = Tokenizer.from_file(self.save_path)
        else:
            # Train a new tokenizer
            print(f"Training new BPE tokenizer on {self.data_path} with vocab size {vocab_size}...")
            self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            self.tokenizer.pre_tokenizer = ByteLevel()
            
            trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"], vocab_size=vocab_size)
            self.tokenizer.train([data_path], trainer)
            self.tokenizer.save(self.save_path)
            print("Tokenizer training complete and saved.")
            
        # Also need to load the full text for compatibility with train.py
        with open(data_path, 'r', encoding='utf-8') as f:
            self.text = f.read()

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(self, s: str) -> list[int]:
        """Encoder: take a string, output a list of integers"""
        return self.tokenizer.encode(s).ids
        
    def decode(self, l: list[int]) -> str:
        """Decoder: take a list of integers, output a string"""
        return self.tokenizer.decode(l)

class CharacterTokenizer:
    def __init__(self, data_path: str):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
            
        chars = sorted(list(set(self.text)))
        self._vocab_size = len(chars)
        
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s if c in self.stoi]
        
    def decode(self, l: list[int]) -> str:
        return ''.join([self.itos[i] for i in l])
