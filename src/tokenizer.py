import os

class CharacterTokenizer:
    def __init__(self, data_path: str):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
            
        # Get all unique characters in the text
        chars = sorted(list(set(self.text)))
        self.vocab_size = len(chars)
        
        # Create mapping from characters to integers and vice versa
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
    def encode(self, s: str) -> list[int]:
        """Encoder: take a string, output a list of integers"""
        return [self.stoi[c] for c in s if c in self.stoi]
        
    def decode(self, l: list[int]) -> str:
        """Decoder: take a list of integers, output a string"""
        return ''.join([self.itos[i] for i in l])
