# pyrefly: ignore [missing-import]
import torch

def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    """
    Generate a small batch of data of inputs x and targets y
    """
    # Generate random starting indices for the batch
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # x is the context, y is the target (shifted by 1)
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    # Move to the specified device (CPU, MPS, or CUDA)
    x, y = x.to(device), y.to(device)
    return x, y
