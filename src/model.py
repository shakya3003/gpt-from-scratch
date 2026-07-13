import torch
import torch.nn as nn
from torch.nn import functional as F
from src.config import GPTConfig

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel (Vectorized with Flash Attention) """
    def __init__(self, num_heads: int, head_size: int, config: GPTConfig):
        super().__init__()
        assert config.n_embd == num_heads * head_size, "Embedding dimension must be divisible by number of heads"
        # Key, Query, Value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout_p = config.dropout
        self.dropout = nn.Dropout(config.dropout)
        
        self.num_heads = num_heads
        self.head_size = head_size
        self.n_embd = config.n_embd
        self.use_rope = config.use_rope
        
        if self.use_rope:
            # Precompute cos and sin for RoPE
            inv_freq = 1.0 / (10000 ** (torch.arange(0, head_size, 2).float() / head_size))
            t = torch.arange(config.block_size, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x):
        B, T, C = x.size()
        
        # calculate query, key, values for all heads in batch
        qkv = self.c_attn(x) # (B, T, 3 * C)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # reshape to (B, num_heads, T, head_size)
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        
        if getattr(self, "use_rope", False):
            cos = self.cos_cached[:, :, :T, :].to(q.device)
            sin = self.sin_cached[:, :, :T, :].to(q.device)
            def rotate_half(x):
                x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
                return torch.cat((-x2, x1), dim=-1)
            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin)

        # Flash Attention (causal)
        out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=self.attn_dropout_p if self.training else 0.0, 
            is_causal=True
        )
        
        # re-assemble all head outputs side by side
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # output projection
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, config: GPTConfig):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.sa = MultiHeadAttention(config.n_head, head_size, config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        # Pre-norm formulation: layer norm BEFORE self-attention and feed-forward
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        if not config.use_rope:
            self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd) # Final layer norm
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        # Better weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        
        if self.config.use_rope:
            x = tok_emb # (B, T, C)
        else:
            pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, C)
            x = tok_emb + pos_emb # (B, T, C)
            
        x = self.blocks(x)    # (B, T, C)
        x = self.ln_f(x)      # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.config.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            
            # apply temperature scaling
            if temperature != 1.0:
                logits = logits / temperature
                
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
