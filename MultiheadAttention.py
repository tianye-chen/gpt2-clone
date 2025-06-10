import torch.nn as nn
import torch

class MultiheadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, n_heads, qkv_bias = False):
        super().__init__()

        self.d_out = d_out
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads

        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.output_projection = nn.Linear(d_out, d_out) # Used to combine the final multiple head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        batch, n_tokens, d_in = x.shape

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # Project each QKV matrices into a 4D matrix to accommodate multiple heads 
        # by unrolling the last dimension
        q = q.view(batch, n_tokens, self.n_heads, self.head_dim)
        k = k.view(batch, n_tokens, self.n_heads, self.head_dim)
        v = v.view(batch, n_tokens, self.n_heads, self.head_dim)

        # Group matrices wrt. n_heads
        # (batch, n_tokens, n_heads, head_dim) -> (b, n_heads, n_token, head_dim)
        # Swap dimension 1 and dimension 2
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Dot product for each head, transpose to make matrices compatible for multiplication
        attn_scores = q @ k.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(self.mask.bool()[:n_tokens, :n_tokens], -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / k.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context_vec = (attn_weights @ v).transpose(1, 2)
        context_vec = context_vec.contiguous().view(batch, n_tokens, self.d_out)
        context_vec = self.output_projection(context_vec)
        
        return context_vec
        
        