import torch.nn as nn
import torch
from collections import defaultdict
import math

class MultiheadLatentAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, n_heads, d_c, d_R, qkv_bias = False):
        super().__init__()

        self.d_out = d_out                  # Output dimension
        self.n_heads = n_heads              # Number of attention heads    
        self.head_dim = d_out // n_heads    # Dimension of each attention head
        self.d_c = d_c                      # Compressed latent dimension
        self.d_R = d_R                      # Decoupled RoPE dimension of each attention head

        # Down Projections
        self.W_DKV = nn.Linear(d_out, d_c, bias=qkv_bias) # Down projection of K/V
        self.W_DQ = nn.Linear(d_out, d_c, bias=qkv_bias) # Down projection of Q
        
        # Up projections
        self.W_UK = nn.Linear(d_c, d_out, bias=qkv_bias) # Up projection of K
        self.W_UV = nn.Linear(d_c, d_out, bias=qkv_bias) # Up projection of V
        self.W_UQ = nn.Linear(d_c, d_out, bias=qkv_bias) # Up projection of Q
        
        # RoPE projections
        self.W_KR = nn.Linear(d_out, d_R, bias=qkv_bias) # Shared key-rope projection
        self.W_QR = nn.Linear(d_c, n_heads * d_R, bias=qkv_bias) # Decoupled queries for each head

        self.output_projection = nn.Linear(d_out, d_out) # Used to combine the final multiple head outputs
        self.dropout = nn.Dropout(dropout)
        self.cache = defaultdict(list)

    def forward(self, x, cache=False):
        batch, n_tokens, d_out = x.shape
        
        # K/V Compression
        c_kv = self.W_DKV(x)    # (batch, n_tokens, d_c)
        K = self.W_UK(c_kv)     # (batch, n_tokens, d_out)
        V = self.W_UV(c_kv)     # (batch, n_tokens, d_out)
        
        K = K.view(batch, n_tokens, self.n_heads, self.head_dim).transpose(1, 2)  # (batch, n_heads, n_tokens, head_dim)
        V = V.view(batch, n_tokens, self.n_heads, self.head_dim).transpose(1, 2)  # (batch, n_heads, n_tokens, head_dim)
        
        # Cache latent vector
        if cache:
            self.cache['c_kv'].append(c_kv)
            
        # Query projection
        c_q = self.W_DQ(x)
        Q = self.W_UQ(c_q)
        Q = Q.view(batch, n_tokens, self.n_heads, self.head_dim).transpose(1, 2)
            
        # Decoupled RoPE
        k_R = self.W_KR(K)
        k_R = self._rope_rotate(k_R)  # (batch, n_tokens, d_R)
        k_R = k_R.unsqueeze(1).expand(batch, self.n_heads, n_tokens, self.d_R) # Broadcast to heads
        q_R = self.W_QR(c_q)
        q_R = q_R.view(batch, n_tokens, self.n_heads, self.d_R)
        q_R = self._rope_rotate(q_R)
        
        # Concat decoupled partsw
        K = torch.cat([K, k_R], dim=-1)  # (batch, n_heads, n_tokens, d_out + d_R)
        Q = torch.cat([Q, q_R], dim=-1)
        
        # Scale dot product
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))
        attn_scores = attn_scores / math.sqrt(self.head_dim + self.d_R)
        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)  # Apply dropout to attention scores
        
        context_vec = torch.matmul(attn, V)
        context_vec = torch.transpose(1, 2).reshape(batch, n_tokens, d_out)
        context_vec = self.output_projection(context_vec)       
        
        return context_vec
        
    def _rope_rotate(self, x):
        '''
        Applies Rotary Position Embedding (RoPE) to the input tensor x.
        '''
        if x.dim() == 4 and x.shape[1] == x.shape[2]:
            raise ValueError("Invalid input shape")

        if x.shape[-1] % 2 != 0:
            raise ValueError("RoPE dim is not even")
        
        half_dim = x.shape[-1] // 2
        freq_seq = torch.arange(half_dim, dtype=torch.float32, device=x.device)
        inv_freq_seq = 1.0 / (10000 ** (freq_seq / half_dim))
        
        seq_len = x.shape[-2]
        t = torch.arange(seq_len, dtype=torch.float32, device=x.device)
        freqs = torch.einsum('i,j->ij', t, inv_freq_seq)
        sin, cos = freqs.sin(), freqs.cos()
        
        for _ in range(x.dim() - 2):
            sin = sin.unsqueeze(0)
            cos = cos.unsqueeze(0)
            
        sin = sin.expand_as(x[..., :half_dim])
        cos = cos.expand_as(x[..., :half_dim])
        
        x1, x2 = x[..., :half_dim], x[..., half_dim:]
        
        x_rotated = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
        
        return x_rotated