import torch
import torch.nn as nn
from MultiheadAttention import MultiheadAttention
from MultiheadLatentAttention import MultiheadLatentAttention

class SimpleGPT(nn.Module):
    def __init__(self, config, use_mla=False):
        super().__init__()
        
        self.token_emb = nn.Embedding(config['vocab_size'], config["emb_dim"])
        self.pos_emb = nn.Embedding(config['context_length'], config['emb_dim'])
        self.dropout = nn.Dropout(config['dropout_rate'])
        
        self.transformer_blocks = nn.Sequential(
            *[Transformer(config, use_mla) for _ in range(config['n_layers'])]
        )
        self.final_norm = LayerNorm(config['emb_dim'])
        self.out = nn.Linear(config['emb_dim'], config['vocab_size'], bias=False)
        
    def forward(self, inp):
        batch, seq_len = inp.shape
        
        token_emb = self.token_emb(inp)
        pos_emb = self.pos_emb(torch.arange(seq_len, device=inp.device, dtype=torch.long))
        
        x = token_emb + pos_emb
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out(x)
        
        return logits
    
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        
    def forward(self, x): 
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        
        x = (x - mean) / torch.sqrt(var + self.eps)
        
        return self.scale * x + self.shift
    
class Transformer(nn.Module):
    def __init__(self, config, use_mla=False):
        super().__init__()
        
        if use_mla:
            self.attn = MultiheadLatentAttention(
                d_in = config['emb_dim'],
                d_out = config['emb_dim'],
                context_length= config['context_length'],
                n_heads = config['n_heads'],
                dropout = config['dropout_rate'],
                d_R = config['d_R'],
                d_c = config['d_c'],
                qkv_bias = config['qkv_bias']
            )
        else:
            self.attn = MultiheadAttention(
                d_in = config['emb_dim'],
                d_out = config['emb_dim'],
                context_length= config['context_length'],
                n_heads = config['n_heads'],
                dropout = config['dropout_rate'],
                qkv_bias = config['qkv_bias']
            )
        
        self.attn_block = nn.Sequential(
            LayerNorm(config['emb_dim']),
            self.attn,
            nn.Dropout(config['dropout_rate'])
        )
        
        self.ff_block = nn.Sequential(
            LayerNorm(config['emb_dim']),
            nn.Linear(config['emb_dim'], config['emb_dim'] * 4),
            nn.GELU(),
            nn.Linear(config['emb_dim'] * 4, config['emb_dim']),
            nn.Dropout(config['dropout_rate'])
        )
        
    def forward(self, x):
        residual = x
        x = self.attn_block(x)
        x += residual
        
        residual = x
        x = self.ff_block(x)
        x += residual
        
        return x
    
