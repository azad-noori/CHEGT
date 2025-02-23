import torch
import torch.nn as nn
import torch.nn.functional as F

class HINormerLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout):
        super(HINormerLayer, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, x, adj):
        x = self.norm1(x)
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm2(x)
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        return x

class HINormerEncoder(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout, n_layers):
        super(HINormerEncoder, self).__init__()
        self.layers = nn.ModuleList([HINormerLayer(hidden_dim, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)
        return x