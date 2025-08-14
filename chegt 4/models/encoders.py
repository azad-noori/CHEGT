import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, GATv2Conv

class GNNEncoder(nn.Module):
    def __init__(self, hidden_dim, out_dim, metadata):
        super().__init__()
        self.conv1 = HeteroConv({
            ('author', 'writes', 'paper'): GATConv((-1, -1), hidden_dim, add_self_loops=False),
            ('paper', 'written_by', 'author'): GATConv((-1, -1), hidden_dim, add_self_loops=False),
            ('paper', 'about', 'subject'): GATConv((-1, -1), hidden_dim, add_self_loops=False),
            ('subject', 'includes', 'paper'): GATConv((-1, -1), hidden_dim, add_self_loops=False),
        }, aggr='mean')
        
        self.conv2 = HeteroConv({
            ('author', 'writes', 'paper'): GATConv((-1, -1), out_dim, add_self_loops=False),
            ('paper', 'written_by', 'author'): GATConv((-1, -1), out_dim, add_self_loops=False),
            ('paper', 'about', 'subject'): GATConv((-1, -1), out_dim, add_self_loops=False),
            ('subject', 'includes', 'paper'): GATConv((-1, -1), out_dim, add_self_loops=False),
        }, aggr='mean')
        
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.elu(value) for key, value in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: self.norm(value) for key, value in x_dict.items()}
        return x_dict

class HINormer(nn.Module):
    def __init__(self, hidden_dim, out_dim, num_heads, metadata):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.metadata = metadata
        
        self.projection = nn.ModuleDict({
            node_type: nn.Linear(in_channels, hidden_dim)
            for node_type, in_channels in metadata[0].items()
        })
        
        self.conv1 = HeteroConv({
            ('author', 'writes', 'paper'): GATv2Conv((-1, -1), hidden_dim, heads=num_heads, add_self_loops=False),
            ('paper', 'written_by', 'author'): GATv2Conv((-1, -1), hidden_dim, heads=num_heads, add_self_loops=False),
            ('paper', 'about', 'subject'): GATv2Conv((-1, -1), hidden_dim, heads=num_heads, add_self_loops=False),
            ('subject', 'includes', 'paper'): GATv2Conv((-1, -1), hidden_dim, heads=num_heads, add_self_loops=False),
        }, aggr='mean')
        
        self.conv2 = HeteroConv({
            ('author', 'writes', 'paper'): GATv2Conv((-1, -1), out_dim, heads=1, add_self_loops=False),
            ('paper', 'written_by', 'author'): GATv2Conv((-1, -1), out_dim, heads=1, add_self_loops=False),
            ('paper', 'about', 'subject'): GATv2Conv((-1, -1), out_dim, heads=1, add_self_loops=False),
            ('subject', 'includes', 'paper'): GATv2Conv((-1, -1), out_dim, heads=1, add_self_loops=False),
        }, aggr='mean')
        
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(self, x_dict, edge_index_dict):
        projected = {
            node_type: self.projection[node_type](x)
            for node_type, x in x_dict.items()
        }
        
        x_dict = self.conv1(projected, edge_index_dict)
        x_dict = {key: F.elu(value) for key, value in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: self.norm(value) for key, value in x_dict.items()}
        
        return x_dict