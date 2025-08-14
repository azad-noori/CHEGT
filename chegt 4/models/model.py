import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoders import GNNEncoder, HINormer
from ..utils.sampling import AdaptiveContrastiveSampler

class ContrastiveHeteroModel(nn.Module):
    def __init__(self, hidden_dim, out_dim, num_heads, metadata, sampler_quantile=0.7, n_clusters=5, update_interval=10):
        super().__init__()
        self.gnn_encoder = GNNEncoder(hidden_dim, out_dim, metadata)
        self.hinormer = HINormer(hidden_dim, out_dim, num_heads, metadata)
        
        self.sampler = AdaptiveContrastiveSampler(
            quantile=sampler_quantile,
            n_clusters=n_clusters,
            update_interval=update_interval
        )
        
        self.gnn_projector = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
        self.hinormer_projector = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
        self.temperature = 0.5
        
    def forward(self, x_dict, edge_index_dict):
        gnn_emb = self.gnn_encoder(x_dict, edge_index_dict)
        hinormer_emb = self.hinormer(x_dict, edge_index_dict)
        
        gnn_proj = {k: self.gnn_projector(v) for k, v in gnn_emb.items()}
        hinormer_proj = {k: self.hinormer_projector(v) for k, v in hinormer_emb.items()}
        
        return gnn_proj, hinormer_proj, gnn_emb, hinormer_emb
    
    def contrastive_loss(self, gnn_proj, hinormer_proj, node_type='paper', current_epoch=0):
        combined_emb = (gnn_emb[node_type] + hinormer_emb[node_type]) / 2
        num_nodes = combined_emb.size(0)
        
        self.sampler(combined_emb, gnn_emb[node_type].edge_index_dict, num_nodes, current_epoch)
        
        positive_pairs = self.sampler.get_all_positive_pairs()
        
        loss = 0
        count = 0
        
        z1 = F.normalize(gnn_proj[node_type], dim=1)
        z2 = F.normalize(hinormer_proj[node_type], dim=1)
        
        for u, v in positive_pairs:
            pos_sim = torch.dot(z1[u], z2[v]) / self.temperature
            
            neg_indices_u = self.sampler.find_negative_samples(u, num_negatives=5)
            if len(neg_indices_u) > 0:
                neg_sim_u = torch.dot(z1[u], z2[neg_indices_u].T) / self.temperature
                loss_u = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim_u).sum()))
                loss += loss_u
            
            neg_indices_v = self.sampler.find_negative_samples(v, num_negatives=5)
            if len(neg_indices_v) > 0:
                neg_sim_v = torch.dot(z1[v], z2[neg_indices_v].T) / self.temperature
                loss_v = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim_v).sum()))
                loss += loss_v
            
            count += 1
        
        if count == 0:
            return torch.tensor(0.0, device=z1.device)
        
        return loss / count