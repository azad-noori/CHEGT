import torch
import torch.nn as nn


class Contrast(nn.Module):
    def __init__(self, hidden_dim, tau, lam):
        super(Contrast, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau
        self.lam = lam
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def forward(self, z_msp, z_sc, pos):
        z_proj_msp = self.proj(z_msp)
        z_proj_sc = self.proj(z_sc)
        matrix_msp2sc = self.sim(z_proj_msp, z_proj_sc)
        matrix_sc2msp = matrix_msp2sc.t()
        
        matrix_msp2sc = matrix_msp2sc/(torch.sum(matrix_msp2sc, dim=1).view(-1, 1) + 1e-8)
        lori_msp = -torch.log(matrix_msp2sc.mul(pos.to_dense()).sum(dim=-1)).mean()

        matrix_sc2msp = matrix_sc2msp / (torch.sum(matrix_sc2msp, dim=1).view(-1, 1) + 1e-8)
        lori_sc = -torch.log(matrix_sc2msp.mul(pos.to_dense()).sum(dim=-1)).mean()
        return self.lam * lori_msp + (1 - self.lam) * lori_sc