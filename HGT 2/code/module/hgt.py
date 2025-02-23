import torch.nn as nn
import torch.nn.functional as F
from .gat_encoder import GATEncoder
from .hinormer_encoder import HINormerEncoder
from .contrast import Contrast

class HGT(nn.Module):
    def __init__(self, hidden_dim, feats_dim_list, feat_drop, attn_drop, sample_rate, nei_num, tau, lam, alpha):
        super(HGT, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True) for feats_dim in feats_dim_list])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x

        self.gat = GATEncoder(hidden_dim, hidden_dim, attn_drop, alpha=0.2, nheads=4)
        self.hinormer = HINormerEncoder(hidden_dim, n_heads=4, dropout=attn_drop, n_layers=2)
        self.contrast = Contrast(hidden_dim, tau, lam)

    def forward(self, feats, pos, adj, nei_index):
        h_all = []
        for i in range(len(feats)):
            h_all.append(F.elu(self.feat_drop(self.fc_list[i](feats[i]))))
        
        z_gat = self.gat(h_all[0], adj)  # استفاده از GAT روی ماتریس مجاورت
        z_hinormer = self.hinormer(h_all[0], adj)  # استفاده از HINormer روی ماتریس مجاورت
        
        loss = self.contrast(z_gat, z_hinormer, pos)  # محاسبه تابع هزینه
        return loss

    def get_embeds(self, feats, adj):
        z_gat = F.elu(self.fc_list[0](feats[0]))
        z_gat = self.gat(z_gat, adj)
        z_hinormer = F.elu(self.fc_list[0](feats[0]))
        z_hinormer = self.hinormer(z_hinormer, adj)
        z_final = 0.5 * z_gat + 0.5 * z_hinormer  # ترکیب دو نمای GAT و HINormer
        return z_final.detach()