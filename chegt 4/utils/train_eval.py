import torch
import torch.nn.functional as F
import numpy as np
from .metrics import calculate_classification_metrics, calculate_clustering_metrics

class EarlyStopping:
    """
    پیاده‌سازی Early Stop برای جلوگیری از overfitting
    
    پارامترها:
    - patience: تعداد epoch‌هایی که منتظر بهبود می‌ماند
    - min_delta: حداقل بهبود برای شمارش بهبود
    - restore_best_weights: آیا بهترین وزن‌ها بازیابی شود
    """
    
    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = np.inf
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict()
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
        return False

def train_epoch(model, data, classifier, optimizer, device, config, current_epoch, target_node):
    """آموزش یک دوره"""
    model.train()
    optimizer.zero_grad()
    
    gnn_proj, hinormer_proj, gnn_emb, hinormer_emb = model(data.x_dict, data.edge_index_dict)
    
    contrastive_loss = model.contrastive_loss(
        gnn_proj, hinormer_proj, 
        node_type=target_node,
        current_epoch=current_epoch
    )
    
    node_emb = (gnn_emb[target_node] + hinormer_emb[target_node]) / 2
    classification_loss = F.cross_entropy(
        classifier(node_emb[data[target_node].train_mask]),
        data[target_node].y[data[target_node].train_mask]
    )
    
    total_loss = contrastive_loss + classification_loss
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item()

def evaluate_model(model, data, classifier, device, mask_type, target_node):
    """ارزیابی مدل برای طبقه‌بندی"""
    model.eval()
    with torch.no_grad():
        gnn_proj, hinormer_proj, gnn_emb, hinormer_emb = model(data.x_dict, data.edge_index_dict)
        node_emb = (gnn_emb[target_node] + hinormer_emb[target_node]) / 2
        
        if mask_type == 'val':
            mask = data[target_node].val_mask
        elif mask_type == 'test':
            mask = data[target_node].test_mask
        else:
            raise ValueError("mask_type must be 'val' or 'test'")
        
        logits = classifier(node_emb[mask])
        pred = logits.argmax(dim=1)
        true = data[target_node].y[mask]
        
        acc, f1_macro, f1_micro = calculate_classification_metrics(true.cpu(), pred.cpu())
        return acc, f1_macro, f1_micro, node_emb[mask].cpu(), true.cpu()

def evaluate_clustering(embeddings, true_labels, n_clusters):
    """ارزیابی خوشه‌بندی روی امبدینگ‌ها"""
    nmi, ari = calculate_clustering_metrics(embeddings, true_labels, n_clusters)
    return nmi, ari