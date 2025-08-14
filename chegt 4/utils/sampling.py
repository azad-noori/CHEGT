import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

class ContrastiveViewSampler:
    """
    نمونه‌ساز برای یادگیری متضاد دیدها
    این کلاس جفت‌های مثبت و منفی را برای یادگیری متضاد بین دو دید تعریف می‌کند
    """
    
    def __init__(self, quantile=0.7, n_clusters=5, update_interval=10):
        self.quantile = quantile
        self.n_clusters = n_clusters
        self.update_interval = update_interval
        self.similarity_matrix = None
        self.cn_matrix = None
        self.clusters = None
        self.thresholds_si = None
        self.threshold_cn = None
        
    def compute_similarity_matrix(self, embeddings):
        """مرحله 1: محاسبه ماتریس شباهت کسینوسی (رابطه 3-11)"""
        emb_np = embeddings.cpu().numpy()
        sim_matrix = cosine_similarity(emb_np)
        self.similarity_matrix = torch.tensor(sim_matrix, device=embeddings.device)
        return self.similarity_matrix
    
    def compute_common_neighbors_matrix(self, edge_index_dict, num_nodes):
        """مرحله 3: محاسبه ماتریس همسایگان مشترک (رابطه 27-3)"""
        device = edge_index_dict[next(iter(edge_index_dict.keys()))].device
        
        # ایجاد ماتریس مجاورت برای همه انواع یال‌ها
        adj_matrix = torch.zeros(num_nodes, num_nodes, device=device)
        
        for edge_type, edge_index in edge_index_dict.items():
            adj_matrix[edge_index[0], edge_index[1]] = 1
            adj_matrix[edge_index[1], edge_index[0]] = 1  # گراف بدون جهت
        
        # محاسبه تعداد همسایگان مشترک
        cn_matrix = torch.mm(adj_matrix, adj_matrix)
        cn_matrix.fill_diagonal_(0)  # حذف خود گره
        
        self.cn_matrix = cn_matrix
        return cn_matrix
    
    def cluster_nodes(self, embeddings):
        """مرحله 2: خوشه‌بندی گره‌ها با k-means بر اساس شباهت کسینوسی"""
        emb_np = embeddings.cpu().numpy()
        
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(emb_np)
        
        self.clusters = {}
        for i in range(self.n_clusters):
            self.clusters[i] = torch.tensor(
                np.where(cluster_labels == i)[0], 
                device=embeddings.device
            )
        
        return self.clusters
    
    def compute_thresholds(self):
        """مرحله 4: محاسبه آستانه‌های پویا (روابط 28-3 و 29-3)"""
        # محاسبه τ_cn (آستانه همسایگی مشترک برای کل گراف)
        all_cn_values = self.cn_matrix[self.cn_matrix > 0].flatten()
        if len(all_cn_values) > 0:
            self.threshold_cn = torch.quantile(all_cn_values, self.quantile)
        else:
            self.threshold_cn = torch.tensor(0.0, device=self.cn_matrix.device)
        
        # محاسبه τ_si برای هر خوشه
        self.thresholds_si = {}
        for i in range(self.n_clusters):
            cluster_nodes = self.clusters[i]
            if len(cluster_nodes) > 1:
                cluster_sim = self.similarity_matrix[cluster_nodes][:, cluster_nodes]
                self.thresholds_si[i] = torch.quantile(cluster_sim.flatten(), self.quantile)
            else:
                self.thresholds_si[i] = torch.tensor(0.0, device=self.similarity_matrix.device)
        
        return self.thresholds_si, self.threshold_cn
    
    def find_positive_samples(self, node_idx):
        """مرحله 5: پیدا کردن نمونه‌های مثبت (رابطه 30-3)"""
        if self.clusters is None or self.thresholds_si is None:
            raise ValueError("ابتدا باید خوشه‌بندی و آستانه‌ها محاسبه شوند")
        
        # پیدا کردن خوشه‌ای که گره در آن قرار دارد
        node_cluster = None
        for i, nodes in self.clusters.items():
            if node_idx in nodes:
                node_cluster = i
                break
        
        if node_cluster is None:
            return torch.tensor([], device=self.similarity_matrix.device)
        
        # گره‌های هم‌خوشه‌ای
        cluster_nodes = self.clusters[node_cluster]
        
        # محاسبه شباهت و همسایگی مشترک با گره هدف
        sim_values = self.similarity_matrix[node_idx, cluster_nodes]
        cn_values = self.cn_matrix[node_idx, cluster_nodes]
        
        # اعمال معیارها (رابطه 30-3)
        threshold_si = self.thresholds_si[node_cluster]
        mask = (sim_values >= threshold_si) & (cn_values >= self.threshold_cn)
        
        # حذف خود گره از نتایج
        self_mask = cluster_nodes != node_idx
        positive_indices = cluster_nodes[mask & self_mask]
        
        return positive_indices
    
    def find_negative_samples(self, node_idx, num_negatives=None):
        """پیدا کردن نمونه‌های منفی"""
        if self.clusters is None or self.thresholds_si is None:
            raise ValueError("ابتدا باید خوشه‌بندی و آستانه‌ها محاسبه شوند")
        
        # پیدا کردن خوشه‌ای که گره در آن قرار دارد
        node_cluster = None
        for i, nodes in self.clusters.items():
            if node_idx in nodes:
                node_cluster = i
                break
        
        # گره‌های در خوشه‌های دیگر (نمونه‌های منفی بالقوه)
        negative_candidates = []
        for i, nodes in self.clusters.items():
            if i != node_cluster:
                negative_candidates.append(nodes)
        
        if not negative_candidates:
            return torch.tensor([], device=self.similarity_matrix.device)
        
        negative_candidates = torch.cat(negative_candidates)
        
        # اگر تعداد نمونه‌های منفی مشخص نشده، همه را برمی‌گردانیم
        if num_negatives is None or len(negative_candidates) <= num_negatives:
            return negative_candidates
        
        # در غیر این صورت، نمونه‌های منفی را بر اساس شباهت کمتر انتخاب می‌کنیم
        sim_values = self.similarity_matrix[node_idx, negative_candidates]
        _, indices = torch.topk(sim_values, num_negatives, largest=False)
        return negative_candidates[indices]
    
    def get_all_positive_pairs(self):
        """دریافت همه جفت‌های مثبت"""
        positive_pairs = []
        
        for i in range(self.n_clusters):
            cluster_nodes = self.clusters[i]
            threshold_si = self.thresholds_si[i]
            
            # محاسبه شباهت و همسایگی مشترک برای همه جفت‌ها در خوشه
            cluster_sim = self.similarity_matrix[cluster_nodes][:, cluster_nodes]
            cluster_cn = self.cn_matrix[cluster_nodes][:, cluster_nodes]
            
            # ایجاد ماسک برای معیارها
            mask = (cluster_sim >= threshold_si) & (cluster_cn >= self.threshold_cn)
            
            # پیدا کردن ایندکس‌های مثبت
            positive_indices = torch.nonzero(mask, as_tuple=True)
            
            # تبدیل به ایندکس‌های اصلی
            for u_idx, v_idx in zip(*positive_indices):
                if u_idx != v_idx:  # حذف خود گره
                    u = cluster_nodes[u_idx]
                    v = cluster_nodes[v_idx]
                    positive_pairs.append((u.item(), v.item()))
        
        return positive_pairs
    
    def __call__(self, embeddings, edge_index_dict, num_nodes):
        """اجرای کامل فرآیند نمونه‌گیری"""
        # مرحله 1: محاسبه ماتریس شباهت کسینوسی
        self.compute_similarity_matrix(embeddings)
        
        # مرحله 2: خوشه‌بندی گره‌ها
        self.cluster_nodes(embeddings)
        
        # مرحله 3: محاسبه ماتریس همسایگی مشترک
        self.compute_common_neighbors_matrix(edge_index_dict, num_nodes)
        
        # مرحله 4: محاسبه آستانه‌ها
        self.compute_thresholds()
        
        return self