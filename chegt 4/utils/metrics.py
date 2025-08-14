from sklearn.metrics import accuracy_score, f1_score, normalized_mutual_info_score, adjusted_rand_score
import numpy as np
from sklearn.cluster import KMeans

def calculate_classification_metrics(y_true, y_pred):
    """محاسبه معیارهای طبقه‌بندی"""
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    return acc, f1_macro, f1_micro

def calculate_clustering_metrics(embeddings, true_labels, n_clusters):
    """
    محاسبه معیارهای خوشه‌بندی روی امبدینگ‌ها
    
    پارامترها:
    - embeddings: امبدینگ‌های گره‌ها
    - true_labels: برچسب‌های واقعی
    - n_clusters: تعداد خوشه‌ها
    """
    # اجرای K-means روی امبدینگ‌ها
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # محاسبه NMI و ARI
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    ari = adjusted_rand_score(true_labels, cluster_labels)
    
    return nmi, ari

def calculate_all_metrics(embeddings, true_labels, y_pred, n_clusters):
    """
    محاسبه همه معیارها (طبقه‌بندی و خوشه‌بندی)
    
    خروجی:
    - dict: دیکشنری حاوی همه معیارها
    """
    # معیارهای طبقه‌بندی
    acc, f1_macro, f1_micro = calculate_classification_metrics(true_labels, y_pred)
    
    # معیارهای خوشه‌بندی
    nmi, ari = calculate_clustering_metrics(embeddings, true_labels, n_clusters)
    
    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'nmi': nmi,
        'ari': ari
    }