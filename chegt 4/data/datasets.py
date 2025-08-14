import torch
import numpy as np
import os
from torch_geometric.data import HeteroData

def load_dataset(dataset_name, root_path, split_ratio=20):
    """
    بارگذاری دیتاست‌های مختلف با ساختار یکسان
    
    پارامترها:
    - dataset_name: نام دیتاست ('acm', 'dblp', یا 'imdb')
    - root_path: مسیر پوشه دیتاست
    - split_ratio: نسبت تقسیم داده‌ها
    """
    
    # تنظیمات دیتاست
    configs = {
        'acm': {
            'node_types': ['paper', 'author', 'subject'],
            'edge_files': {
                ('author', 'writes', 'paper'): 'pa',
                ('paper', 'about', 'subject'): 'ps'
            },
            'feature_files': {
                'paper': 'p_feat',
                'author': 'a_feat',
                'subject': None  # ویژگی‌های موضوعات تصادفی
            }
        },
        'dblp': {
            'node_types': ['paper', 'author', 'conference', 'term'],
            'edge_files': {
                ('author', 'writes', 'paper'): 'pa',
                ('paper', 'published_in', 'conference'): 'pc',
                ('paper', 'has_term', 'term'): 'pt'
            },
            'feature_files': {
                'paper': 'p_feat',
                'author': 'a_feat',
                'conference': 'c_feat',
                'term': None  # ویژگی‌های ترم‌ها تصادفی
            }
        },
        'imdb': {
            'node_types': ['movie', 'actor', 'director'],
            'edge_files': {
                ('actor', 'acts_in', 'movie'): 'ma',
                ('director', 'directs', 'movie'): 'md'
            },
            'feature_files': {
                'movie': 'm_feat',
                'actor': 'a_feat',
                'director': 'd_feat'
            }
        }
    }
    
    if dataset_name not in configs:
        raise ValueError(f"Dataset {dataset_name} not supported. Supported datasets: {list(configs.keys())}")
    
    config = configs[dataset_name]
    
    # مسیر فایل‌ها
    paths = {
        'labels': os.path.join(root_path, dataset_name, 'labels'),
        'train': os.path.join(root_path, dataset_name, f'train_{split_ratio}'),
        'val': os.path.join(root_path, dataset_name, f'val_{split_ratio}'),
        'test': os.path.join(root_path, dataset_name, f'test_{split_ratio}')
    }
    
    # اضافه کردن مسیر فایل‌های ویژگی‌ها
    for node_type, filename in config['feature_files'].items():
        if filename:
            paths[f'{node_type}_feat'] = os.path.join(root_path, dataset_name, filename)
    
    # اضافه کردن مسیر فایل‌های یال‌ها
    for edge_type, filename in config['edge_files'].items():
        paths[f'{edge_type[0]}_{edge_type[2]}'] = os.path.join(root_path, dataset_name, filename)
    
    # بررسی وجود فایل‌ها
    for key, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
    
    print(f"Loading {dataset_name.upper()} dataset...")
    
    # بارگذاری ویژگی‌ها
    features = {}
    for node_type in config['node_types']:
        feat_file = config['feature_files'][node_type]
        if feat_file and os.path.exists(paths[f'{node_type}_feat']):
            print(f"Loading {node_type} features...")
            with open(paths[f'{node_type}_feat'], 'r') as f:
                feat_data = []
                for line in f:
                    if line.strip():
                        features_list = list(map(float, line.strip().split()))
                        feat_data.append(features_list)
                features[node_type] = torch.tensor(feat_data, dtype=torch.float32)
                print(f"{node_type} features shape: {features[node_type].shape}")
        else:
            # ویژگی‌های تصادفی برای گره‌هایی که فایل ویژگی ندارند
            print(f"No feature file for {node_type}, using random features...")
            features[node_type] = torch.randn(1000, 128)  # اندازه پیش‌فرض
    
    # بارگذاری یال‌ها
    edges = {}
    for edge_type, filename in config['edge_files'].items():
        print(f"Loading {edge_type[0]}-{edge_type[2]} edges...")
        edge_list = []
        with open(paths[f'{edge_type[0]}_{edge_type[2]}'], 'r') as f:
            for line in f:
                if line.strip():
                    node1, node2 = map(int, line.strip().split())
                    edge_list.append((node1, node2))
        edges[edge_type] = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        print(f"{edge_type[0]}-{edge_type[2]} edges: {edges[edge_type].shape[1]}")
    
    # بارگذاری برچسب‌ها
    print("Loading labels...")
    with open(paths['labels'], 'r') as f:
        labels = []
        for line in f:
            if line.strip():
                labels.append(int(line.strip()))
    labels = torch.tensor(labels, dtype=torch.long)
    print(f"Labels shape: {labels.shape}, Classes: {labels.unique().numel()}")
    
    # بارگذاری تقسیم‌های آموزش، اعتبارسنجی و تست
    print("Loading data splits...")
    def load_split(file_path):
        with open(file_path, 'r') as f:
            indices = [int(line.strip()) for line in f if line.strip()]
        mask = torch.zeros(len(labels), dtype=torch.bool)
        mask[indices] = True
        return mask
    
    train_mask = load_split(paths['train'])
    val_mask = load_split(paths['val'])
    test_mask = load_split(paths['test'])
    
    print(f"Train: {train_mask.sum().item()}, Val: {val_mask.sum().item()}, Test: {test_mask.sum().item()}")
    
    # ایجاد ساختار گراف ناهمگن
    data = HeteroData()
    
    # افزودن ویژگی‌ها
    for node_type in config['node_types']:
        data[node_type].x = features[node_type]
    
    # افزودن برچسب‌ها و ماسک‌ها (فقط برای گره هدف)
    target_node = config['node_types'][0]  # معمولاً اولین گره هدف است
    data[target_node].y = labels
    data[target_node].train_mask = train_mask
    data[target_node].val_mask = val_mask
    data[target_node].test_mask = test_mask
    
    # افزودن یال‌ها
    for edge_type, edge_index in edges.items():
        data[edge_type].edge_index = edge_index
        # افزودن یال‌های معکوس
        reverse_edge_type = (edge_type[2], f'reverse_{edge_type[1]}', edge_type[0])
        data[reverse_edge_type].edge_index = torch.flip(edge_index, [0])
    
    print(f"\n{dataset_name.upper()} dataset summary:")
    for node_type in config['node_types']:
        print(f"- {node_type.capitalize()}s: {data[node_type].x.size(0)}")
    for edge_type in edges.keys():
        print(f"- {edge_type[0]}-{edge_type[2]} edges: {edges[edge_type].shape[1]}")
    
    return data