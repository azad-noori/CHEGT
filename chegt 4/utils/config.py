class Config:
    # هایپرپارامترهای مدل
    HIDDEN_DIM = 128      # بعد مخفی
    OUT_DIM = 64          # بعد خروجی
    NUM_HEADS = 4         # تعداد سرهای توجه
    LR = 0.001           # نرخ یادگیری
    EPOCHS = 200         # تعداد دوره‌های آموزش
    WEIGHT_DECAY = 1e-5  # کاهش وزن
    
    # تنظیمات دیتاست
    DATASET_NAME = 'acm'  # می‌تواند 'acm', 'dblp', یا 'imdb' باشد
    DATASET_ROOT = './data'
    SPLIT_RATIO = 20      # 20, 40, یا 60
    
    # تنظیمات نمونه‌گیری تطبیقی
    SAMPLER_QUANTILE = 0.7    # کوانتایل برای آستانه
    N_CLUSTERS = 5            # تعداد خوشه‌ها
    UPDATE_INTERVAL = 10      # فاصله به‌روزرسانی خوشه‌بندی
    
    # تنظیمات Early Stopping
    EARLY_STOPPING_PATIENCE = 15  # صبر برای early stopping
    EARLY_STOPPING_MIN_DELTA = 0.001  # حداقل بهبود برای early stopping
    
    # تنظیمات خاص هر دیتاست
    DATASET_CONFIGS = {
        'acm': {
            'node_types': ['paper', 'author', 'subject'],
            'edge_types': ['pa', 'ps'],
            'target_node': 'paper',
            'files': {
                'features': ['p_feat', 'a_feat'],
                'edges': ['pa', 'ps'],
                'labels': 'labels'
            }
        },
        'dblp': {
            'node_types': ['paper', 'author', 'conference', 'term'],
            'edge_types': ['pa', 'pc', 'pt'],
            'target_node': 'paper',
            'files': {
                'features': ['p_feat', 'a_feat', 'c_feat', 't_feat'],
                'edges': ['pa', 'pc', 'pt'],
                'labels': 'labels'
            }
        },
        'imdb': {
            'node_types': ['movie', 'actor', 'director'],
            'edge_types': ['ma', 'md'],
            'target_node': 'movie',
            'files': {
                'features': ['m_feat', 'a_feat', 'd_feat'],
                'edges': ['ma', 'md'],
                'labels': 'labels'
            }
        }
    }