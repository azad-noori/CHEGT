from torch_geometric.data import HeteroData

def convert_to_aps(data):
    """
    تبدیل گراف ناهمگن به ساختار چندبخشی
    این تابع به صورت خودکار ساختار را حفظ می‌کند
    """
    aps_data = HeteroData()
    
    # کپی کردن تمام ویژگی‌ها و برچسب‌ها
    for node_type in data.node_types:
        aps_data[node_type].x = data[node_type].x
        if hasattr(data[node_type], 'y'):
            aps_data[node_type].y = data[node_type].y
        if hasattr(data[node_type], 'train_mask'):
            aps_data[node_type].train_mask = data[node_type].train_mask
        if hasattr(data[node_type], 'val_mask'):
            aps_data[node_type].val_mask = data[node_type].val_mask
        if hasattr(data[node_type], 'test_mask'):
            aps_data[node_type].test_mask = data[node_type].test_mask
    
    # کپی کردن تمام یال‌ها
    for edge_type in data.edge_types:
        aps_data[edge_type].edge_index = data[edge_type].edge_index
    
    return aps_data