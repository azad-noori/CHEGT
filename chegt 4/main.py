import torch
import numpy as np
from data.datasets import load_dataset
from data.preprocessing import convert_to_aps
from models.model import ContrastiveHeteroModel
from models.classifier import Classifier
from utils.config import Config
from utils.train_eval import train_epoch, evaluate_model, evaluate_clustering, EarlyStopping

def main():
    config = Config()
    
    print(f"Loading {config.DATASET_NAME.upper()} dataset for Contrastive View Learning...")
    data = load_dataset(config.DATASET_NAME, config.DATASET_ROOT, config.SPLIT_RATIO)
    
    print("Converting to APS structure...")
    aps_data = convert_to_aps(data)
    
    print("Creating Contrastive Heterogeneous Model...")
    model = ContrastiveHeteroModel(
        hidden_dim=config.HIDDEN_DIM,
        out_dim=config.OUT_DIM,
        num_heads=config.NUM_HEADS,
        metadata=aps_data.metadata(),
        sampler_quantile=config.SAMPLER_QUANTILE,
        n_clusters=config.N_CLUSTERS,
        update_interval=config.UPDATE_INTERVAL
    )
    
    target_node = config.DATASET_CONFIGS[config.DATASET_NAME]['target_node']
    num_classes = aps_data[target_node].y.unique().numel()
    classifier = Classifier(config.OUT_DIM, num_classes)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    classifier = classifier.to(device)
    aps_data = aps_data.to(device)
    
    print("Starting Contrastive View Learning training...")
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(classifier.parameters()),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Early Stopping
    early_stopping = EarlyStopping(
        patience=15,  # صبر برای 15 epoch
        min_delta=0.001,  # حداقل بهبود 0.001
        restore_best_weights=True
    )
    
    best_val_acc = 0
    best_model_state = None
    best_epoch = 0
    
    for epoch in range(config.EPOCHS):
        # آموزش
        train_loss = train_epoch(model, aps_data, classifier, optimizer, device, config, epoch, target_node)
        
        # ارزیابی روی اعتبارسنجی
        val_acc, val_f1_macro, val_f1_micro, _, _ = evaluate_model(
            model, aps_data, classifier, device, 'val', target_node
        )
        
        # محاسبه loss برای early stopping (1 - accuracy)
        val_loss = 1 - val_acc
        
        # چاپ نتایج
        if epoch % 5 == 0:
            print(f'Epoch {epoch:03d}, Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}, '
                  f'Val F1-Macro: {val_f1_macro:.4f}, Val F1-Micro: {val_f1_micro:.4f}')
        
        # ذخیره بهترین مدل
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {
                'model': model.state_dict(),
                'classifier': classifier.state_dict()
            }
            best_epoch = epoch
        
        # Early Stopping
        if early_stopping(val_loss, model):
            print(f'\nEarly stopping at epoch {epoch}!')
            print(f'Best epoch was {best_epoch} with validation accuracy: {best_val_acc:.4f}')
            break
    
    # بارگذاری بهترین مدل
    if best_model_state:
        model.load_state_dict(best_model_state['model'])
        classifier.load_state_dict(best_model_state['classifier'])
        print(f"\nLoaded best model from epoch {best_epoch}")
    
    print("\n" + "="*50)
    print("FINAL EVALUATION")
    print("="*50)
    
    # ارزیابی روی مجموعه تست
    test_acc, test_f1_macro, test_f1_micro, test_embeddings, test_labels = evaluate_model(
        model, aps_data, classifier, device, 'test', target_node
    )
    
    print(f"\nClassification Metrics:")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1-Macro: {test_f1_macro:.4f}")
    print(f"Test F1-Micro: {test_f1_micro:.4f}")
    
    # ارزیابی خوشه‌بندی روی امبدینگ‌های تست
    print(f"\nClustering Metrics:")
    nmi, ari = evaluate_clustering(test_embeddings, test_labels, num_classes)
    print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    
    # محاسبه معیارهای کلیدی برای گزارش
    print(f"\nSummary:")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1-Macro: {test_f1_macro:.4f}")
    print(f"Test F1-Micro: {test_f1_micro:.4f}")
    print(f"Test NMI: {nmi:.4f}")
    print(f"Test ARI: {ari:.4f}")
    
    # ذخیره نتایج در فایل
    results = {
        'dataset': config.DATASET_NAME,
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_f1_macro': test_f1_macro,
        'test_f1_micro': test_f1_micro,
        'test_nmi': nmi,
        'test_ari': ari,
        'config': {
            'hidden_dim': config.HIDDEN_DIM,
            'out_dim': config.OUT_DIM,
            'num_heads': config.NUM_HEADS,
            'lr': config.LR,
            'epochs': config.EPOCHS,
            'sampler_quantile': config.SAMPLER_QUANTILE,
            'n_clusters': config.N_CLUSTERS,
            'update_interval': config.UPDATE_INTERVAL
        }
    }
    
    # ذخیره نتایج
    import json
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to 'results.json'")

if __name__ == "__main__":
    main()