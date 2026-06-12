import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .model import DeepFacialGAT
from .dataset import FacialLandmarkDataset
from .utils import FocalLoss
from sklearn.metrics import f1_score
import onnx

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for coords, imgs, labels in loader:
        coords, imgs, labels = coords.to(device), imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Vector Mixup augmentation
        alpha = 0.2
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0
            
        batch_size = coords.size(0)
        index = torch.randperm(batch_size, device=device)
        
        mixed_coords = lam * coords + (1 - lam) * coords[index]
        mixed_imgs = lam * imgs + (1 - lam) * imgs[index]
        labels_a, labels_b = labels, labels[index]
        
        outputs, _ = model(mixed_coords, mixed_imgs)
        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        
        loss.backward()
        # Gradient clipping to prevent exploding parameters in deep GAT layers
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item() * coords.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels_a).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total = 0
    
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for coords, imgs, labels in loader:
            coords, imgs, labels = coords.to(device), imgs.to(device), labels.to(device)
            outputs, _ = model(coords, imgs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * coords.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            
            all_targets.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
                
    val_loss = running_loss / total
    
    # Compute F1 scores
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    class_f1s = f1_score(all_targets, all_preds, average=None, zero_division=0)
    
    # Compute accuracy for reference
    correct = sum(1 for t, p in zip(all_targets, all_preds) if t == p)
    val_acc = correct / total
            
    return val_loss, val_acc, macro_f1, class_f1s

def main():
    parser = argparse.ArgumentParser(description="Facial Emotion Recognition GAT Training Loop")
    parser.add_argument("--epochs", type=int, default=350, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimensions of GAT layers")
    parser.add_argument("--depth", type=int, default=4, help="Number of stacked GAT layers")
    parser.add_argument("--dropout", type=float, default=0.15, help="Dropout rate between GAT layers")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="L2 regularization weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing preprocessed PyTorch data files")
    parser.add_argument("--regions-json", type=str, default="regions.json", help="Path to regions.json configuration file")
    parser.add_argument("--export-onnx", type=str, default="gcn_emotion.onnx", help="Path to export the final ONNX model")
    parser.add_argument("--use-focal-loss", action="store_true", default=True, help="Use class-balanced Focal Loss instead of CrossEntropyLoss")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Datasets
    train_path = os.path.join(args.data_dir, "train_data.pt")
    test_path = os.path.join(args.data_dir, "test_data.pt")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"[Error] Preprocessed data files not found in '{args.data_dir}'. Please run preprocess.py first.")
        return
        
    print("Loading datasets...")
    train_dataset = FacialLandmarkDataset(train_path, args.regions_json, is_train=True)
    test_dataset = FacialLandmarkDataset(test_path, args.regions_json, is_train=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=os.cpu_count() or 4, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=os.cpu_count() or 4, 
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # 2. Setup Model, Optimizer and Scheduler
    subset_indices = train_dataset.subset_indices
    model = DeepFacialGAT(
        num_nodes=len(subset_indices),
        hidden_dim=args.hidden_dim,
        num_classes=7,
        depth=args.depth
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total model parameters: {num_params:,}")
    
    # Identify parameter groups for split learning rates
    mobilenet_params = list(model.texture_net.parameters())
    mobilenet_param_ids = set(id(p) for p in mobilenet_params)
    other_params = [p for p in model.parameters() if id(p) not in mobilenet_param_ids]
    
    optimizer = optim.AdamW([
        {'params': mobilenet_params, 'lr': args.lr * 0.1}, # 1e-4 if args.lr is 1e-3
        {'params': other_params, 'lr': args.lr}            # 1e-3
    ], weight_decay=args.weight_decay)
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=120, 
        T_mult=1, 
        eta_min=1e-6
    )
    
    if args.use_focal_loss:
        print("Computing class weights dynamically for Focal Loss...")
        # Get ground-truth labels from training dataset
        train_labels = train_dataset.data['labels']
        # Compute dynamic class frequencies (number of classes = 7)
        class_counts = torch.bincount(train_labels, minlength=7).float()
        # Handle zero-count classes if any (to avoid division by zero)
        class_counts = torch.clamp(class_counts, min=1.0)
        # Balanced weights formula: total / (num_classes * class_count)
        class_weights = len(train_labels) / (7.0 * class_counts)
        class_weights = class_weights.to(device)
        
        # Display the computed weights for debugging
        emotions = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]
        print("Dynamic Class Weights for Loss balancing:")
        for name, w in zip(emotions, class_weights):
            print(f"  {name:<12}: {w.item():.4f}")
            
        criterion = FocalLoss(alpha=class_weights, gamma=2.2)
    else:
        print("Using standard CrossEntropyLoss...")
        criterion = nn.CrossEntropyLoss()
    
    # Emotion categories for printing
    emotions = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]
    
    # 4. Training Loop
    print("\nStarting GAT training...")
    best_test_f1 = 0.0
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, test_f1, class_f1s = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        
        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            torch.save(model.state_dict(), "model.pth")
            
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:02d}/{args.epochs:02d} | "
                  f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc*100:.2f}% | "
                  f"Test Loss: {test_loss:.4f} - Test F1: {test_f1*100:.2f}% (Best F1: {best_test_f1*100:.2f}%)")
            
    # 5. Final Evaluation and Details
    print(f"\nTraining completed. Loading best model state (Best Test F1: {best_test_f1*100:.2f}%)...")
    model.load_state_dict(torch.load("model.pth"))
    
    final_loss, final_acc, final_f1, final_class_f1s = evaluate(model, test_loader, criterion, device)
    
    print("\n" + "="*50)
    print("             FINAL EVALUATION REPORT")
    print("="*50)
    print(f"Overall Test Macro F1: {final_f1*100:.2f}%")
    print(f"Overall Test Accuracy: {final_acc*100:.2f}%")
    print(f"Overall Test Loss:     {final_loss:.4f}")
    print("-"*50)
    print("Class-wise F1-Score:")
    for i, name in enumerate(emotions):
        print(f"  {i+1}. {name:<12} : {final_class_f1s[i]*100:.2f}%")
    print("="*50)
    
    # 6. Export to ONNX
    if args.export_onnx:
        print(f"\nExporting GAT model to ONNX format: {args.export_onnx}")
        model.eval()
        # Create dummy inputs: coords (shape [1, len(subset_indices), 6]) and img (shape [1, 3, 224, 224])
        dummy_coords = torch.zeros((1, len(subset_indices), 6), device=device)
        dummy_img = torch.zeros((1, 3, 224, 224), device=device)
        
        try:
            torch.onnx.export(
                model,
                (dummy_coords, dummy_img),
                args.export_onnx,
                export_params=True,
                opset_version=18, # Use modern opset 18
                do_constant_folding=True,
                input_names=['input_coords', 'input_img'],
                output_names=['emotion_probabilities', 'attention_matrix'],
                dynamo=False, # Force legacy TorchScript exporter for maximum WebAssembly compatibility
                dynamic_axes={
                    'input_coords': {0: 'batch_size'},
                    'input_img': {0: 'batch_size'},
                    'emotion_probabilities': {0: 'batch_size'},
                    'attention_matrix': {0: 'batch_size'}
                }
            )
            
            # Load and save with ONNX library to merge external data
            onnx_model = onnx.load(args.export_onnx)
            onnx.save(onnx_model, args.export_onnx)
            
            print(f"Model successfully exported to self-contained ONNX: {os.path.abspath(args.export_onnx)}")
        except Exception as e:
            print(f"[Warning] ONNX export failed: {e}")

if __name__ == "__main__":
    main()
