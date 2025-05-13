import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler, SubsetRandomSampler
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os
import random
import pandas as pd
from collections import defaultdict
import argparse
from tqdm import tqdm

class CSVImageDataset(Dataset):
    def __init__(self, csv_path, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform
        self.df = pd.read_csv(csv_path)
        self.samples = []
        self.targets = []
        self.class_to_idx = {0: 0, 1: 1}
        self.classes = ['AI-generated', 'Real']
        
        for _, row in self.df.iterrows():
            file_path = os.path.join(data_root, row['file_name'])
            if os.path.exists(file_path):
                self.samples.append((file_path, int(row['label'])))
                self.targets.append(int(row['label']))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class CustomSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
    
    def __len__(self):
        return len(self.indices)

class FsdModel(nn.Module):
    def __init__(self, out_dim=1024):
        super().__init__()
        rn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.enc = nn.Sequential(*list(rn.children())[:-1])
        in_feat = rn.fc.in_features
        self.proj = nn.Linear(in_feat, out_dim)
        
        self.classifier = nn.Linear(out_dim, 1)

    def forward(self, x, return_features=False):
        emb = self.enc(x)
        emb = emb.view(emb.size(0), -1)
        features = self.proj(emb)
        
        if return_features:
            return features
        
        logits = self.classifier(features)
        return logits.squeeze(-1)

def euclidean_dist(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise ValueError("Incompatible dimensions")
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, proto_loss_weight=0.5):
    best_val_acc = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training")
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device).float()  # Convert to float for BCE loss
            
            optimizer.zero_grad()
            
            # Forward pass with combined losses
            features = model(inputs, return_features=True)
            logits = model.classifier(features)
            logits = logits.squeeze(-1)
            
            # Standard binary classification loss
            cls_loss = criterion(logits, labels)
            
            # Calculate prototypical loss - only if batch has both classes
            if len(torch.unique(labels)) > 1:
                proto_loss = compute_prototypical_loss(features, labels, device)
                # Combine losses
                loss = cls_loss + proto_loss_weight * proto_loss
            else:
                # Only use classification loss if only one class in batch
                loss = cls_loss
            
            loss.backward()
            optimizer.step()
            
            # Track statistics
            running_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            corrects = torch.sum(preds == labels).item()
            running_corrects += corrects
            total_samples += inputs.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'batch_acc': corrects / inputs.size(0)
            })
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")
        
        # Validation phase
        if val_loader:
            val_acc = validate_model(model, val_loader, device)
            print(f"Validation Acc: {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"New best validation accuracy: {best_val_acc:.4f}")
        
        if scheduler:
            scheduler.step()
    
    return model

def compute_prototypical_loss(features, labels, device):
    """Compute prototypical network loss for binary classification"""
    class_indices = {0: [], 1: []}
    for i, label in enumerate(labels):
        class_indices[int(label.item())].append(i)
    
    # Make sure we have examples from both classes
    if len(class_indices[0]) == 0 or len(class_indices[1]) == 0:
        return torch.tensor(0.0, device=device)
    
    # Calculate prototypes for each class
    prototypes = {}
    for cls, indices in class_indices.items():
        if len(indices) > 0:
            prototypes[cls] = features[indices].mean(0, keepdim=True)
    
    # Convert prototypes to tensor
    proto_tensor = torch.cat([prototypes[0], prototypes[1]], dim=0)
    
    # Compute distances from each point to each prototype
    dists = euclidean_dist(features, proto_tensor)
    
    # Convert distances to probabilities with log-softmax
    log_p_y = torch.nn.functional.log_softmax(-dists, dim=1)
    
    # Create target indices (0 for class 0, 1 for class 1)
    target_inds = labels.long()
    
    # Compute NLL loss
    loss = torch.nn.functional.nll_loss(log_p_y, target_inds)
    
    return loss

def validate_model(model, val_loader, device):
    model.eval()
    running_corrects = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            inputs = inputs.to(device)
            labels = labels.to(device).float()
            
            logits = model(inputs)
            preds = (torch.sigmoid(logits) > 0.5).float()
            
            running_corrects += torch.sum(preds == labels).item()
            total_samples += inputs.size(0)
    
    acc = running_corrects / total_samples
    return acc

def split_dataset(dataset, train_size=8000, test_size=800, random_seed=42):
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Get indices for each class
    class_indices = {0: [], 1: []}
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)
    
    # Shuffle indices for each class
    for cls in class_indices:
        random.shuffle(class_indices[cls])
    
    # Calculate samples per class
    train_per_class = train_size // 2
    test_per_class = test_size // 2
    
    # Select indices for train and test sets
    train_indices = []
    test_indices = []
    
    for cls, indices in class_indices.items():
        if len(indices) < train_per_class + test_per_class:
            print(f"Warning: Not enough samples for class {cls}. Using {len(indices)} samples.")
            split_idx = int(0.9 * len(indices))
            train_indices.extend(indices[:split_idx])
            test_indices.extend(indices[split_idx:])
        else:
            train_indices.extend(indices[:train_per_class])
            test_indices.extend(indices[train_per_class:train_per_class + test_per_class])
    
    # Create train and test subsets
    train_dataset = CustomSubset(dataset, train_indices)
    test_dataset = CustomSubset(dataset, test_indices)
    
    return train_dataset, test_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='/Users/yatharthgupta/vscode/PatternProject/data/4/train.csv', help='path to csv file')
    parser.add_argument('--data_root', type=str, default='/Users/yatharthgupta/vscode/PatternProject/data/4', help='path to image data root')
    parser.add_argument('--save_path', type=str, default='./fsd.pth', help='path to save model')
    parser.add_argument('--train_size', type=int, default=8000, help='number of training images')
    parser.add_argument('--test_size', type=int, default=800, help='number of test images')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--proto_weight', type=float, default=0.5, help='weight for prototypical loss')
    parser.add_argument('--workers', type=int, default=4, help='number of worker threads')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")
    
    # Data transforms
    tf_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tf_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    print(f"Loading data from {args.csv_path}...")
    full_dataset = CSVImageDataset(args.csv_path, args.data_root, transform=tf_train)
    
    print(f"Dataset loaded with {len(full_dataset)} images")
    print(f"Class distribution: {sum(1 for _, lbl in full_dataset.samples if lbl == 0)} AI-generated, "
          f"{sum(1 for _, lbl in full_dataset.samples if lbl == 1)} Real")
    
    # Split dataset
    train_ds, test_ds = split_dataset(full_dataset, args.train_size, args.test_size, args.seed)
    
    print(f"Split into {len(train_ds)} training and {len(test_ds)} test samples")
    
    # Setup data loaders - avoid multiprocessing issues on macOS
    num_workers = 0 if device.type == 'mps' else args.workers
    
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Initialize model
    print("Initializing model...")
    model = FsdModel(out_dim=1024).to(device)
    
    # Loss function, optimizer and scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Train model
    print("Starting training...")
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        proto_loss_weight=args.proto_weight
    )
    
    # Final validation
    final_acc = validate_model(model, test_loader, device)
    print(f"Final test accuracy: {final_acc:.4f}")
    
    # Save model
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")

if __name__ == "__main__":
    main()