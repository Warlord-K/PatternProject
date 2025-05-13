import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler
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
                self.samples.append((file_path, row['label']))
                self.targets.append(row['label'])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class FsdModel(nn.Module):
    def __init__(self, out_dim=1024):
        super().__init__()
        rn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.enc = nn.Sequential(*list(rn.children())[:-1])
        in_feat = rn.fc.in_features
        self.proj = nn.Linear(in_feat, out_dim)

    def forward(self, x):
        emb = self.enc(x)
        emb = emb.view(emb.size(0), -1)
        emb = self.proj(emb)
        return emb

class EpisodicBatchSampler(Sampler):
    def __init__(self, ds_labels, n_cls, n_spt, n_qry, iterations):
        self.ds_labels = ds_labels
        self.n_cls = n_cls
        self.n_spt = n_spt
        self.n_qry = n_qry
        self.iterations = iterations

        self.cls_indices = defaultdict(list)
        for idx, lbl in enumerate(self.ds_labels):
            self.cls_indices[lbl].append(idx)
        self.avail_cls = list(self.cls_indices.keys())

        if n_cls > len(self.avail_cls):
             raise ValueError("n_cls > number of classes in dataset")
        for c in self.avail_cls:
             if len(self.cls_indices[c]) < n_spt + n_qry:
                  print(f"Warning: Class {c} has only {len(self.cls_indices[c])} samples, less than n_spt+n_qry={n_spt+n_qry}")

    def __iter__(self):
        for _ in range(self.iterations):
            sel_cls = random.sample(self.avail_cls, self.n_cls)
            batch_idxs = []
            for c in sel_cls:
                cls_idxs = self.cls_indices[c]
                req_samples = self.n_spt + self.n_qry
                if len(cls_idxs) < req_samples:
                    sampled = random.choices(cls_idxs, k=req_samples)
                else:
                    sampled = random.sample(cls_idxs, req_samples)
                batch_idxs.extend(sampled)
            yield batch_idxs

    def __len__(self):
        return self.iterations

def euclidean_dist(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise ValueError("Incompatible dimensions")
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)

def train_proto(model, dl, opt, n_spt, n_qry, n_cls, dev):
    model.train()
    tot_loss = 0.0
    tot_acc = 0.0
    crit = nn.NLLLoss()

    for batch in tqdm(dl):
        opt.zero_grad()
        data, _ = batch
        data = data.to(dev)
        p = n_spt * n_cls
        data_spt, data_qry = data[:p], data[p:]

        emb = model(data_spt)
        emb_dim = emb.size(-1)
        proto = emb.view(n_cls, n_spt, emb_dim).mean(dim=1)

        qry_emb = model(data_qry)
        dists = euclidean_dist(qry_emb, proto)

        log_p_y = nn.functional.log_softmax(-dists, dim=1)

        lbl_qry = torch.arange(n_cls).view(n_cls, 1).expand(n_cls, n_qry).reshape(-1)
        lbl_qry = lbl_qry.to(dev)

        loss = crit(log_p_y, lbl_qry)
        loss.backward()
        opt.step()

        _, y_hat = log_p_y.max(1)
        acc = torch.eq(y_hat, lbl_qry).float().mean()

        tot_loss += loss.item()
        tot_acc += acc.item()

    return tot_loss / len(dl), tot_acc / len(dl) * 100

def test_proto(model, ds, n_way, k_shot, n_query_test, n_episodes, dev):
    model.eval()
    tot_acc = 0.0
    cls_indices = defaultdict(list)
    for idx, (_, lbl) in enumerate(ds.samples):
        cls_indices[lbl].append(idx)
    avail_cls = list(cls_indices.keys())

    if n_way > len(avail_cls):
        raise ValueError("n_way > number of classes")

    with torch.no_grad():
        for _ in range(n_episodes):
            sel_cls_idx = random.sample(range(len(avail_cls)), n_way)
            sel_cls = [avail_cls[i] for i in sel_cls_idx]

            spt_idxs = []
            qry_idxs = []
            for i, c in enumerate(sel_cls):
                cls_idxs = cls_indices[c]
                req_samples = k_shot + n_query_test
                if len(cls_idxs) < req_samples:
                    sampled = random.choices(cls_idxs, k=req_samples)
                else:
                    sampled = random.sample(cls_idxs, req_samples)
                spt_idxs.extend(sampled[:k_shot])
                qry_idxs.extend(sampled[k_shot:])

            spt_data = torch.stack([ds[i][0] for i in spt_idxs]).to(dev)
            qry_data = torch.stack([ds[i][0] for i in qry_idxs]).to(dev)

            emb_spt = model(spt_data)
            emb_dim = emb_spt.size(-1)
            proto = emb_spt.view(n_way, k_shot, emb_dim).mean(dim=1)

            emb_qry = model(qry_data)
            dists = euclidean_dist(emb_qry, proto)
            log_p_y = nn.functional.log_softmax(-dists, dim=1)

            lbl_qry = torch.arange(n_way).view(n_way, 1).expand(n_way, n_query_test).reshape(-1)
            lbl_qry = lbl_qry.to(dev)

            _, y_hat = log_p_y.max(1)
            acc = torch.eq(y_hat, lbl_qry).float().mean()
            tot_acc += acc.item()

    return tot_acc / n_episodes * 100

def split_dataset(dataset, train_size=8000, test_size=800, random_seed=42):
    random.seed(random_seed)
    
    class_indices = {0: [], 1: []}
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)
    
    train_indices = []
    test_indices = []
    
    for label, indices in class_indices.items():
        random.shuffle(indices)
        train_per_class = train_size // 2
        test_per_class = test_size // 2
        
        if len(indices) < train_per_class + test_per_class:
            raise ValueError(f"Not enough samples for class {label}: {len(indices)} < {train_per_class + test_per_class}")
        
        train_indices.extend(indices[:train_per_class])
        test_indices.extend(indices[train_per_class:train_per_class + test_per_class])
    
    from torch.utils.data import Subset
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)
    
    return train_subset, test_subset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='/Users/yatharthgupta/vscode/PatternProject/data/4/train.csv', help='path to csv file')
    parser.add_argument('--data_root', type=str, default='/Users/yatharthgupta/vscode/PatternProject/data/4', help='path to image data root')
    parser.add_argument('--save_path', type=str, default='./fsd.pth', help='path to save model')
    parser.add_argument('--train_size', type=int, default=8000, help='number of training images')
    parser.add_argument('--test_size', type=int, default=800, help='number of test images')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    return parser.parse_args()

def main():
    args = parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    n_way_train = 2
    k_shot_train = 5
    n_query_train = 5
    train_iterations = 1000
    epochs = args.epochs
    
    n_way_test = 2
    k_shot_test = 10
    n_query_test = 15
    test_episodes = 200
    
    lr = args.lr
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        dev = torch.device("mps")
    print(f"Using device: {dev}")
    
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
    
    print(f"Loading data from {args.csv_path}...")
    full_dataset = CSVImageDataset(args.csv_path, args.data_root, transform=tf_train)
    
    print(f"Dataset loaded with {len(full_dataset)} images")
    print(f"Class distribution: {sum(1 for _, lbl in full_dataset.samples if lbl == 0)} AI-generated, "
          f"{sum(1 for _, lbl in full_dataset.samples if lbl == 1)} Real")
    
    train_ds, test_ds = split_dataset(full_dataset, args.train_size, args.test_size, args.seed)
    
    print(f"Split into {len(train_ds)} training and {len(test_ds)} test samples")
    
    train_sampler = EpisodicBatchSampler(
        [train_ds.dataset.samples[i][1] for i in train_ds.indices], 
        n_way_train, k_shot_train, n_query_train, train_iterations
    )
    
    train_batch_sz = n_way_train * (k_shot_train + n_query_train)
    train_dl = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=2)
    
    print("Initializing model...")
    mod = FsdModel(out_dim=1024).to(dev)
    opt = optim.Adam(mod.parameters(), lr=lr)
    sched = optim.lr_scheduler.StepLR(opt, step_size=80000, gamma=0.5)
    
    print("Starting training...")
    for ep in range(epochs):
        print(f"Epoch {ep+1}/{epochs}")
        avg_loss, avg_acc = train_proto(mod, train_dl, opt, k_shot_train, n_query_train, n_way_train, dev)
        
        print(f"Epoch {ep+1}/{epochs} -> Avg Loss: {avg_loss:.4f}, Avg Train Acc: {avg_acc:.2f}%")
        
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            test_acc = test_proto(mod, test_ds.dataset, n_way_test, k_shot_test, n_query_test, test_episodes, dev)
            print(f"  --> Test Acc ({n_way_test}-way {k_shot_test}-shot): {test_acc:.2f}%")
    
    print("Training finished.")
    final_test_acc = test_proto(mod, test_ds.dataset, n_way_test, k_shot_test, n_query_test, test_episodes, dev)
    print(f"Final Test Acc ({n_way_test}-way {k_shot_test}-shot): {final_test_acc:.2f}%")
    
    torch.save(mod.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")

if __name__ == "__main__":
    main()