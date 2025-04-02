import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms, datasets, models
from PIL import Image
import numpy as np
import os
import random
from collections import defaultdict
import math

class FsdModel(nn.Module):
    def __init__(self, out_dim=1024):
        super().__init__()
        rn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.enc = nn.Sequential(*list(rn.children())[:-1]) # Remove final fc
        in_feat = rn.fc.in_features # Should be 2048
        self.proj = nn.Linear(in_feat, out_dim) # Project to 1024

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
                # Ensure enough samples, sample with replacement if necessary
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

    for batch in dl:
        opt.zero_grad()
        data, _ = batch # Labels are implicit in the sampling order
        data = data.to(dev)
        p = n_spt * n_cls
        data_spt, data_qry = data[:p], data[p:]

        emb = model(data_spt)
        emb_dim = emb.size(-1)
        proto = emb.view(n_cls, n_spt, emb_dim).mean(dim=1)

        qry_emb = model(data_qry)
        dists = euclidean_dist(qry_emb, proto)

        log_p_y = nn.functional.log_softmax(-dists, dim=1)

        # Create query labels (0, 0, ..., 1, 1, ..., N-1, N-1, ...)
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


# --- Config ---
data_dir = './data' # CHANGE THIS to your dataset path (e.g., './genimage')
n_way_train = 3 # Nc in paper (Number of classes per episode)
k_shot_train = 5 # Ns in paper (Support samples per class)
n_query_train = 5 # Nq in paper (Query samples per class)
train_iterations = 1000 # Number of episodes per epoch
epochs = 10 # Number of epochs (each epoch runs train_iterations episodes)

n_way_test = 2 # N-way for testing (e.g., 2 for Real vs Fake_X)
k_shot_test = 10 # K-shot for testing
n_query_test = 15 # Query samples per class for testing
test_episodes = 600 # Number of test episodes

lr = 1e-4
# -------------

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

print("Loading data...")
try:
    train_ds = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=tf_train)
    test_ds = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=tf_test)

    if len(train_ds) == 0 or len(test_ds) == 0:
         raise ValueError("Datasets are empty. Check data_dir structure.")
    if len(train_ds.classes) <= n_way_train:
        raise ValueError(f"Need more than {n_way_train} classes in training set for episodic sampling.")

    print(f"Train classes: {train_ds.classes}")
    print(f"Test classes: {test_ds.classes}")

    train_sampler = EpisodicBatchSampler(train_ds.targets, n_way_train, k_shot_train, n_query_train, train_iterations)
    # Batch size must be n_way * (k_shot + n_query) for the sampler to work correctly
    train_batch_sz = n_way_train * (k_shot_train + n_query_train)
    train_dl = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=2)

except FileNotFoundError:
    print(f"Error: Data directory '{data_dir}' not found or incorrectly structured.")
    print("Ensure 'train' and 'test' subfolders exist, each with class subdirectories.")
    exit()
except ValueError as ve:
    print(f"Error: {ve}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during data loading: {e}")
    exit()

print("Initializing model...")
mod = FsdModel(out_dim=1024).to(dev)
opt = optim.Adam(mod.parameters(), lr=lr)
# Scheduler from paper: StepLR(gamma=0.5, step_size=80000) - apply per step/iteration, not epoch
sched = optim.lr_scheduler.StepLR(opt, step_size=80000, gamma=0.5)

print("Starting training...")
for ep in range(epochs):
    avg_loss, avg_acc = train_proto(mod, train_dl, opt, k_shot_train, n_query_train, n_way_train, dev)
    # Note: Scheduler step should ideally be called after each optimizer step within train_proto if step_size refers to iterations.
    # Calling it per epoch here for simplicity, adjust if needed based on paper's intent for 'step=80000'.
    # sched.step() # Uncomment if step refers to epochs

    print(f"Epoch {ep+1}/{epochs} -> Avg Loss: {avg_loss:.4f}, Avg Train Acc: {avg_acc:.2f}%")

    if (ep + 1) % 5 == 0: # Evaluate every 5 epochs
        test_acc = test_proto(mod, test_ds, n_way_test, k_shot_test, n_query_test, test_episodes, dev)
        print(f"  --> Test Acc ({n_way_test}-way {k_shot_test}-shot): {test_acc:.2f}%")


print("Training finished.")
final_test_acc = test_proto(mod, test_ds, n_way_test, k_shot_test, n_query_test, test_episodes, dev)
print(f"Final Test Acc ({n_way_test}-way {k_shot_test}-shot): {final_test_acc:.2f}%")

# torch.save(mod.state_dict(), 'fsd_model.pth')
