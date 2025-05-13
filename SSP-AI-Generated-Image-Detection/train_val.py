import os
import torch
from utils.util import set_random_seed, poly_lr

from utils.datasets import create_dataloader, get_dataset

from torch.utils.data import random_split, DataLoader
from options import TrainOptions
from networks.ssp import ssp
from utils.loss import bceLoss
from datetime import datetime
from tensorboardX import SummaryWriter
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train(train_loader, model, optimizer, epoch, save_path, writer, total_step):
    model.train()
    global step
    epoch_step = 0
    loss_all = 0
    all_preds = []
    all_labels = []
    try:
        for i, (images, labels) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            images = images.cuda() if torch.cuda.is_available() else images
            labels = labels.cuda() if torch.cuda.is_available() else labels
            if torch.backends.mps.is_available():
                images = images.to(torch.device("mps"))
                labels = labels.to(torch.device("mps"))
            preds = model(images).ravel()
            loss_fn = bceLoss()
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data
            
            pred_labels = (torch.sigmoid(preds) > 0.5).float()
            all_preds.append(pred_labels)
            all_labels.append(labels)
            
            if i % 20 == 0 or i == total_step or i == 1:
                print(
                    f'{datetime.now()} Epoch [{epoch:03d}/{opt.nepoch:03d}], Step [{i:04d}/{total_step:04d}], Total_loss: {loss.data:.4f}')
        
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        train_accuracy = (all_preds == all_labels).float().mean().item()
        
        loss_all /= epoch_step
        writer.add_scalar('train/loss', loss_all, epoch)
        writer.add_scalar('train/accuracy', train_accuracy, epoch)
        print(f'Training accuracy: {train_accuracy:.4f}')
        
        if epoch % 5 == 0:
            torch.save(model.state_dict(), save_path + f'Net_epoch_{epoch}.pth')

    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')


def validate(val_loader, model, epoch, writer):
    model.eval()
    val_loss_all = 0
    all_preds = []
    all_labels = []
    loss_fn = bceLoss()
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader, start=1):
            if torch.backends.mps.is_available():
                images = images.to(torch.device("mps"))
                labels = labels.to(torch.device("mps"))
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            preds = model(images).ravel()
            loss = loss_fn(preds, labels)
            val_loss_all += loss.item()
            
            pred_labels = (torch.sigmoid(preds) > 0.5).float()
            all_preds.append(pred_labels)
            all_labels.append(labels)
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    val_accuracy = (all_preds == all_labels).float().mean().item()
    val_loss_avg = val_loss_all / len(val_loader)
    
    writer.add_scalar('val/loss', val_loss_avg, epoch)
    writer.add_scalar('val/accuracy', val_accuracy, epoch)
    
    print(f'Validation Loss: {val_loss_avg:.4f}, Accuracy: {val_accuracy:.4f}')
    return val_accuracy


if __name__ == '__main__':
    set_random_seed()
    opt = TrainOptions().parse()
    
    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    log_dir = os.path.join(save_path, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)
    
    # Get dataset
    print(f"Loading dataset from {opt.dataset_root or opt.image_root}...")
    try:
        full_dataset = get_dataset(opt)
        print(f"Dataset loaded with {len(full_dataset)} images.")
    except Exception as e:
        print(f"Error with get_dataset: {e}")
        try:
            dataloader = create_dataloader(opt)
            full_dataset = dataloader.dataset
            print(f"Dataset loaded with {len(full_dataset)} images.")
        except Exception as e:
            print(f"Error with create_dataloader: {e}")
            exit(1)
    
    # Split dataset
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=opt.batchsize,
        shuffle=True,
        num_workers=min(4, opt.num_workers),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=opt.batchsize,
        shuffle=False,
        num_workers=min(4, opt.num_workers),
        pin_memory=True
    )
    
    total_step = len(train_loader)
    
    # Initialize model
    model = ssp().cuda() if torch.cuda.is_available() else ssp() 
    if torch.backends.mps.is_available():
        print("Using MPS for training.")
        model = model.to(torch.device("mps"))
    if opt.load is not None:
        model.load_state_dict(torch.load(opt.load))
        print('load model from', opt.load)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    
    # Training variables
    step = 0
    best_epoch = 0
    best_val_acc = 0
    
    print(f"Start training with {train_size} training samples and {val_size} validation samples")
    print(f"Training parameters: BatchSize={opt.batchsize}, LR={opt.lr}, CropSize={opt.cropSize}, LoadSize={opt.loadSize}")
    print(f"Augmentations: BlurProb={opt.blur_prob}, JpgProb={opt.jpg_prob}")
    
    for epoch in range(1, opt.nepoch + 1):
        cur_lr = poly_lr(optimizer, opt.lr, epoch, opt.nepoch)
        train(train_loader, model, optimizer, epoch, save_path, writer, total_step)
        val_acc = validate(val_loader, model, epoch, writer)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
            print(f'Save state_dict successfully! Best epoch:{epoch}, Accuracy:{val_acc:.4f}')
        
        print(f'Epoch:{epoch}, Val Accuracy:{val_acc:.4f}, Best epoch:{best_epoch}, Best Accuracy:{best_val_acc:.4f}')
    
    writer.close()