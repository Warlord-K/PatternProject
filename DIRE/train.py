from utils.config import cfg  # isort: split

import os
import time
import numpy as np
import torch

from tensorboardX import SummaryWriter
from tqdm import tqdm

from utils.datasets import create_dataloader
from utils.trainer import Trainer
from utils.utils import Logger

if __name__ == "__main__":
    cfg.datasets = ["train"]
    
    data_loader = create_dataloader(cfg)
    dataset_size = len(data_loader)

    log = Logger()
    log.open(cfg.logs_path, mode="a")
    log.write("Num of training images = %d\n" % (dataset_size * cfg.batch_size))
    log.write("Config:\n" + str(cfg.to_dict()) + "\n")

    train_writer = SummaryWriter(os.path.join(cfg.exp_dir, "train"))

    trainer = Trainer(cfg)
    
    for epoch in range(cfg.nepoch):
        epoch_start_time = time.time()
        epoch_iter = 0
        epoch_losses = []
        all_preds = []
        all_labels = []

        trainer.train()
        for data in tqdm(data_loader, dynamic_ncols=True):
            trainer.total_steps += 1
            epoch_iter += cfg.batch_size

            trainer.set_input(data)
            trainer.optimize_parameters()
            
            # Store loss for epoch average
            epoch_losses.append(trainer.loss.item())
            
            # Get predictions and labels for accuracy calculation
            preds = trainer.get_current_predictions()
            labels = trainer.get_current_labels()
            
            all_preds.append(preds)
            all_labels.append(labels)

            # Log per-step loss
            train_writer.add_scalar("loss_step", trainer.loss.item(), trainer.total_steps)

            if trainer.total_steps % cfg.save_latest_freq == 0:
                log.write(
                    "saving the latest model %s (epoch %d, model.total_steps %d)\n"
                    % (cfg.exp_name, epoch, trainer.total_steps)
                )
                trainer.save_networks("latest")

        # Calculate epoch training metrics
        epoch_loss_avg = np.mean(epoch_losses)
        
        # Concatenate all predictions and labels
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Calculate accuracy
        correct = (all_preds == all_labels).sum().item()
        total = all_labels.size(0)
        accuracy = correct / total
        
        # Log epoch metrics
        train_writer.add_scalar("loss_epoch", epoch_loss_avg, epoch)
        train_writer.add_scalar("accuracy", accuracy, epoch)
        
        # Log to console
        log.write(f"[Epoch {epoch}/{cfg.nepoch}] Loss: {epoch_loss_avg:.4f}, Accuracy: {accuracy:.4f}\n")
        
        # Print epoch timing
        time_per_epoch = time.time() - epoch_start_time
        log.write(f"Time taken for epoch {epoch}: {time_per_epoch:.2f}s\n")

        # Save model periodically
        if epoch % cfg.save_epoch_freq == 0:
            log.write("saving the model at the end of epoch %d, iters %d\n" % (epoch, trainer.total_steps))
            trainer.save_networks("latest")
            trainer.save_networks(epoch)

        # Learning rate warmup if configured
        if cfg.warmup:
            trainer.scheduler.step()