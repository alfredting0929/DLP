import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.optim,self.scheduler = self.configure_optimizers()
        self.prepare_training()

    def prepare_training(self):
        if args.load != None:
            self.model.load_transformer_checkpoint(args.load)
        os.makedirs("transformer_checkpoints", exist_ok=True)

    def train_one_epoch(self, args, epoch, step, pbar, train_loader):
        pass

    def eval_one_epoch(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.transformer.parameters(), lr=args.learning_rate, betas=(0.9, 0.96))
        #scheduler = WarmupLinearLRSchedule(optimizer=optimizer, init_lr=1e-6, peak_lr=args.learning_rate, end_lr=0., warmup_epochs=10, epochs=args.epochs, current_step=args.start_from_epoch)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[40, 60], gamma=0.1)
        return optimizer,scheduler
    

class WarmupLinearLRSchedule:
    """
    Implements Warmup learning rate schedule until 'warmup_steps', going from 'init_lr' to 'peak_lr' for multiple optimizers.
    """
    def __init__(self, optimizer, init_lr, peak_lr, end_lr, warmup_epochs, epochs=100, current_step=0):
        self.init_lr = init_lr
        self.peak_lr = peak_lr
        self.optimizer = optimizer
        self.warmup_rate = (peak_lr - init_lr) / warmup_epochs
        self.decay_rate = (end_lr - peak_lr) / (epochs - warmup_epochs)
        self.update_steps = current_step
        self.lr = init_lr
        self.warmup_steps = warmup_epochs
        self.epochs = epochs
        if current_step > 0:
            self.lr = self.peak_lr + self.decay_rate * (current_step - 1 - warmup_epochs)

    def set_lr(self, lr):
        print(f"Setting lr: {lr}")
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def step(self):
        if self.update_steps <= self.warmup_steps:
            lr = self.init_lr + self.warmup_rate * self.update_steps
        # elif self.warmup_steps < self.update_steps <= self.epochs:
        else:
            lr = max(0., self.lr + self.decay_rate)
        self.set_lr(lr)
        self.lr = lr
        self.update_steps += 1
        return self.lr

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="../lab5_dataset/cat_face/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="../lab5_dataset/cat_face/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='start-from-epoch.')
    parser.add_argument('--learning-rate', type=float, default=0.4, help='Learning rate.')
    parser.add_argument('--load', type=str, default=None, help='Load weight')
    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
#TODO2 step1-5:    
    writer = SummaryWriter()
    # step = args.start_from_epoch * len(train_loader)
    for epoch in range(args.start_from_epoch+1, args.epochs+1):
        print(f'Epoch {epoch}:\nLearning rate: {train_transformer.optim.param_groups[0]['lr']}')
        total_loss = 0
        with tqdm(range(len(train_loader)), ncols=160) as pbar:
            for i, img in zip(pbar, train_loader):
                train_transformer.optim.zero_grad()
                img = img.to(args.device)
                logits, target = train_transformer.model(img)    
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
                loss.backward()
                # if step % args.accum_grad == 0:
                #     train_transformer.optim.step()  
                #     train_transformer.optim.zero_grad()
                train_transformer.optim.step()
                # step += 1
                pbar.set_postfix(Transformer_Loss=np.round(loss.cpu().detach().numpy().item(), 4))
                pbar.update(0)

                total_loss += loss.cpu().detach().item()
            train_transformer.scheduler.step()
            writer.add_scalar('Cross Entropy Loss', np.round(total_loss/len(train_loader), 4), epoch + args.start_from_epoch)
        
        total_val_loss = 0
        with tqdm(range(len(val_loader)), ncols=160) as pbar:
            for i, img in zip(pbar, val_loader):
                train_transformer.model.eval()
                with torch.no_grad():
                    img = img.to(args.device)
                    logits, target = train_transformer.model(img)
                    val_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
                    total_val_loss += val_loss
            total_val_loss = total_val_loss.cpu().detach().numpy().item()
            writer.add_scalar('Validation Loss', np.round(total_val_loss/len(train_loader), 4), epoch + args.start_from_epoch)
            
        torch.save(train_transformer.model.transformer.state_dict(), f'./transformer_checkpoints/{epoch}.pt')
