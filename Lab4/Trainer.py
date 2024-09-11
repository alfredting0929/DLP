import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import matplotlib.pyplot as plt
from math import log10

def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= batch_size  
  return KLD


class kl_annealing():
    def __init__(self, args, current_epoch=0):
        # TODO
        self.args = args
        self.current_epoch = current_epoch
        self.beta = 0

        if self.args.kl_anneal_type == 'Cyclical':
            self.beta_schedule = self.frange_cycle_linear(self.args.num_epoch, start=0.0, stop=1.0,
                                                         n_cycle=self.args.kl_anneal_cycle, ratio=0.5)
        elif self.args.kl_anneal_type == 'Monotonic':
            self.beta_schedule = np.linspace(0, 1, self.args.num_epoch)
        else:
            raise ValueError('Unsupported KL annealing type.')
        
    def update(self):
        # TODO
        if self.current_epoch < len(self.beta_schedule):
            self.beta = self.beta_schedule[self.current_epoch]
        else:
            self.beta = self.beta_schedule[-1]
        self.current_epoch += 1
    
    def get_beta(self):
        # TODO
        return self.beta

    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0,  n_cycle=1, ratio=1):
        # TODO
        L = np.ones(n_iter) * stop
        period = n_iter/n_cycle
        step = (stop-start)/(period*ratio) # linear schedule

        for c in range(n_cycle):
            v, i = start, 0
            while v <= stop and (int(i+c*period) < n_iter):
                L[int(i+c*period)] = v
                v += step
                i += 1
        return L 
        

class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.optim      = optim.Adamax(self.parameters(), lr=self.args.lr)
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[60, 100], gamma=0.1)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size

        self.writer = SummaryWriter()
        
        
    def forward(self, img, label):
        pass
    
    def training_stage(self):
        for i in range(100, self.args.num_epoch):
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False
            beta = self.kl_annealing.get_beta()

            for (img, label) in (pbar := tqdm(train_loader, ncols=180)):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                mse_loss, kl_loss = self.training_one_step(img, label, adapt_TeacherForcing)
                if adapt_TeacherForcing:
                    self.train_tqdm_bar('Train [TeacherForcing: ON, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, mse_loss.detach().cpu(), kl_loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                else:
                    self.train_tqdm_bar('Train [TeacherForcing: OFF, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, mse_loss.detach().cpu(), kl_loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])

            if self.current_epoch % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root, f"ckpt/epoch={self.current_epoch}.ckpt"))
            
            self.writer.add_scalar('Train/MSE_Loss', mse_loss, i)
            self.writer.add_scalar('Train/KL_Loss', kl_loss, i)
            self.writer.add_scalar('Train/Total_Loss', mse_loss + beta * kl_loss, i)
            self.writer.add_scalar('TFR', self.tfr, i)

            self.eval()
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()
            
            
    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        for (img, label) in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            mse_loss, psnr_list = self.val_one_step(img, label)
            self.val_tqdm_bar('Val', pbar, mse_loss.detach().cpu(), np.mean(psnr_list), lr=self.scheduler.get_last_lr()[0])

        self.writer.add_scalar('Val/MSE_Loss', mse_loss, self.current_epoch)
        self.writer.add_scalar('Val/PSNR', np.mean(psnr_list), self.current_epoch)
    
    def training_one_step(self, img, label, adapt_TeacherForcing=0):
        # TODO
        self.optim.zero_grad()
        # Changing dimension -> (seq, batch, channel, height, width)
        img = img.permute(1, 0, 2, 3, 4) 
        label = label.permute(1, 0, 2, 3, 4)

        kl_divergence = 0
        mse_loss = 0
        total_loss = 0
        pred_frame = [img[0]]
        beta = self.kl_annealing.get_beta()
        weight_ratio = 1.1

        for frame in range(1, self.train_vi_len):
            # frame, label transformation
            img_en = self.frame_transformation(img[frame])
            label_en = self.label_transformation(label[frame])
            z, mu, logvar = self.Gaussian_Predictor(img_en, label_en)

            if adapt_TeacherForcing:
                x = img[frame - 1]
            else:
                x = pred_frame[-1]
            x = self.frame_transformation(x)
            # print(x.size(), label_en.size(), z.size())
            decoder_output = self.Decoder_Fusion(x, label_en, z)
            pred = self.Generator(decoder_output)

            # Compute loss
            kl_divergence += kl_criterion(mu, logvar, self.batch_size)
            mse_loss += self.mse_criterion(pred, img[frame])
            if frame % 4 == 0:
                total_loss += weight_ratio * (self.mse_criterion(pred, img[frame]) + beta * kl_criterion(mu, logvar, self.batch_size))
                weight_ratio *= weight_ratio
            else:
                total_loss += self.mse_criterion(pred, img[frame]) + beta * kl_criterion(mu, logvar, self.batch_size)
            pred_frame.append(pred)

        # Back propagation
        total_loss.backward()
        self.optimizer_step()

        return mse_loss, kl_divergence
    
    def val_one_step(self, img, label):
        # TODO
        # Changing dimension -> (seq, batch, channel, height, width)
        img = img.permute(1, 0, 2, 3, 4) 
        label = label.permute(1, 0, 2, 3, 4)

        decoded_frame_list = [img[0]]
        psnr_list = []
        mse_list = []
        mse_loss = 0

        for frame in range(1, self.val_vi_len):
            x = self.frame_transformation(decoded_frame_list[-1])
            p = self.label_transformation(label[frame])
            z = torch.randn(1, self.args.N_dim, self.args.frame_H, self.args.frame_W).to(self.args.device)
            # print(x.size(), p.size(), z.size())
            decoder_output = self.Decoder_Fusion(x, p, z)
            pred = self.Generator(decoder_output)
            mse_list.append(self.mse_criterion(pred, img[frame]).cpu())
            mse_loss += mse_list[-1]

            psnr = Generate_PSNR(pred, img[frame])
            psnr_list.append(psnr.cpu())

            decoded_frame_list.append(pred)

        # Plotting PSNR
        plt.clf()
        fig = plt.figure()
        plt.plot(torch.linspace(1, len(psnr_list), len(psnr_list)), psnr_list, label=f'Avg_PSNR: {np.mean(psnr_list):.3f}')
        plt.xlabel('Frame index')
        plt.ylabel('PSNR')
        plt.ylim((10, 40))
        plt.title(f'Per frame Quality (PSNR) Epoch:{self.current_epoch}')
        plt.legend(loc='upper right')
        # plt.savefig(f'../his/PSNR/PSNR_{self.current_epoch}.png')
        self.writer.add_figure(f'Frame/PSNR{self.current_epoch}', fig)
        
        plt.clf()
        fig = plt.figure()
        plt.plot(torch.linspace(1, len(mse_list), len(mse_list)), mse_list)
        plt.xlabel('Frame index')
        plt.ylabel('MSE Loss')
        plt.title(f'Loss Epoch:{self.current_epoch}')
        self.writer.add_figure(f'Frame/Loss{self.current_epoch}', fig)

        generated_frame = stack(decoded_frame_list).permute(1, 0, 2, 3, 4)
        # print(generated_frame.size())
        self.make_gif(generated_frame[0], os.path.join(self.args.save_root, f'pred/{self.current_epoch}.gif'))

        return mse_loss, psnr_list


    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        # TODO
        if self.current_epoch > self.args.tfr_sde:
            self.tfr = self.tfr * self.tfr_d_step
            
    def train_tqdm_bar(self, mode, pbar, mse_loss, kl_loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(MSE_loss=float(mse_loss), KL_loss=float(kl_loss), refresh=False)
        pbar.refresh()
            
    def val_tqdm_bar(self, mode, pbar, mse_loss, psnr, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(MSE_loss=float(mse_loss), PSNR=psnr, refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = 1e-4
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.Adamax(self.parameters(), lr=self.args.lr)
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[20, 60], gamma=0.1)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch'] + 1

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()



def main(args):
    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=100,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.9,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=10,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=1,              help="")
    

    

    args = parser.parse_args()
    main(args)
