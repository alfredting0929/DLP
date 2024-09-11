import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from torch.optim import RAdam
from torchvision import transforms
import numpy as np

from oxford_pet import *
from utils import *
from evaluate import *
from models.unet import UNet
from models.resnet34_unet import ResNet34UNet

def train(args):
    # implement the training function here    
    T = {'image':transforms.Compose([transforms.Resize((256,256)),
                                     transforms.RandomVerticalFlip(),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])]),
         'mask':transforms.Compose([transforms.Resize((256,256), interpolation=Image.NEAREST),
                                    transforms.RandomVerticalFlip(),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor()])}

    test_dataset = load_dataset(data_path=args.data_path, mode='test', transform={'image':transforms.Compose([transforms.Resize((256,256)),
                                                                                                              transforms.ToTensor(),
                                                                                                              transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])]),
                                                                                  'mask':transforms.Compose([transforms.Resize((256,256)),
                                                                                                             transforms.ToTensor()])})
    train_dataset = load_dataset(data_path=args.data_path, mode='train', transform=T)
    valid_dataset = load_dataset(data_path=args.data_path, mode='valid', transform={'image':transforms.Compose([transforms.Resize((256,256)),
                                                                                                                transforms.ToTensor(),
                                                                                                                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])]),
                                                                                    'mask':transforms.Compose([transforms.Resize((256,256)),
                                                                                                               transforms.ToTensor()])})
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size)
  
    if args.model == 'unet':
        model = UNet() 
    else:
        model = ResNet34UNet()

    model = model.cuda()
    criterion = nn.BCELoss()
    optimizer = RAdam(model.parameters(), lr=args.learning_rate)

    train_loss_his, train_score_his, valid_score_his, valid_loss_his = [], [], [], []
    best_score = 0

    print('Training')
    for epoch in range(1, args.epochs+1):
        # if epoch == 25:
        #     optimizer = Adam(model.parameters(), lr=args.learning_rate/10)
        model.train()
        train_loss = 0
        train_dice = 0
        total = 0
        for data in tqdm(train_loader, desc=f'Epoch {epoch}'):
            x = data['image'].cuda()
            y = data['mask'].cuda()          

            optimizer.zero_grad()
            # Foward
            pred_mask = model(x)
            # Compute BCE loss
            loss = criterion(pred_mask, y)
            # Backward
            loss.backward()
            # Update
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            total += x.size(0)
            train_dice += dice_score(pred_mask, y) * x.size(0)
            
        # Validation
        val_loss, val_dice = evaluate(model, valid_loader, criterion)

        # Save model for best score
        if (val_dice) > best_score:
            best_score = val_dice
            print(f'Best valid score: {best_score}, model saved')
            torch.save(model, f'../saved_models/{args.model}.pth')

        print(f'Training loss: {train_loss/total:.4f}    Training dice score: {train_dice/total:.4f}     Valid loss: {val_loss:.4f}     Valid dice score: {val_dice:.4f}')

        train_loss_his.append(train_loss/total)
        train_score_his.append(train_dice/total)
        valid_score_his.append(val_dice)
        valid_loss_his.append(val_loss)

        np.save(f'../history/{args.model}_valid_score.npy', np.array(valid_score_his))
        np.save(f'../history/{args.model}_train_loss.npy', np.array(train_loss_his))
        np.save(f'../history/{args.model}_train_score.npy', np.array(train_score_his))
        np.save(f'../history/{args.model}_valid_loss.npy', np.array(valid_loss_his))

        model.eval()
        with torch.no_grad():
            data = test_dataset[0]
            image = data['image'].cuda()
            gt = data['mask'].cuda()
            pred = model(image.unsqueeze(0))
            pred = torch.round(pred)
            mean = torch.tensor([0.485,0.456,0.406])
            std = torch.tensor([0.229,0.224,0.225])
            mean = mean[:, None, None]
            std = std[:, None, None]
            
            reverse = (image.squeeze(0).detach().cpu() * std) + mean
            plt.subplot(131)
            plt.imshow(reverse.permute(1,2,0).numpy())
            plt.title('image')
            plt.subplot(132)
            plt.imshow(pred.squeeze(0).permute(1,2,0).detach().cpu().numpy())
            plt.title('pred')
            plt.subplot(133)
            plt.title('gt')
            plt.imshow(gt.permute(1,2,0).detach().cpu().numpy())
            plt.savefig(f'../history/pics/{epoch}.png')

        print('-'*80)



        
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--model', '-m', type=str, help='choose the model')

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    train(args)

